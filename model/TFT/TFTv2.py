import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

# --- 1. เตรียมข้อมูล ---
df = pd.read_csv("weather-PM2.5-44T.csv") # เปลี่ยนเป็นชื่อไฟล์จริง

df = df.rename(columns={
    "Temp.": "Temp",
    "Humi.": "Humi",
    "Pres.": "Pres",
    "Prec.": "Prec",
    "Vis.": "Vis",
    "PM2.5": "PM25"  # เปลี่ยนจาก PM2.5 เป็น PM25
})

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Feature Engineering (เหมือนเดิม)
df['month'] = df['Date'].dt.month
df['day_of_week'] = df['Date'].dt.dayofweek.astype(str)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# จัดการ Wind Dir (แปลงองศาเป็น Sin/Cos)
wind_rad = df['Wind Dir'] * np.pi / 180
df['wind_dir_sin'] = np.sin(wind_rad)
df['wind_dir_cos'] = np.cos(wind_rad)

# TFT Requirements
df['time_idx'] = np.arange(len(df))
df['group_id'] = "station_44T"  # ใช้ชื่อสถานีเป็น group_id

# --- 2. แบ่งกลุ่มตัวแปร (หัวใจสำคัญของ TFT) ---

# ตัวแปรที่รู้ล่วงหน้า (Known $X$): ข้อมูลเวลา/ปฏิทิน
known_reals = ["time_idx", "month_sin", "month_cos"]
known_categoricals = ["day_of_week"]

# ตัวแปรที่ไม่รู้ล่วงหน้า (Unknown $Z$): ค่า PM2.5 และสภาพอากาศที่วัดได้จริง
# (เพราะในอนาคตเราไม่รู้ค่าที่แท้จริงของ Temp, Wind Speed, ฯลฯ จนกว่าจะถึงวันนั้น)
unknown_reals = [
    "PM25",         # แก้ตรงนี้
    "Wind Speed", 
    "Temp",         # แก้ตรงนี้
    "Humi",         # แก้ตรงนี้
    "Pres",         # แก้ตรงนี้
    "Prec",         # แก้ตรงนี้
    # "Vis",          # แก้ตรงนี้
    "wind_dir_sin", 
    "wind_dir_cos"
]

# --- 3. สร้าง Dataset ---
max_encoder_length = 30 # ดูย้อนหลัง 30 step
max_prediction_length = 1 # ทายล่วงหน้า 1 step

training_cutoff = int(df["time_idx"].max() * 0.8)

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="PM25",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    
    static_categoricals=["group_id"],
    time_varying_known_categoricals=known_categoricals,
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=unknown_reals, # ใส่สภาพอากาศทั้งหมดไว้ที่นี่
    
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ), # Normalizes the target to stable range
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# สร้าง Validation Set (20% of data)
# Use the remaining data for validation. Ensure we have enough history for the first validation point.
validation_cutoff = training_cutoff
validation = TimeSeriesDataSet.from_dataset(
    training, 
    df, 
    min_prediction_idx=validation_cutoff + 1,
    stop_randomization=True
)

# สร้าง DataLoader
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# --- 4. สร้างโมเดล ---
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.2,              # increased from 0.1 to reduce overfitting
    hidden_continuous_size=32,
    output_size=7,            # 7 quantiles
    loss=QuantileLoss(),
    optimizer="adamw",        # AdamW with decoupled weight decay
    weight_decay=1e-2,        # L2 regularization to prevent overfitting
    reduce_on_plateau_patience=5,   # reduce LR if val_loss stalls for 5 epochs
    reduce_on_plateau_reduction=10, # divide LR by 10 on plateau
    reduce_on_plateau_min_lr=1e-6,  # minimum LR floor
)

# --- 5. Trainer and Training ---

# Configure CSVLogger
logger = CSVLogger("logs", name="44T_model_v2")

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", 
    mode="min",
    save_top_k=1,
    filename="best-checkpoint"
)
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=1e-4, 
    patience=20,  # increased: give scheduler time to reduce LR before stopping
    verbose=True, 
    mode="min"
)

trainer = Trainer(
    max_epochs=200, # more epochs — scheduler + early stopping will handle convergence
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    logger=logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    enable_progress_bar=True,
)

# Train the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Load best model for testing (if needed, though we track val loss)
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")

# --- 6. Plotting Loss History ---
# Load metrics from the CSV file generated by CSVLogger
metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
print(f"Loading metrics from: {metrics_path}")
metrics = pd.read_csv(metrics_path)

# Group by epoch to handle multiple steps per epoch if logged per step
# Taking the mean is a safe aggregation if multiple logs per epoch exist for step-based metrics, 
# but for epoch-based metrics, they are unique per epoch usually.
epoch_metrics = metrics.groupby("epoch").mean()

plt.figure(figsize=(10, 6))

# Plot train loss
# Check for 'train_loss_epoch' or 'train_loss' (depends on logging setup inside the LightningModule)
if 'train_loss_epoch' in epoch_metrics.columns:
    plt.plot(epoch_metrics.index, epoch_metrics['train_loss_epoch'], label='Train Loss')
elif 'train_loss' in epoch_metrics.columns:
    plt.plot(epoch_metrics.index, epoch_metrics['train_loss'], label='Train Loss')

# Plot validation loss
if 'val_loss' in epoch_metrics.columns:
    plt.plot(epoch_metrics.index, epoch_metrics['val_loss'], label='Test / Validation Loss')


plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
print("Loss plot saved to loss_plot.png")

# --- 7. Calculate R-squared on Test Set ---
print("Calculating R-squared on validation set...")
# Load the best model
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Predict on validation/test set
# return_y=True returns (predictions, x, y)
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)

# Predictions shape is (n_samples, prediction_length, n_quantiles) if output_size > 1 and quantiles enabled
# But .predict() by default returns mean prediction or a specific quantile if configured. 
# For TFT with QuantileLoss, .predict() usually returns the prediction for the central quantile (0.5).

# Convert to numpy
# Flatten if necessary, assuming 1-step prediction for now as per configuration
y_true = actuals.cpu().numpy().flatten()
y_pred = predictions.cpu().numpy().flatten()

print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_true contains NaNs: {np.isnan(y_true).any()}")
print(f"y_pred contains NaNs: {np.isnan(y_pred).any()}")

if np.isnan(y_pred).any():
    print("Predictions contain NaNs. This might be due to scaling issues or unstable training.")

r2 = r2_score(y_true, y_pred)
print(f"R-squared Score on Validation Set: {r2:.4f}")

try:
    plt.show()
except:
    pass