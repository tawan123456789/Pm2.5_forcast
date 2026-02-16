import pandas as pd
import os
import glob

source_folder = './data/data/weather-PM2.5-data'
output_folder = 'CLEAN_BIGGEST_CHUNK-weather-PM2.5-data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_files = glob.glob(os.path.join(source_folder, '*.csv'))

MAX_GAP = 3   # อนุญาต NaN ต่อเนื่องได้ไม่เกินกี่แถว

for file_path in csv_files:
    try:
        
        file_name = os.path.basename(file_path)

        #try with 1 file
        if file_name != "weather-PM2.5-05T.csv":
            continue
        df = pd.read_csv(file_path)

        # -------------------------------
        # 1) สร้าง mask ว่า PM2.5 เป็น NaN หรือไม่
        # -------------------------------
        pm_nan = df['PM2.5'].isna()

        # นับจำนวน NaN ต่อเนื่อง
        nan_groups = pm_nan.ne(pm_nan.shift()).cumsum()
        nan_run_length = pm_nan.groupby(nan_groups).transform('sum')

        # ถ้าเป็น NaN และยาวเกิน MAX_GAP → ถือว่า break
        valid_mask = ~(pm_nan & (nan_run_length > MAX_GAP))

        # -------------------------------
        # 2) หา chunk ที่ valid ต่อเนื่องที่สุด
        # -------------------------------
        groups = valid_mask.ne(valid_mask.shift()).cumsum()
        df['__group'] = groups

        candidate_chunks = df[valid_mask].groupby('__group')

        if candidate_chunks.ngroups == 0:
            print(f"⚠️ ไม่มี chunk ที่เข้าเงื่อนไขในไฟล์ {file_name}")
            continue

        # เลือก chunk ที่ยาวที่สุด
        longest_group = candidate_chunks.size().idxmax()
        final_df = candidate_chunks.get_group(longest_group).drop(columns='__group')

        # -------------------------------
        # 3) Fill NaN ด้วย interpolate (between)
        # -------------------------------
        final_df = final_df.interpolate(
            method='linear',
            limit=MAX_GAP,
            limit_direction='both'
        )

        # -------------------------------
        # 4) Save file
        # -------------------------------
        output_path = os.path.join(output_folder, file_name)
        final_df.to_csv(output_path, index=False)

        print(f"✅ จัดการเรียบร้อย: {file_name} (คงเหลือ {len(final_df)} แถว)")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file_name}: {e}")

print("\n--- เสร็จสิ้นการทำงาน ---")
