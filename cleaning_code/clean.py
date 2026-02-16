import pandas as pd
import os
import glob

source_folder = './data/data/weather-PM2.5-data'
output_folder = 'CLEAN-weather-PM2.5-data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_files = glob.glob(os.path.join(source_folder, '*.csv'))

for file_path in csv_files:
    try:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        
        # เลือกคอลัมน์ที่ต้องการจัดการ (ไม่รวม Vis.)
        target_cols = [c for c in df.columns if c != 'Vis.']
        
        # 1. จัดการลบแถวที่ว่างเกิน 3 แถวติดกัน
        # สร้าง mask เพื่อดูว่าแถวไหนว่างบ้าง (ในคอลัมน์ที่เราสนใจ)
        is_na = df[target_cols].isna().any(axis=1)
        
        # คำนวณหาจำนวนแถวว่างที่ติดกัน (Consecutive NaNs)
        group_na = is_na.groupby((is_na != is_na.shift()).cumsum()).transform('size')
        
        # สร้าง Filter: ถ้าว่าง (is_na) และ จำนวนที่ติดกัน > 3 ให้ลบทิ้ง
        mask_to_drop = is_na & (group_na > 3)
        df = df[~mask_to_drop].reset_index(drop=True)
        
        # 2. ทำ Linear Interpolation สำหรับกรณีว่าง 1-3 แถว
        # หมายเหตุ: การทำ linear interpolation บน 1 แถว จะได้ค่าเฉลี่ย (Average) 
        # ระหว่างแถวก่อนหน้าและถัดไปโดยอัตโนมัติ ตามโจทย์ข้อ 1 และ 2
        df[target_cols] = df[target_cols].interpolate(method='linear', limit=3)
        
        # 3. ลบแถวที่ยังคงเหลือค่าว่างอยู่ (เช่น แถวที่อยู่บนสุดหรือล่างสุดของไฟล์ซึ่ง interpolate ไม่ได้)
        # หรือกรณีที่ว่างไม่เกิน 3 แต่ไม่มีข้อมูลหัวท้ายให้เฉลี่ย
        final_df = df.dropna(subset=target_cols, how='any')
        
        save_path = os.path.join(output_folder, file_name)
        final_df.to_csv(save_path, index=False)
        
        print(f"✅ จัดการเรียบร้อย: {file_name} (คงเหลือ {len(final_df)} แถว)")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file_name}: {e}")

print("\n--- เสร็จสิ้นการทำงาน ---")