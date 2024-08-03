# data_processing.py

import pandas as pd

def normalize_data(df):
    # แยกคอลัมน์ที่ไม่ใช่ตัวเลขออกจาก DataFrame
    df_non_numeric = df[['filename']]  # คงไว้เฉพาะคอลัมน์ 'filename'
    
    # ตรวจสอบชนิดข้อมูลของแต่ละคอลัมน์และแปลงคอลัมน์ที่เป็นสตริงให้เป็นตัวเลข
    df_numeric = df.select_dtypes(include=['number'])
    
    # ปรับค่าทั้งหมดให้อยู่ในช่วง 0-1
    df_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    
    # รวมคอลัมน์ที่ไม่เป็นตัวเลข (เช่น คอลัมน์ชื่อไฟล์) กลับเข้าไป
    df_normalized = pd.concat([df_non_numeric, df_normalized], axis=1)
    
    return df_normalized
