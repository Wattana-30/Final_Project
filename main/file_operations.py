# file_operations.py

import os
import pandas as pd

def check_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"ไม่พบโฟลเดอร์: {folder_path}")
        return False
    return True

def read_existing_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=['filename'])

def save_to_csv(df, csv_path):
    df.to_csv(csv_path, index=False)
    
def save_both_csv(df_raw, df_normalized, raw_csv_path, normalized_csv_path):
    save_to_csv(df_raw, raw_csv_path)
    save_to_csv(df_normalized, normalized_csv_path)
