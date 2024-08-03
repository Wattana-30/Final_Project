# main.py

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from image_processing import cropped_image, remove_background, increase_brightness, extract_color_features, extract_color_ratios, extract_glcm_features, extract_retinex_features, compute_gradient
from data_processing import normalize_data
from file_operations import check_folder_exists, read_existing_csv, save_both_csv

def show_image_for_seconds(image, filename, display_time=0.5):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'ภาพพร้อมจุดที่อ่านค่า: {filename}')
    plt.show(block=False)
    plt.pause(display_time)  # แสดงภาพเป็นเวลา 3 วินาที
    plt.close()  # ปิดหน้าต่างภาพ

def process_images_from_folder(folder_path):
    if not check_folder_exists(folder_path):
        return

    csv_path_raw = 'image_features_raw.csv'
    csv_path_normalized = 'image_features_normalized.csv'
    existing_df = read_existing_csv(csv_path_raw)

    all_features_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"ไม่พบภาพ: {filename}")
                continue

            if filename in existing_df['filename'].values:
                print(f"ภาพ {filename} มีอยู่แล้วในไฟล์ CSV, ข้ามการประมวลผล")
                continue

            crop_img = cropped_image(image)
            image_re_bg = remove_background(crop_img)
            bright_image = increase_brightness(image_re_bg, value=40)
            bright_image_pil = Image.fromarray(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))

            features = extract_color_features(np.array(bright_image_pil))
            color_ratios = extract_color_ratios(bright_image)
            glcm_features = extract_glcm_features(bright_image)
            retinex_features = extract_retinex_features(bright_image)
            gradient_features = compute_gradient(bright_image)

            all_features = {**features, **color_ratios, **glcm_features, **retinex_features, **gradient_features, 'filename': filename}
            all_features_list.append(all_features)

            show_image_for_seconds(bright_image, filename)

    if all_features_list:
        new_features_df = pd.DataFrame(all_features_list)
        normalized_df = normalize_data(new_features_df)
        
        # รวมข้อมูลใหม่กับข้อมูลที่มีอยู่
        combined_df_raw = pd.concat([existing_df, new_features_df], ignore_index=True)
        combined_df_normalized = pd.concat([existing_df, normalized_df], ignore_index=True)
        
        # บันทึกข้อมูลที่รวมแล้ว
        save_both_csv(combined_df_raw, combined_df_normalized, csv_path_raw, csv_path_normalized)
    else:
        save_both_csv(existing_df, existing_df, csv_path_raw, csv_path_normalized)

if __name__ == '__main__':
    folder_path = r'E:\Project\data_image\captured_images_day1\200'
    process_images_from_folder(folder_path)
