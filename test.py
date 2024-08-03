import pandas as pd
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import cv2
import os

# ฟังก์ชันสำหรับลบพื้นหลังและปรับพื้นหลังเป็นสีดำ
def remove_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0, 45, 0])
    upper_green = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask)
    white_background = np.full_like(image, 255)
    final_image = cv2.bitwise_or(result, white_background, mask=mask_inv)
    return result

# ฟังก์ชันสำหรับเพิ่มความสว่างให้กับภาพ
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

# ฟังก์ชันสำหรับดึงคุณลักษณะของสีจากภาพ
def extract_color_features(image_array):
    mean_red = np.mean(image_array[:, :, 0])
    mean_green = np.mean(image_array[:, :, 1])
    mean_blue = np.mean(image_array[:, :, 2])

    std_red = np.std(image_array[:, :, 0])
    std_green = np.std(image_array[:, :, 1])
    std_blue = np.std(image_array[:, :, 2])

    image_lab = rgb2lab(image_array / 255.0)
    mean_l = np.mean(image_lab[:, :, 0])
    mean_a = np.mean(image_lab[:, :, 1])
    mean_b = np.mean(image_lab[:, :, 2])

    std_l = np.std(image_lab[:, :, 0])
    std_a = np.std(image_lab[:, :, 1])
    std_b = np.std(image_lab[:, :, 2])

    features = {
        'mean_red': mean_red, 'mean_green': mean_green, 'mean_blue': mean_blue,
        'std_red': std_red, 'std_green': std_green, 'std_blue': std_blue,
        'mean_l': mean_l, 'mean_a': mean_a, 'mean_b': mean_b,
        'std_l': std_l, 'std_a': std_a, 'std_b': std_b
    }

    return features

# ฟังก์ชันสำหรับดึง Color Ratios
def extract_color_ratios(image):
    total_pixels = image.shape[0] * image.shape[1]
    red_ratio = np.sum(image[:, :, 2] > 128) / total_pixels
    green_ratio = np.sum(image[:, :, 1] > 128) / total_pixels
    blue_ratio = np.sum(image[:, :, 0] > 128) / total_pixels
    return {'red_ratio': red_ratio, 'green_ratio': green_ratio, 'blue_ratio': blue_ratio}

# ฟังก์ชันหลักสำหรับประมวลผลหลายๆ ภาพในโฟลเดอร์
def process_images_from_folder(folder_path):
    all_features_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"ไม่พบภาพ: {filename}")
                continue

            image_re_bg = remove_background(image)
            bright_image = increase_brightness(image_re_bg, value=40)
            bright_image_pil = Image.fromarray(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))

            features = extract_color_features(np.array(bright_image_pil))
            color_ratios = extract_color_ratios(bright_image)

            # Combine all features into one dictionary
            all_features = {**features, **color_ratios, 'filename': filename}
            all_features_list.append(all_features)

            # Display the processed image
            plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
            plt.title(f'ภาพพร้อมจุดที่อ่านค่า: {filename}')
            plt.show()

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features_list)

    # Save to CSV
    csv_path = 'image_features.csv'
    features_df.to_csv(csv_path, index=False)

    print(f'Features saved to {csv_path}')
    print(features_df)

# Specify the folder containing the images
folder_path = 'E:\Project\data_image\captured_images_day1\00'
process_images_from_folder(folder_path)
