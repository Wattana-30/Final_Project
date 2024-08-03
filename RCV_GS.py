import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Tahoma'
rcParams['font.size'] = 10

# ฟังก์ชันสำหรับลบพื้นหลังและปรับพื้นหลังเป็นสีดำ
def remove_background(image):
    # แปลงภาพเป็น HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # กำหนดขอบเขตสีเขียวในภาพ HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # สร้างหน้ากาก (mask) สำหรับสีเขียว
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # ทำการกลับสีของหน้ากากเพื่อให้พื้นหลังเป็นสีดำและวัตถุเป็นสีขาว
    mask_inv = cv2.bitwise_not(mask)

    # แยกวัตถุออกจากภาพโดยใช้หน้ากาก
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # แปลงภาพจาก BGR เป็น HSV
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)  # เพิ่มความสว่าง
    v = np.clip(v, 0, 255)  # จำกัดค่าความสว่างให้อยู่ในช่วง 0-255
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

def normalize_to_0_1(img):
    return img.astype(float) / 255.0

image_path = 'image_RGB_20240509_065049_1.jpg'
image = cv2.imread(image_path)

if image is None:
    print("ไม่พบภาพ")
else:
    image_re_bg = remove_background(image)
    bright_image = increase_brightness(image_re_bg, value=40)

    # แปลงภาพเป็นสีเทา
    gray_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)

    # ปรับค่าภาพสีเทาให้อยู่ในช่วง 0-1
    gray_image_normalized = normalize_to_0_1(gray_image)

    points = {
        'base': (80, 220),
        'middle': (180, 230),
        'top': (330, 220)
    }
    gray_values = {}
    for point_name, (x, y) in points.items():
        gray_value = gray_image_normalized[y, x]  # อ่านค่าที่ตำแหน่ง
        gray_values[point_name] = gray_value
        print(f"ค่าภาพสีเทาที่ {point_name}: {gray_value:.2f}")

    # แสดงภาพพร้อมจุดที่อ่านค่าภาพสีเทาโดยใช้ matplotlib
    for point_name, (x, y) in points.items():
        cv2.circle(gray_image, (x, y), 5, (0, 0, 255), -1)  # วาดวงกลมสีแดงที่จุด
        cv2.putText(gray_image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    plt.title('ภาพพร้อมจุดที่อ่านค่า')
    plt.show()
