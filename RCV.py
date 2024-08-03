import cv2
import numpy as np
import matplotlib.pyplot as plt

# ฟังก์ชันสำหรับลบพื้นหลังและปรับพื้นหลังเป็นสีดำ
def remove_background(image):
    # แปลงภาพเป็นแบบ HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # กำหนดขอบเขตสีเขียวในภาพ HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # สร้างหน้ากาก (mask) สำหรับสีเขียว
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # ทำการกลับสีของหน้ากากเพื่อให้พื้นหลังเป็นสีขาวและวัตถุเป็นสีดำ
    mask_inv = cv2.bitwise_not(mask)

    # แยกใบไม้ออกจากภาพโดยใช้หน้ากาก
    result = cv2.bitwise_and(image, image, mask=mask)

    # สร้างพื้นหลังสีขาว
    white_background = np.full_like(image, 255)

    # รวมภาพใบไม้กับพื้นหลังสีขาว
    final_image = cv2.bitwise_or(result, white_background, mask=mask_inv)
    return result

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # แปลงภาพจาก BGR เป็น HSV
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)  # เพิ่มค่า brightness
    v = np.clip(v, 0, 255)  # ตรวจสอบว่าไม่เกินขอบเขต 0-255
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

image_path = f"E:/Project/data_image/captured_images_day1/00/image_RGB_20240509_054023_1.jpg"
image = cv2.imread(image_path)

if image is None:
    print("NOT Image")
else:
    image_re_bg = remove_background(image)
    img = cv2.cvtColor(image_re_bg, cv2.COLOR_BGR2HSV)
    bright_image = increase_brightness(img, value=40)

    points = {
        'base': (90, 190),
        'middle': (180, 230),
        'top': (320, 270)
    }
    colors = {}
    for point_name, (x, y) in points.items():
        color = bright_image[y, x]  # ค่าสีในรูปแบบ BGR
        color_rgb = (color[2], color[1], color[0])  # แปลงเป็น RGB
        colors[point_name] = color_rgb
        print(f"ค่าสีที่ {point_name} (RGB): {color_rgb}")

    # แสดงผลภาพพร้อมจุดที่อ่านค่าสีโดยใช้ matplotlib
    for point_name, (x, y) in points.items():
        cv2.circle(bright_image, (x, y), 5, (0, 0, 255), -1)  # วาดวงกลมสีแดงที่จุดที่อ่านค่าสี
        cv2.putText(bright_image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Points')
    plt.show()
