import cv2
import numpy as np
from matplotlib import pyplot as plt

# อ่านภาพจากไฟล์ที่อัปโหลด
image_path = "E:/Project/data_image/captured_images_day1/00/image_RGB_20240509_052359_1.jpg"
image = cv2.imread(image_path)

if image is not None:
    # แปลงภาพจาก BGR เป็น HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงสีเขียวในช่องสี HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Thresholding เพื่อแยกสีเขียว
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # หา Contours ของบริเวณที่เป็นสีเขียว
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # ใช้ Contour ที่ใหญ่ที่สุดในการกำหนดกรอบการครอบภาพ
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # ขยายขอบเขตการครอบ
        padding = 20  # ขยายขอบเขตการครอบ 20 พิกเซลในแต่ละด้าน
        x = max(x - padding, 2)
        y = max(y - padding, 2)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)
        
        # ครอบภาพ
        cropped_image = image[y:y+h, x:x+w]
        
        # แสดงภาพที่ครอบ
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')
        plt.show()
        

    else:
        'ไม่พบใบในภาพ'
else:
    'ไม่สามารถอ่านภาพได้ ตรวจสอบ path ของไฟล์'
