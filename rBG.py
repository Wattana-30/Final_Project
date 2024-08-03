import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพ
image = cv2.imread('image_RGB_20240509_051944_1.jpg')

# แปลงภาพเป็นแบบ HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# กำหนดขอบเขตสีเขียวในภาพ HSV
lower_green = np.array([25, 40, 40])
upper_green = np.array([80, 255, 255])

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

# แสดงผลลัพธ์โดยใช้ matplotlib
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Masked Image")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Final Image")
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

plt.show()

# บันทึกผลลัพธ์
cv2.imwrite('masked_image.jpg', result)
cv2.imwrite('final_image.jpg', final_image)
