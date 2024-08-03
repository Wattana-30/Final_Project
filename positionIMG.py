import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# โหลดภาพ
img = mpimg.imread('image_RGB_20240509_055700_1.jpg')

# สร้างกราฟ
fig, ax = plt.subplots()

# แสดงภาพในกราฟ
ax.imshow(img)

# ระบุตำแหน่งในภาพ (ตัวอย่างการระบุตำแหน่งจุด A และ B)
x_positions = [100, 300]
y_positions = [200, 400]
labels = ['A', 'B']

# วางตำแหน่งและเพิ่มฉลาก
for x, y, label in zip(x_positions, y_positions, labels):
    ax.scatter(x, y, color='red')  # ระบุตำแหน่งด้วยจุดสีแดง
    ax.text(x, y, label, color='white', fontsize=12, ha='right')  # เพิ่มฉลาก

# ซ่อนแกน
ax.axis('off')

# แสดงกราฟ
plt.show()
