import cv2

# Test if xphoto module is available
if hasattr(cv2, 'xphoto'):
    print("xphoto is available")
else:
    print("xphoto is not available")

