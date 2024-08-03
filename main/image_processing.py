# image_processing.py

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.color import rgb2gray
from skimage import img_as_float
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


def cropped_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    padding = 20
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def remove_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask)
    white_background = np.full_like(image, 255)
    final_image = cv2.bitwise_or(result, white_background, mask=mask_inv)
    return result

def increase_brightness(img, value=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

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

def extract_color_ratios(image):
    total_pixels = image.shape[0] * image.shape[1]
    red_ratio = np.sum(image[:, :, 2] > 128) / total_pixels
    green_ratio = np.sum(image[:, :, 1] > 128) / total_pixels
    blue_ratio = np.sum(image[:, :, 0] > 128) / total_pixels
    return {'red_ratio': red_ratio, 'green_ratio': green_ratio, 'blue_ratio': blue_ratio}


# image_processing.py



def extract_glcm_features(image):
    # แปลงภาพเป็นโทนสีเทา
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # คำนวณ GLCM
    distances = [1, 2, 3]  # ระยะห่างที่ใช้ในการคำนวณ GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # มุมที่ใช้ในการคำนวณ GLCM
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # คำนวณคุณลักษณะจาก GLCM
    features = {
        'contrast': graycoprops(glcm, prop='contrast').mean(),
        'dissimilarity': graycoprops(glcm, prop='dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, prop='homogeneity').mean(),
        'energy': graycoprops(glcm, prop='energy').mean(),
        'correlation': graycoprops(glcm, prop='correlation').mean(),
        'ASM': graycoprops(glcm, prop='ASM').mean()  
    }
    
    return features


def retinex(image, sigma_list=[15, 80, 250]):
    """
    Perform Retinex on the input image with multiple Gaussian scales.
    """
    image = img_as_float(image)
    image_gray = rgb2gray(image)
    retinex_result = np.zeros_like(image_gray)

    for sigma in sigma_list:
        gaussian = cv2.GaussianBlur(image_gray, (0, 0), sigma)
        retinex_result += np.log1p(image_gray) - np.log1p(gaussian)
    
    retinex_result = np.exp(retinex_result)
    retinex_result = np.clip(retinex_result, 0, 1)
    return retinex_result

def extract_retinex_features(image):
    """
    Extract Retinex features from the image.
    """
    retinex_image = retinex(image)
    mean_retinex = np.mean(retinex_image)
    std_retinex = np.std(retinex_image)
    
    features = {
        'mean_retinex': mean_retinex,
        'std_retinex': std_retinex,
    }
    
    return features

def compute_gradient(image):
    """
    Compute the gradient magnitude of the image.
    """
    image_gray = rgb2gray(image)  # แปลงเป็นโทนสีเทา
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    mean_gradient = np.mean(grad_magnitude)
    std_gradient = np.std(grad_magnitude)
    
    features = {
        'mean_gradient': mean_gradient,
        'std_gradient': std_gradient,
    }
    
    return features



