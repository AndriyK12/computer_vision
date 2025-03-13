import os
from dotenv import load_dotenv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_and_display_image(image_path):
    image_cv2 = cv2.imread(image_path)
    if image_cv2 is None:
        print("Не вдалося завантажити зображення через OpenCV.")
        return None, None

    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Зображення з використанням OpenCV")
    plt.axis('off')
    plt.show()
    
    try:
        image_pil = Image.open(image_path)
        image_pil.show()
    except Exception as e:
        print("Помилка при завантаженні зображення через PIL:", e)
        image_pil = None
        
    return image_cv2, image_pil

def plot_histogram(image_cv2):
    # Перетворення зображення в сіре
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    # Розрахунок гістограми
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Гістограма яскравості")
    plt.xlabel("Інтенсивність")
    plt.ylabel("Кількість пікселів")
    plt.show()

def enhance_contrast(image_cv2):
    # Перетворення зображення у відтінки сірого
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Метод 1: Гістограмне вирівнювання
    equalized = cv2.equalizeHist(gray)
    
    # Метод 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Оригінал (сіре)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title("Гістограмне вирівнювання")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(clahe_img, cmap='gray')
    plt.title("CLAHE")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH")
    image_cv2, _ = load_and_display_image(image_path)
    if image_cv2 is not None:
        plot_histogram(image_cv2)
    if image_cv2 is not None:
        enhance_contrast(image_cv2)