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

def print_image_characteristics(image_cv2):
    if image_cv2 is None:
        print("Зображення не завантажене.")
        return
    # Отримання розмірів та кількості каналів зображення
    height, width = image_cv2.shape[:2]
    channels = image_cv2.shape[2] if len(image_cv2.shape) > 2 else 1
    print(f"Розмір зображення: {width} x {height} пікселів")
    print(f"Кількість каналів: {channels}")

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH")
    image_cv2, image_pil = load_and_display_image(image_path)
    print_image_characteristics(image_cv2)
