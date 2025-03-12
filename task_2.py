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

    # Перетворення BGR -> RGB для коректного відображення в matplotlib
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # Відображення зображення за допомогою matplotlib
    plt.imshow(image_rgb)
    plt.title("Зображення з використанням OpenCV")
    plt.axis('off')
    plt.show()
    
    try:
        image_pil = Image.open(image_path)
        image_pil.show()  # Відкриває зображення в дефолтному переглядачі
    except Exception as e:
        print("Помилка при завантаженні зображення через PIL:", e)
        image_pil = None
        
    return image_cv2, image_pil

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH")
    load_and_display_image(image_path)
