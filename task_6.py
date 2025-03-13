import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_sharpness(image):
    # Розмиття зображення за допомогою Гаусового фільтра
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Обчислення різкості за допомогою зваженого додавання
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH2")
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
    else:
        sharpened = enhance_sharpness(image)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Оригінал")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        plt.title("Підвищення різкості")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
