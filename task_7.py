import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_segmentation(image_path):
    # Завантаження зображення у відтінках сірого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return

    # 1. Фіксована порогова сегментація
    fixed_threshold = 127  # приклад фіксованого порогу
    ret_fixed, thresh_fixed = cv2.threshold(image, fixed_threshold, 255, cv2.THRESH_BINARY)

    # 2. Сегментація за методом Otsu
    ret_otsu, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Оригінальне зображення")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(thresh_fixed, cmap='gray')
    plt.title(f"Фіксована сегментація (threshold = {fixed_threshold})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f"Otsu's метод (оптимальний поріг = {ret_otsu:.2f})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH2")
    threshold_segmentation(image_path)
