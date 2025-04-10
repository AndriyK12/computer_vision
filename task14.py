import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_transformations(image, kernel_size=(5,5)):
    """
    Виконує операції ерозії, дилатації, відкриття та закриття на заданому зображенні.
    Очікується, що вхідне зображення є в градаціях сірого.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # Ерозиція
    erosion = cv2.erode(image, kernel, iterations=1)
    # Дилатація
    dilation = cv2.dilate(image, kernel, iterations=1)
    # Відкриття: ерозія, потім дилатація
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    # Закриття: дилатація, потім ерозія
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return erosion, dilation, opening, closing

if __name__ == "__main__":
    # Завантажуємо зображення в градаціях сірого
    image_path = os.getenv("PHOTO_PATH8")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Не вдалося завантажити зображення:", image_path)
        exit()

    erosion, dilation, opening, closing = morphological_transformations(image, kernel_size=(5,5))

    plt.figure(figsize=(16,4))
    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Оригінал (маска)")
    plt.axis("off")
    
    plt.subplot(1, 5, 2)
    plt.imshow(erosion, cmap="gray")
    plt.title("Ерозія")
    plt.axis("off")
    
    plt.subplot(1, 5, 3)
    plt.imshow(dilation, cmap="gray")
    plt.title("Дилатація")
    plt.axis("off")
    
    plt.subplot(1, 5, 4)
    plt.imshow(opening, cmap="gray")
    plt.title("Відкриття")
    plt.axis("off")
    
    plt.subplot(1, 5, 5)
    plt.imshow(closing, cmap="gray")
    plt.title("Закриття")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    