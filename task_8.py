import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grabcut_segmentation(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return

    mask = np.zeros(image.shape[:2], np.uint8)

    # Визначення початкового прямокутника, де, ймовірно, знаходиться об'єкт (передній план)
    height, width = image.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

    # Ініціалізація моделей фону та переднього плану
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Застосування алгоритму GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Перетворення отриманої маски: пікселі з позначками 0 і 2 (фон) стають 0, а з 1 і 3 (передній план) – 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_fg = image * mask2[:, :, np.newaxis]

    # Відображення результату
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Оригінальне зображення")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_fg, cv2.COLOR_BGR2RGB))
    plt.title("GrabCut сегментація (передній план)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH3")
    grabcut_segmentation(image_path)
