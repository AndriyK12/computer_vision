import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image):
    # Гаусовий фільтр
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Медіанний фільтр
    median = cv2.medianBlur(image, 5)
    
    # Реалізація біквадратного фільтра за допомогою кастомного ядра
    def bisquare_kernel(size, r0):
        center = size // 2
        kernel = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center)**2 + (j - center)**2)
                if r <= r0:
                    kernel[i, j] = (1 - (r / r0)**2)**2
                else:
                    kernel[i, j] = 0
        # Нормалізація ядра
        kernel /= kernel.sum()
        return kernel
    
    # Для прикладу заюзаємо ядро 5x5 із r0=2
    kernel = bisquare_kernel(5, 2)
    bisquare = cv2.filter2D(image, -1, kernel)
    
    return gaussian, median, bisquare

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH2")
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
    else:
        gaussian, median, bisquare = apply_filters(image)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
        plt.title("Гаусовий фільтр")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
        plt.title("Медіанний фільтр")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(bisquare, cv2.COLOR_BGR2RGB))
        plt.title("Біквадратний фільтр")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
