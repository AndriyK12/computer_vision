import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles, figsize=(15,5)):
    """Функція для відображення кількох зображень із заголовками."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def scale_image(image, scale_x=1.0, scale_y=1.0):
    """Масштабування зображення за коефіцієнтами scale_x, scale_y (з використанням нових розмірів)."""
    h, w = image.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled

def rotate_image(image, angle, center=None, scale=1.0):
    """Обертання зображення навколо центру; angle – кут в градусах."""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def perspective_transform(image, src_points, dst_points):
    """
    Виконує перспективну трансформацію.
    src_points: np.array (4,2) – координати точок на вхідному зображенні.
    dst_points: np.array (4,2) – координати точок, до яких перетворюється зображення.
    """
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return transformed

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH7")
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення:", image_path)
        exit()

    scaled = scale_image(image, scale_x=2, scale_y=2)
    
    rotated = rotate_image(image, angle=45)

    h, w = image.shape[:2]
    src_pts = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    # Задати нові точки: трохи «зснути» кути для створення перспективного ефекту
    offset = 50
    dst_pts = [[0 + offset, 0 + offset], [w - 1 - offset, 0 + offset],
               [w - 1 - offset, h - 1 - offset], [0 + offset, h - 1 - offset]]
    perspective = perspective_transform(image, src_pts, dst_pts)

    display_images(
        [image, scaled, rotated, perspective],
        ["Оригінал", "Масштабоване", "Обертання (45°)", "Перспективна трансформація"]
    )
    