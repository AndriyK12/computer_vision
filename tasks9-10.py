import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_face_contours(image_path):
    """
    Завдання 9.
    Детекція області обличчя за допомогою кольорової сегментації в YCrCb‑просторі,
    подальший пошук країв (метод Canny) та знаходження контурів у зоні, яку визначено як обличчя.
    """
    # Завантаження зображення
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return
    output = image.copy()

    # Перетворення зображення до простору YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Значення для сегментації шкіри
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Морфологічні операції для усунення шуму
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Обличчя не знайдено за заданими параметрами сегментації.")
        return

    # Вважаємо, що найбільший контур – це обличчя
    face_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(face_contour)
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Виділення ROI (області обличчя)
    face_roi = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    # Знаходження країв методом Canny
    edges = cv2.Canny(face_roi, 50, 150)
    # Знаходження контурів в ROI
    roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Оскільки контури мають координати, відносні до ROI, зміщуємо їх до глобальних
    roi_contours = [cnt + np.array([x, y]) for cnt in roi_contours]
    # Малюємо контури зеленим кольором
    cv2.drawContours(output, roi_contours, -1, (0, 255, 0), 2)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Оригінальне зображення")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Маска шкіри (YCrCb)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Контури обличчя")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def segment_face_gray(image_path, thresh_value=120, invert=False):
    """
    Спроба знайти обличчя на чорно-білому зображенні шляхом порогової сегментації
    та вибору найбільшого контуру. Параметри:
      - thresh_value: глобальний поріг
      - invert: чи потрібно інвертувати результат (True/False)
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Звичайна (глобальна) порогова обробка
    # Якщо invert = False, поріг робить темніше за thresh – чорним,
    # інакше – навпаки
    ttype = cv2.THRESH_BINARY
    if invert:
        ttype = cv2.THRESH_BINARY_INV
    
    _, mask = cv2.threshold(gray, thresh_value, 255, ttype)

    # Морфологія для прибирання шуму та «відновлення» об’єкта
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Не знайдено жодного контуру.")
        return

    # Припускаємо, що найбільший контур – це обличчя
    face_contour = max(contours, key=cv2.contourArea)
    output = image.copy()
    cv2.drawContours(output, [face_contour], -1, (0, 255, 0), 2)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Оригінал")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Маска, thresh={thresh_value}, invert={invert}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Найбільший контур (імовірне обличчя)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def feature_detection(image_path):
    """
    Завдання 10.
    Аналіз особливостей зображення за допомогою декількох методів:
      - SIFT
      - SURF (недоступний)
      - ORB
      - HOG (глобальний дескриптор; демонструється графічно)
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. SIFT
    try:
        sift = cv2.SIFT_create()
    except Exception as e:
        print("Помилка при створенні SIFT:", e)
        return
    kp_sift, des_sift = sift.detectAndCompute(gray, None)
    image_sift = cv2.drawKeypoints(image, kp_sift, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 2. SURF (недоступний у безкоштовінй версії)
    try:
        surf = cv2.xfeatures2d.SURF_create()
        kp_surf, des_surf = surf.detectAndCompute(gray, None)
        image_surf = cv2.drawKeypoints(image, kp_surf, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    except Exception as e:
        print("SURF недоступний:", e)
        image_surf = image.copy()
    
    # 3. ORB
    orb = cv2.ORB_create()
    kp_orb, des_orb = orb.detectAndCompute(gray, None)
    image_orb = cv2.drawKeypoints(image, kp_orb, None, color=(0, 255, 0), flags=0)
    
    # 4. HOG – обчислення глобального дескриптора
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    hog_features_plot = hog_features[:500]

    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))
    plt.title("SIFT")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image_surf, cv2.COLOR_BGR2RGB))
    plt.title("SURF")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(image_orb, cv2.COLOR_BGR2RGB))
    plt.title("ORB")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.plot(hog_features_plot)
    plt.title("HOG (перші 500 значень)")
    plt.xlabel("Індекс")
    plt.ylabel("Значення")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = os.getenv("PHOTO_PATH4")
    detect_face_contours(image_path)
    feature_detection(image_path)

    image_path = os.getenv("PHOTO_PATH5")
    segment_face_gray(image_path)
    feature_detection(image_path)
