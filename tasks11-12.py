import os
import cv2
import numpy as np

def optical_flow_demo(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не вдалося відкрити відеофайл:", video_path)
        return

    # Зчитуємо перший кадр
    ret, frame1 = cap.read()
    if not ret:
        print("Не вдалося прочитати перший кадр.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Створюємо HSV зображення для візуалізації optical flow (вони будуть мати ті ж розміри, що і кадр)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # максимальна насиченість

    # Створення вікна один раз перед циклом:
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow", 640, 480)
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Обчислення dense optical flow за алгоритмом Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Перетворення векторного поля в магнітуду та кут
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # напрямок руху у відтінках (H)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # швидкість руху визначає яскравість (V)

        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Показуємо кадр з optical flow
        cv2.imshow("Optical Flow", bgr_flow)
        
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC для виходу
            break

        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()

def background_subtraction_demo(video_path="video.mp4", method="MOG2"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не вдалося відкрити відеофайл:", video_path)
        return

    # Створення BackgroundSubtractor із тінями (detectShadows=True)
    if method == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
    
    # Створення вікон з конкретними розмірами
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 640, 480)
    cv2.namedWindow("Foreground Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foreground Mask", 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Отримання маски переднього плану
        fg_mask = backSub.apply(frame)

        # Видалення тіней:
        # Якщо detectShadows увімкнено, тіні зазвичай позначаються значенням близьким до 127.
        # Ми використовуємо порогову обробку, щоб отримати лише значення 255 (передній план)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Морфологічна обробка для очищення маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Знаходимо контури у масці та малюємо прямокутники навколо фрагментів, що мають достатню площу
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Фільтруємо дрібні шумові області
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Відображення оригінального кадру з прямокутниками та маски
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":    
    video_path = os.getenv("VIDEO_PATH")
    optical_flow_demo(video_path)

    video_path = os.getenv("VIDEO_PATH2")
    background_subtraction_demo(video_path, method="MOG2")
    