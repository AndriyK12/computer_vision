import face_recognition
import cv2
import numpy as np
import os
import sys

known_faces_dir = r"C:\Users\katsm\Desktop\computer_vision\smaller_dataset_2"#os.getenv("SMALLER_DATASET_PATH2")
video_path = os.getenv("VIDEO_PATH")

if not known_faces_dir:
    print("Помилка: Змінна середовища DATASET_PATH не встановлена.")
    sys.exit(1)

if not video_path:
    print("Помилка: Змінна середовища VIDEO_PATH не встановлена.")
    sys.exit(1)

if not os.path.isdir(known_faces_dir):
    print(f"Помилка: Директорія для відомих облич не знайдена: {known_faces_dir}")
    sys.exit(1)

known_face_encodings = []
known_face_names = []

print(f"Завантаження відомих облич з {known_faces_dir}...")
try:
    for class_name in os.listdir(known_faces_dir):
        class_dir_path = os.path.join(known_faces_dir, class_name)

        if os.path.isdir(class_dir_path):
            print(f"Обробка класу (директорії): {class_name}")

            for filename in os.listdir(class_dir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir_path, filename)
                    print(f"  Обробка файлу: {filename}...")
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings_list = face_recognition.face_encodings(image)

                        if face_encodings_list:
                            known_face_encodings.append(face_encodings_list[0])
                            known_face_names.append(class_name)
                            print(f"    Знайдено та додано обличчя для класу '{class_name}'")
                        else:
                            print(f"    Попередження: Обличчя не знайдено у файлі {filename}")
                    except Exception as e:
                        print(f"    Помилка обробки файлу {filename}: {e}")
        else:
            print(f"Пропуск елемента (не директорія): {class_name}")
except FileNotFoundError:
    print(f"Помилка: Не вдалося отримати доступ до директорії: {known_faces_dir}")
    sys.exit(1)

if not known_face_encodings:
    print("Попередження: Не завантажено жодного відомого обличчя. Розпізнавання неможливе.")

print(f"Завантажено {len(known_face_names)} відомих облич.")

print(f"Відкриття відеофайлу: {video_path}")
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Помилка: Не вдалося відкрити відео: {video_path}")
    sys.exit(1)

face_locations = []
face_names = []
process_this_frame = True

print("Початок обробки відео...")
while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Кінець відео або помилка читання кадру.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except TypeError as e:
                 print("\nПомилка TypeError під час виклику face_recognition.face_encodings:")
                 print(f"Аргументи: image shape={rgb_small_frame.shape}, dtype={rgb_small_frame.dtype}, locations={face_locations}")
                 print(f"Повідомлення про помилку: {e}")
            except Exception as e:
                 print(f"\nНеочікувана помилка під час face_recognition.face_encodings: {e}")
                 break

            face_names = []
            for face_encoding in face_encodings:
                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Невідомо"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                else:
                    name = "I have no idea"

                face_names.append(name)
        else:
             face_names = []

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        label_y = bottom - 15 if bottom - 15 > 15 else bottom + 15
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    cv2.imshow('Face recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Вихід за запитом користувача.")
        break

print("Звільнення ресурсів...")
video_capture.release()
cv2.destroyAllWindows()
print("Завершено.")