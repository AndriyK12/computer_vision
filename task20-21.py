import os
import cv2
import face_recognition
import numpy as np

# ========== Налаштування ==========
MODE            = "video"  # "image" або "video"
INPUT_PATH      = os.getenv("VIDEO_PATH3") #os.getenv("VIDEO_PATH") #None
KNOWN_FACES_DIR = os.getenv("SMALLER_DATASET_PATH2")
FRAME_RESIZE    = 0.25
TOLERANCE       = 0.6
MODEL           = "hog"
ALLOWED_NAMES   = {"pins_Adriana Lima", "pins_Norman Reedus"}

# ========== Завантаження відомих облич ==========
known_encodings = []
known_names     = []

for person in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        if not file.lower().endswith(('.jpg','jpeg','png')):
            continue
        path = os.path.join(person_dir, file)
        img = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(img, model=MODEL)
        encs  = face_recognition.face_encodings(img, boxes)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(person)

print(f"Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people.")

# ========== Функції розпізнавання ==========
def recognize_on_image(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=MODEL)
    encs  = face_recognition.face_encodings(rgb, boxes)

    for (top,right,bottom,left), enc in zip(boxes, encs):
        dists = face_recognition.face_distance(known_encodings, enc)
        idx   = np.argmin(dists)
        name  = known_names[idx] if dists[idx] <= TOLERANCE else "Unknown"
        color = (0,255,0) if name in ALLOWED_NAMES else (0,0,255)

        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, name, (left, bottom+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Image Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_on_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0,0), fx=FRAME_RESIZE, fy=FRAME_RESIZE, interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model=MODEL)
        encs  = face_recognition.face_encodings(rgb, boxes)

        for (top,right,bottom,left), enc in zip(boxes, encs):
            dists = face_recognition.face_distance(known_encodings, enc)
            idx   = np.argmin(dists)
            name  = known_names[idx] if dists[idx] <= TOLERANCE else "Unknown"
            color = (0,255,0) if name in ALLOWED_NAMES else (0,0,255)

            top    = int(top/FRAME_RESIZE)
            right  = int(right/FRAME_RESIZE)
            bottom = int(bottom/FRAME_RESIZE)
            left   = int(left/FRAME_RESIZE)

            status = "ACCESS GRANTED" if name in ALLOWED_NAMES else "ACCESS DENIED"
            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.putText(frame, f"{name}: {status}", (left, bottom+25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

        cv2.imshow("Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== Основний запуск ==========
if MODE == "image":
    if not INPUT_PATH:
        raise ValueError("MODE='image' requires INPUT_PATH to be set.")
    recognize_on_image(INPUT_PATH)
else:
    src = INPUT_PATH if INPUT_PATH else 0
    recognize_on_video(src)