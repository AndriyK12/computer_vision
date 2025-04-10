import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_dataset(dataset_dir, image_size=(512, 512)):
    """
    Завантажує зображення з датасету, що має наступну структуру:
      dataset_dir/
         Person1/   (фото для особи 1)
         Person2/   (фото для особи 2)
         ...
    Кожне зображення змінюється до розміру image_size,
    перетворюється в градації сірого та витягуються HOG-ознаки.
    """
    features = []
    labels = []
    for label in os.listdir(dataset_dir):
        class_folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            if file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                img_path = os.path.join(class_folder, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Зміна розміру
                img = cv2.resize(img, image_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Витягування ознак HOG
                hog_features = hog(gray, 
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   visualize=False,
                                   feature_vector=True)
                features.append(hog_features)
                labels.append(label)
    return np.array(features), np.array(labels)

def classify_faces(dataset_dir):
    X, y = load_dataset(dataset_dir, image_size=(128,128))
    print(f"Завантажено {len(X)} зразків з датасету.")
    # Розділення на тренувальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. SVM з лінійним ядром
    svm_clf = SVC(kernel='linear', probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    
    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    
    # 3. KNN (k=5)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred_knn))
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

if __name__ == "__main__":
    dataset_dir = os.getenv("SMALLER_DATASET_PATH")
    classify_faces(dataset_dir)
    