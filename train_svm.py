# %load train_svm
import os
import cv2
import joblib
import itertools
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

# augment.py
from augment import random_augmentations

# read images from these directories
images_list = list(Path("dataset").glob("**/*.png"))

# initialize face and mesh detector
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_list = []
face_encodings_list = []

for file_path in tqdm(images_list):
    file_path = str(file_path)
    image = cv2.imread(file_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image, [224, 224])

    augmented_images = random_augmentations(image_resized)

    for aug_image in augmented_images:
        results = face_detector.process(aug_image)

        if not results.detections:
            continue

        bbox = results.detections[0].location_data.relative_bounding_box
        xmin = int(bbox.xmin * aug_image.shape[1])
        ymin = int(bbox.ymin * aug_image.shape[0])
        width = int(bbox.width * aug_image.shape[1])
        height = int(bbox.height * aug_image.shape[0])
        face_image  = aug_image[ymin:height, xmin:width]

        
        results = face_mesh.process(aug_image)
        if results.multi_face_landmarks:
            landmark_values = []
            for landmark in results.multi_face_landmarks[0].landmark:
                x = int(landmark.x * face_image.shape[1])
                y = int(landmark.y * face_image.shape[0])
                landmark_values.append(x)
                landmark_values.append(y)
                
            face_encodings_list.append(np.array(landmark_values).flatten())
            labels_list.append(file_path.split(os.path.sep)[-2])

X = np.array(face_encodings_list)
y = np.array(labels_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

svm_model = svm.SVC(kernel='linear', C=1000, gamma=0.001, probability=True, verbose=2)
# svm_model = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True)

print("fitting model...")
svm_model.fit(X_train, y_train)
print("done!")

y_pred = svm_model.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Accuracy Score:', accuracy_score(y_test, y_pred))

print("saving the model and labels to file...")
joblib.dump((svm_model, label_encoder), "models/face_recognizer.sav")

# classes = svm_model.classes_
# cm = confusion_matrix(y_test, y_pred, labels=classes)

# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)
# fmt = 'd'
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(svm_model.decision_function(X_test))
# ax.set_xlabel('Samples')
# ax.set_ylabel('Confidence Scores')
# plt.show()