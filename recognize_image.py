import cv2
import pickle
import imutils
import numpy as np
from functions import get_face_detector, get_face_embedder

face_detector = get_face_detector()
face_embedder = get_face_embedder()

recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
label_encoder = pickle.loads(open('output/label_encodings.pickle', "rb").read())

image =  cv2.imread("dataset/$B26/100.png")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
face_detector.setInput(image_blob)
detections = face_detector.forward()

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > 0.5:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(start_x, start_y, end_x, end_y) = box.astype("int")

		face = image[start_y:end_y, start_x:end_x]
		(fH, fW) = face.shape[:2]

		if fW < 20 or fH < 20:
			continue

		face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
		face_embedder.setInput(face_blob)
		vec = face_embedder.forward()

		preds = recognizer.predict_proba(vec)[0]

		j = np.argmax(preds)
		probability = preds[j]

		name = label_encoder.classes_[j]
		print(name)



cv2.waitKey()
cv2.destroyAllWindows()