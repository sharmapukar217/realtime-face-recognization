import os
import cv2
import joblib
import urllib
import argparse
import numpy as np
import mediapipe as mp

svm_model, label_encoder = joblib.load("output/face_recognizer.sav")

face_detector = mp.solutions.face_detection.FaceDetection()
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

parser = argparse.ArgumentParser(prog="SVM Face Recognization")
parser.add_argument("-i", "--image", help="recognize and get label from provided image path")


def predict_image(image_path):
	if image_path.startswith("http"):
		req = urllib.request.urlopen(image_path)
		arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
		image = cv2.imdecode(arr, -1)
	elif os.path.isfile(args.image):
		image = cv2.imread(image_path)
	else:
		print("couldn't load the provided image. exitting...")
		return

	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_resized = cv2.resize(image, [224, 224])

	results = face_detector.process(image_resized)
	if not results.detections:
		print("no face detected.")
		return

	bbox = results.detections[0].location_data.relative_bounding_box
	xmin = int(bbox.xmin * image_resized.shape[1])
	ymin = int(bbox.ymin * image_resized.shape[0])
	width = int(bbox.width * image_resized.shape[1])
	height = int(bbox.height * image_resized.shape[0])

	face_image  = image_resized[ymin:height, xmin:width]

	results = face_mesh.process(image_resized)

	if results.multi_face_landmarks:
		face_landmarks = []
		for landmark in results.multi_face_landmarks[0].landmark:
			x = landmark.x * face_image.shape[1]
			y = landmark.y * face_image.shape[0]
			face_landmarks.append(x)
			face_landmarks.append(y)
			face_landmarks.append(landmark.z)

		face_landmarks = np.array(face_landmarks)
		face_data = face_landmarks.reshape(1, -1)
		pred_label = svm_model.predict(face_data)[0]

		label = label_encoder.inverse_transform([pred_label])[0]

		if pred_label == len(label_encoder.classes_):
			print("Unknown face.")
		else:
			accuracy = round(svm_model.predict_proba(face_data)[0][pred_label] * 100, 2)
			print(f"predicted: {label}, accuracy: {accuracy}%")

args = parser.parse_args()

if args.image is not None:
	predict_image(args.image)