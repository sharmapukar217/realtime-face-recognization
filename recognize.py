import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

svm_model, label_encoder = joblib.load("output/face_recognizer.sav")
face_detector = mp.solutions.face_detection.FaceDetection()
face_mesh = mp.solutions.face_mesh.FaceMesh()

frame_count = 0
cap = cv2.VideoCapture(0)
started_time = time.time()

while cap.isOpened():
	frame_count += 1
	success, frame = cap.read()
	if not success:
		break

	frame.flags.writeable = False
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = face_detector.process(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	frame.flags.writeable = True

	image_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
	image_resized = cv2.resize(image_gray, [224, 224])

	results = face_detector.process(image_resized)
	if not results.detections:
		continue

	for detection in results.detections:
		if detection.score[0] < 0.7:
			continue

		bbox = results.detections.location_data.relative_bounding_box
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
				accuracy = 0
				label = "Unknown"
				color = (0, 0, 255)
			else:
				color = (0, 255, 0)
				accuracy = round(svm_model.predict_proba(face_data)[0][pred_label] * 100, 2)

			cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
			cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# if results.detections:
	# 	for detection in results.detections:
	# 		if detection.score[0] > 0.8:
	# 			bounding_box = detection.location_data.relative_bounding_box
	# 			x = int(bounding_box.xmin * frame.shape[1])
	# 			y = int(bounding_box.ymin * frame.shape[0])
	# 			w = int(bounding_box.width * frame.shape[1])
	# 			h = int(bounding_box.height * frame.shape[0])

	# 			face_roi = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
	# 			face_roi = face_roi[y:y+h, x:x+w]

	# 			if face_roi.size == 0:
	# 				continue

	# 			face_roi = cv2.resize(face_roi, (128, 128))
	# 			face_roi = face_roi.reshape(1, -1)

	# 			face_roi_norm = pca.transform(face_roi)
	# 			pred_label = svm_model.predict(face_roi_norm)[0]

	# 			if pred_label >= len(label_encoder.classes_):
	# 				label = "Unknown"
	# 				confidence = 0.0
	# 			else:
	# 				label = label_encoder.inverse_transform([pred_label])[0]
	# 				# confidence = svm_model.decision_function(face_roi_norm)[0][pred_label]

	# 			# print(confidence)

	# 			if label == "Unknown":
	# 				color = (0, 0, 255)
	# 			else:
	# 				color = (0, 255, 0)

	# 			cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
	# 			cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	fps = frame_count / (time.time() - started_time)


	cv2.putText(frame, f"fps: {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
	cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
	cv2.imshow("frame", frame)

	if cv2.waitKey(1) == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()