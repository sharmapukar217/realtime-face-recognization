import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from imutils import paths, resize

from functions import get_face_detector, get_face_embedder

face_detector = get_face_detector()
face_embedder = get_face_embedder()

image_paths = list(paths.list_images("dataset"))

known_names = []
known_embeddings = []
total_faces_processed = 0


for image_path in tqdm(image_paths):
	name = image_path.split(os.path.sep)[-2]

	image = cv2.imread(image_path)
	image = resize(image, width=600)

	(h, w) = image.shape[:2]

	image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
	face_detector.setInput(image_blob)

	detections = face_detector.forward()

	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
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

			known_names.append(name)
			known_embeddings.append(vec.flatten())
			total_faces_processed += 1



print(f"serializing {total_faces_processed} encodings...")
data = {"embeddings": known_embeddings, "names": known_names}
with open("output/embeddings.pickle", "wb") as file:
	file.write(pickle.dumps(data))
	file.close()