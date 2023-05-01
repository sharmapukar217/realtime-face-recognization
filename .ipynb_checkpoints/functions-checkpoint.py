import cv2
import mediapipe as mp

detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_face(image):
	h, w, c = image.shape
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = detector.process(image_rgb)
	if results.detections:
		for detection in results.detections:
			bbox = detection.location_data.relative_bounding_box

			x = int(bbox.xmin * image_width)
			y = int(bbox.ymin * image_height)
			width = int(bbox.width * image_width)
			height = int(bbox.height * image_height)

			return image[y:y+height, x:x+width], (x, y, width, height)
	else:
		return None, None