import os
import cv2
import sys
import time
import shutil
import mediapipe as mp

face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()
name = input("Enter the name: ").lower().replace(" ", "_")

face_directory = os.path.join("dataset", name)

if os.path.isdir(face_directory):
	if input("Data with same label already exists. replace it? [y/N] ").lower() == "y":
		print("Clearing the directory")
		shutil.rmtree(face_directory)
	else:
		print("Aborting...")
		exit()

clear_screen()
os.mkdir(face_directory)
print("[INFO]: after camera widow popups, (c) to start capturing, (p) to pause & (q) to exit.")
input("Press <enter> to continue")

started = False
frame_count = 0
sample_count = 1
SNAPSHOT_DELAY = 0.1
TOTAL_SNAPSHOTS = 200
started_time = time.time()

cap = cv2.VideoCapture(0)

time_counter = 0
while True:
	frame_count += 1
	ret, frame = cap.read()

	if not ret:
		break

	sys.stdout.write(f"\rtaking #{sample_count} out of {TOTAL_SNAPSHOTS} samples.")
	sys.stdout.flush()

	fps = frame_count / (time.time() - started_time)

	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	result = face_detector.process(frame)
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

	if result.detections and result.detections[0].score[0] > 0.8:
		bounding_box = result.detections[0].location_data.relative_bounding_box

		x = int(bounding_box.xmin * frame.shape[1])
		y = int(bounding_box.ymin * frame.shape[0])
		w = int(bounding_box.width * frame.shape[1])
		h = int(bounding_box.height * frame.shape[0])

		image = frame.copy()
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)

		if time_counter == 0: time_counter = time.time()

		if started and time.time() - time_counter > SNAPSHOT_DELAY:
			roi_image = image[y:y+h, x:x+w]
			if roi_image.size:
				roi_image = cv2.resize(roi_image, [224, 224], interpolation=cv2.INTER_AREA)
				cv2.imwrite(os.path.join(face_directory, f"{time.time()}.jpg"), image)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 8)
				sample_count += 1
				time_counter = 0

		if sample_count == TOTAL_SNAPSHOTS:
			break

	frame = cv2.flip(frame, 1)
	cv2.putText(frame, f"fps: {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
	cv2.putText(frame, f"#{sample_count}/{TOTAL_SNAPSHOTS} samples.", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

	cv2.namedWindow("snapshow", cv2.WINDOW_NORMAL)
	cv2.imshow("snapshow", frame)

	key = cv2.waitKey(1)
	if key == ord("q"): break
	elif key == ord("c"): started = True
	elif key == ord("p"): started = False

cap.release()
cv2.destroyAllWindows()



if sample_count == TOTAL_SNAPSHOTS:
	clear_screen()
	print("Operation completed...")
	input("Press <enter> to continue.")
	clear_screen()
