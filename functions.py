import os
import sys
import cv2
from urllib import request

def reporthook(count, block_size, total_size):
    percent = min((count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r%.2f%%" % percent)
    sys.stdout.flush()

# utility function to download from url
def download_file(url, path):
  print(f"downloading from {url}...")
  request.urlretrieve(url,path,reporthook)
  print(f"\nfile saved to `{path}`.")


def get_face_detector():
	proto_path = "models/deploy.prototxt"
	model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

	if not os.path.isdir("models"):
		os.mkdir("models")

	for file in [proto_path, model_path]:
		if not os.path.isfile(file):
			download_file(f"https://raw.githubusercontent.com/sharmapukar217/realtime-face-recognization/main/{file}", file)

	detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

	return detector

def get_face_embedder():
	model_path = "models/openface_nn4.small2.v1.t7"

	if not os.path.isdir("models"):
		os.mkdir("models")

	if not os.path.isfile(model_path):
		download_file(f"https://raw.githubusercontent.com/sharmapukar217/realtime-face-recognization/main/{model_path}", model_path)

	embedder = cv2.dnn.readNetFromTorch(model_path)
	return embedder