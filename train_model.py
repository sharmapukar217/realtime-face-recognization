from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


data = pickle.loads(open("output/embeddings.pickle", "rb").read())

label_encodings = LabelEncoder()
labels = label_encodings.fit_transform(data["names"])


recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)


with open("output/recognizer.pickle", "wb") as file:
	file.write(pickle.dumps(recognizer))
	file.close()

with open("output/label_encodings.pickle", "wb") as file:
	file.write(pickle.dumps(label_encodings))
	file.close()
