# USAGE
# python liveness_demo.py

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Cai dat cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='liveness.model',
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load model nhan dien khuon mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load model nhan dien fake/real
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

#  Doc video tu webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
	# Doc anh tu webcam
	frame = vs.read()

	# Chuyen thanh blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# Phat hien khuon mat
	net.setInput(blob)
	detections = net.forward()

	# Loop qua cac khuon mat
	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		# Neu conf lon hon threshold
		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# Lay vung khuon mat
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# Dua vao model de nhan dien fake/real
			preds = model.predict(face)[0]

			j = np.argmax(preds)
			label = le.classes_[j]

			# Ve hinh chu nhat quanh mat
			label = "{}: {:.4f}".format(label, preds[j])
			if (j==0):
				# Neu la fake thi ve mau do
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
			else:
				# Neu real thi ve mau xanh
				cv2.putText(frame, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (0,  255,0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Bam 'q' de thoat
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()