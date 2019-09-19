import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset',
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, default='liveness.model',
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
	help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Cai dat learn rate, so epoch=40, batch size =8
INIT_LR = 1e-4
BS = 8
EPOCHS = 40

# Doc du lieu khuon mat trong Dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	# Resize cac khuon mat ve 32x32 va them vao du lieu train gom image va label
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	data.append(image)
	labels.append(label)

# Chuan hoa du lieu anh ve [0,1]
data = np.array(data, dtype="float") / 255.0

# Chuyen du lieu nhan ve one hot vector
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# Phan chia du lieu train, test, 75% cho train va 25% cho test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# Augment anh cho phong phu du lieu train
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, brightness_range=[1,1.5],
	horizontal_flip=True, fill_mode="nearest")

# Compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# eval
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# Ghi model ra file
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Ghi label ra file
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()