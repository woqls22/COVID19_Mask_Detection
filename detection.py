import plaidml.keras
plaidml.keras.install_backend()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import cv2.dnn
import os
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True,
	help="path to input image")

args = vars(ap.parse_args())
Default_val = 0.5 # Confidence base Metric

#Face Detector 모델 로드
print("[INFO] loading face detector model...")
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Mask Detector 모델 로드
print("[INFO] loading face mask detector model...")
modelPath = './mask_detector.model'
model = load_model(modelPath)

image = cv2.imread(args["image"])
orig = image.copy()
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))
# Face Detection 계산. Network를 통해 blob 계산.
print("[INFO] computing face Detections..")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	# confidence 메트릭 추출
	confidence = detections[0,0,i,2]

	if(confidence>Default_val):
		# Face Case
		box = detections[0,0,i,3:7]*np.array([w,h,w,h])
		(startX, startY, endX, endY) = box.astype("int")

		(startX, startY) = (max(0,startX), max(0,startY))
		(endX, endY) = (max(w-1,endX), max(h-1,endY))
		face = image[startY:endY, startX:endX]
		# RGB Convert
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		(mask, withoutMask) = model.predict(face)[0] # 마스크 모델 Prediction

		if(mask>withoutMask):
			label = "Mask"
			color = (0, 255, 0)
		else:
			label = "No Mask"
			color = (0, 0, 255)
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)









