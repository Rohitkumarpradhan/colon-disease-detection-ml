import cv2
import pickle
import numpy as np

with open("models/image_model.pkl","rb") as f:
    model = pickle.load(f)

img = cv2.imread("normalcolon.jpeg")
img = cv2.resize(img,(64,64))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = img.flatten().reshape(1,-1)

prediction = model.predict(img)

if prediction == 1:
    print("Colon Disease Detected")
else:
    print("Normal Colon")