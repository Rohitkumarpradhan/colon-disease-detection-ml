import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = "dataset/images"

for category in ["normal","cancer"]:
    
    path = os.path.join(dataset_path, category)
    label = 0 if category=="normal" else 1

    for img in os.listdir(path):

        img_path = os.path.join(path,img)

        image = cv2.imread(img_path)
        image = cv2.resize(image,(64,64))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        data.append(image.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(
    data,labels,test_size=0.2,random_state=42
)

model = SVC()

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Image Model Accuracy:",accuracy)

with open("models/image_model.pkl","wb") as f:
    pickle.dump(model,f)