import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset/text/medical_reports.csv")

X = data["report"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")

X_vector = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_vector,y,test_size=0.2,random_state=42
)

model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Text Model Accuracy:",accuracy)

with open("models/text_model.pkl","wb") as f:
    pickle.dump(model,f)

with open("models/vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)