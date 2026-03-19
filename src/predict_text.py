import pickle

with open("models/text_model.pkl","rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

report = input("Enter medical report: ")

report_vector = vectorizer.transform([report])

prediction = model.predict(report_vector)

if prediction == 1:
    print("Colon Disease Detected")
else:
    print("Normal Colon")