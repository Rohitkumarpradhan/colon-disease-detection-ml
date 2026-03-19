# Colon Disease Detection using Machine Learning

## Project Overview

This project presents a **machine learning-based system for detecting colon disease using both medical images and clinical text reports**.
The system applies classical machine learning techniques to analyze **colonoscopy images and medical report text** to determine whether a patient shows signs of colon disease.

The goal of this project is to demonstrate how machine learning can assist in **early disease detection and medical decision support systems**.

The project uses two separate pipelines:

* **Image Classification Pipeline**
* **Text Classification Pipeline**

The system predicts whether a sample indicates **Colon Disease** or **Normal Colon Tissue**.

---

## Problem Statement

Early detection of colon diseases such as **colon cancer** is critical for effective treatment.
Manual analysis of colonoscopy images and clinical reports can be time-consuming and prone to human error.

This project demonstrates how machine learning techniques can assist healthcare professionals by automatically analyzing medical data.

---

## Key Features

* Colon disease detection using **medical images**
* Disease prediction using **clinical text reports**
* Machine learning implementation using **Scikit-learn**
* Image preprocessing using **OpenCV**
* Text feature extraction using **TF-IDF**
* Model serialization using **Pickle**
* Command-line interface for predictions

---

## Technologies Used

| Category             | Tools / Libraries    |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Machine Learning     | Scikit-learn         |
| Image Processing     | OpenCV               |
| Text Processing      | TF-IDF Vectorization |
| Model Storage        | Pickle               |
| Version Control      | Git                  |
| Repository Hosting   | GitHub               |

---

## Machine Learning Models

Two machine learning models are used in this project.

### Image Classification

Model: **Support Vector Machine (SVM)**

Process:

1. Image loading
2. Image resizing
3. Grayscale conversion
4. Pixel feature extraction
5. SVM classification

### Text Classification

Model: **Logistic Regression**

Process:

1. Text preprocessing
2. TF-IDF vectorization
3. Logistic Regression classification

---

## Project Structure

```
colon-disease-detection-ml
│
├── dataset
│   ├── images
│   │   ├── cancer
│   │   └── normal
│   │
│   └── text
│       medical_reports.csv
│
├── src
│   ├── train_image_model.py
│   ├── train_text_model.py
│   ├── predict_image.py
│   └── predict_text.py
│
├── models
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Rohitkumarpradhan/colon-disease-detection-ml.git
```

Navigate to the project directory:

```
cd colon-disease-detection-ml
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

## Train the Models

Run the following scripts to train the models.

Train image classification model:

```
python src/train_image_model.py
```

Train text classification model:

```
python src/train_text_model.py
```

The trained models will be stored locally in the `models` directory.

---

## Run the Application

Start the prediction system:

```
python main.py
```

Example output:

```
Colon Disease Detection System
1. Predict using Image
2. Predict using Text
```

Users can select the input type and obtain disease predictions.

---

## Example Predictions

Text Input:

```
Enter medical report: abnormal colon tissue growth detected
```

Output:

```
Colon Disease Detected
```

---

## Dataset

The project uses two types of datasets:

**Image Dataset**

Histopathological colon tissue images used for training the image classifier.

**Text Dataset**

A labeled dataset of clinical reports stored in:

```
dataset/text/medical_reports.csv
```

Each record contains:

* Medical report text
* Corresponding disease label

---

## Future Improvements

Possible improvements for this project include:

* Using **deep learning models such as CNN for image classification**
* Improving text classification using **advanced NLP models**
* Building a **web interface for real-time prediction**
* Expanding the dataset for improved accuracy
* Integrating clinical data for multi-modal disease detection

---




## License

This project is intended for **educational and research purposes only**.
