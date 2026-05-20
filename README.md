# 🌿 Palm Leaf Nutrient Deficiency Detection using YOLO

## 📌 Project Overview
This project uses **YOLO (You Only Look Once)** object detection to identify and detect nutrient deficiencies in palm leaf images. The model detects deficiency regions and classifies them based on visual symptoms such as yellowing, edge burning, and abnormal leaf patterns.

The system helps in early nutrient deficiency detection and supports faster agricultural monitoring.

---

## 🎯 Problem Statement
Manual inspection of palm leaves for nutrient deficiencies is time-consuming and requires expert knowledge. Early symptoms are often difficult to detect, which may reduce crop yield and cause financial loss.

This project aims to automate nutrient deficiency detection using deep learning and computer vision.

---

## 🚀 Objectives
- Detect nutrient deficiencies from palm leaf images
- Automate inspection process
- Reduce manual effort
- Improve early diagnosis

---

## 📂 Dataset
The dataset contains approximately **14,000+ palm leaf images** categorized into:

- Healthy
- Nitrogen Deficiency
- Magnesium (Mg) Deficiency
- Potassium (Kalium) Deficiency
- Boron Deficiency

Images were labeled for YOLO object detection training.

---

## 🛠️ Technologies Used

- Python
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Google Colab / Jupyter Notebook

---

## ⚙️ Data Preprocessing
The following preprocessing and augmentation techniques were applied:

- Image resizing
- Normalization
- Rotation
- Zoom
- Horizontal flip
- Brightness adjustment
- Shearing
- Width and height shifting

These techniques improved model generalization and reduced overfitting.

---

## 🧠 Model Architecture
This project uses **YOLO object detection**.

YOLO works by:
1. Processing the entire image at once
2. Detecting object location using bounding boxes
3. Predicting object class and confidence score

Advantages:
- Fast detection
- Real-time prediction
- High efficiency

---

## 📊 Training Details

- Model: YOLO
- Image Size: 640×640
- Epochs: 50
- Batch Size: 16
- Optimizer: Default YOLO optimizer
- Device: GPU

---

## 📈 Model Performance
Evaluation metrics used:

- Precision
- Recall
- mAP (Mean Average Precision)
- Confusion Matrix

The model achieved strong detection performance on palm leaf deficiency images.

---

## 🖼️ Sample Results

(Add prediction images here)

Example:

- Detected deficiency class
- Bounding box
- Confidence score

---

## 🌐 Deployment
The trained YOLO model can be deployed using **Streamlit** for real-time image upload and prediction.

Users can:
- Upload leaf image
- View detected deficiency
- See prediction confidence

---

## ⚠️ Challenges Faced

During development, several challenges were encountered:

- Mixed image classes
- Corrupted images
- Class imbalance
- Labeling issues
- Model tuning
- Preprocessing mismatch

These were handled through dataset cleaning and proper preprocessing.

---

## 🔮 Future Improvements

Future work may include:

- Larger dataset
- Better annotation quality
- More advanced YOLO versions
- Mobile deployment
- Real-time field monitoring

---

## 📬 Author

**Prachi Ambarte**  
Aspiring AI/ML Engineer | Data Science Enthusiast

LinkedIn: (Add Link)  
GitHub: (Add Link)
