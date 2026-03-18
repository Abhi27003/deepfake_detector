# 🧠 Deepfake Detection System

## 🚀 Overview

This project is a **Deepfake Detection System** that uses **Deep Learning (ResNet50)** to classify images as **Real or Fake**.
It includes a **Streamlit-based web interface** with face detection and real-time prediction.

---

## 🎯 Features

* 🔍 Face Detection using OpenCV
* 🤖 Deep Learning Model (ResNet50)
* 🖼 Upload Image Detection
* 📸 Camera-based Detection
* 📊 Confidence Score Display
* ⚡ Fast Prediction using optimized model

---

## 🧠 Tech Stack

* **Frontend/UI:** Streamlit
* **Backend/Model:** PyTorch
* **Model Architecture:** ResNet50
* **Image Processing:** OpenCV, PIL
* **Deployment:** Streamlit Cloud

---

## ⚙️ How It Works

1. User uploads or captures an image
2. Face is detected using OpenCV
3. Image is cropped and resized (224×224)
4. Passed to trained ResNet50 model
5. Model predicts Real or Fake
6. Confidence score is displayed

---

## 📁 Project Structure

```
deepfake/
│
├── app.py              # Streamlit UI
├── train.py            # Model training code
├── test.py             # Testing script
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## 🧪 Model Details

* Architecture: ResNet50
* Training: Transfer Learning
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Dataset: Real vs Fake images

---

## ▶️ How to Run Locally

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Run App

```
streamlit run app.py
```

---

## 🌐 Deployment

The project is deployed using **Streamlit Cloud** and can be accessed via a web link.

---

## ⚠️ Limitations

* Accuracy depends on dataset quality
* May fail on low-resolution images
* Limited training data may affect performance

---

## 🚀 Future Improvements

* 🎥 Video deepfake detection
* 🔥 Grad-CAM heatmaps (AI explainability)
* ☁️ Cloud deployment optimization
* 📊 Improved dataset for higher accuracy

---

## 👨‍💻 Author

Developed as part of a Deep Learning / AI project.

---

## 💡 Conclusion

This project demonstrates how **AI + Computer Vision** can be used to detect manipulated media and help prevent misinformation.

---
