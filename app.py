# ---------------- IMPORTS ----------------
import streamlit as st
import time
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, models
import gdown

url = "https://drive.google.com/uc?id=171wZwE0DjI2gXPdk-fm1iloZWcTN7WkT"
gdown.download(url, "best_model.pth", quiet=False)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection System", layout="wide")

# ---------------- LOGO ----------------
logo_url = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

col1, col2, col3 = st.columns([5, 5, 5])
with col2:
    st.image(logo_url, width=180)

# ---------------- TITLE ----------------
st.title("Deepfake Detection System")
st.write("Upload or capture image to detect deepfakes with face auto-cropping")


# ---------------- MODEL LOAD ----------------
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

classes = ["fake", "real"]

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FACE CROP FUNCTION ----------------


def detect_and_crop_face(image):
    img = np.array(image)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return image, None, image

    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]

    cropped_face = img[y:y+h, x:x+w]

    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return Image.fromarray(cropped_face), (x, y, w, h), Image.fromarray(img_with_box)

# ---------------- PREDICTION FUNCTION ----------------


def predict_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]
    conf = confidence.item() * 100

    return label.capitalize(), conf


# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["Upload", "Camera"])

# ---------------- UPLOAD TAB ----------------
with tab1:
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Drag & Drop Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        cropped_face, face_coords, boxed_image = detect_and_crop_face(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(boxed_image, caption="Detected Face",
                     use_container_width=True)

        with col2:
            if face_coords:
                st.image(cropped_face, caption="Cropped Face",
                         use_container_width=True)
            else:
                st.warning("No face detected - using full image")

        if st.button("Run Detection"):
            with st.spinner("Analyzing..."):
                result, confidence = predict_image(cropped_face)

            if result == "Fake":
                st.error(f"Fake Image Detected ({confidence:.2f}%)")
            else:
                st.success(f"Real Image ({confidence:.2f}%)")

            st.progress(int(confidence))

# ---------------- CAMERA TAB ----------------
with tab2:
    st.subheader("Camera Detection")

    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        cropped_face, face_coords, boxed_image = detect_and_crop_face(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(boxed_image, caption="Detected Face",
                     use_container_width=True)

        with col2:
            if face_coords:
                st.image(cropped_face, caption="Cropped Face",
                         use_container_width=True)
            else:
                st.warning("No face detected - using full image")

        if st.button("Run Detection (Camera)"):
            with st.spinner("Analyzing..."):
                result, confidence = predict_image(cropped_face)

            if result == "Fake":
                st.error(f"Fake Image Detected ({confidence:.2f}%)")
            else:
                st.success(f"Real Image ({confidence:.2f}%)")

            st.progress(int(confidence))

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Deepfake Detection System | AI Powered 🚀")
