import streamlit as st
import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load model and labels
model = load_model("models/keras_model.h5")
labels = ["Hello", "ILoveYou", "No", "Thanks", "Yes"]

# Initialize hand detector
detector = HandDetector(maxHands=1)
imgSize = 300
offset = 20

st.title("🤟 Real-Time Hand Sign Recognition")
st.markdown("Use your webcam to detect hand signs in real-time.")

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

start = st.button("Start Webcam")
stop = st.button("Stop Webcam")

if start:
    st.session_state.camera_active = True
if stop:
    st.session_state.camera_active = False

FRAME_WINDOW = st.empty()

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)

    while st.session_state.camera_active:
        ret, img = cap.read()
        if not ret:
            st.warning("Unable to access the camera.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            height, width, _ = img.shape
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, width)
            y2 = min(y + h + offset, height)

            imgCrop = img[y1:y2, x1:x2]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Prepare image for model
                inputImg = cv2.resize(imgWhite, (224, 224))
                inputImg = np.asarray(inputImg, dtype=np.float32).reshape(1, 224, 224, 3)
                inputImg = (inputImg / 127.5) - 1

                prediction = model.predict(inputImg)
                index = np.argmax(prediction)
                label = labels[index]
                confidence = prediction[0][index]

                # Annotate result
                cv2.rectangle(imgOutput, (x1, y1 - 40), (x1 + 200, y1), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f"{label} ({confidence:.2f})", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 2)

            except Exception as e:
                print("⚠️ Error:", e)
                continue

        FRAME_WINDOW.image(imgOutput, channels="BGR", use_container_width=True)

    cap.release()
