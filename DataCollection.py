import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# === Global Settings ===
offset = 20
imgSize = 300
# Modify this to your user Desktop path
folder = os.path.join(os.path.expanduser("~"), "Desktop", "HandSignDetection", "data", "Yes")
maxHands = 1

# === Ensure Data Folder Exists ===
def create_data_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === Initialize Hand Detector ===
def init_detector():
    return HandDetector(maxHands=maxHands)

# === Crop and Resize Hand Image ===
def crop_and_resize(img, bbox):
    x, y, w, h = bbox

    # Ensure boundaries are within the image
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(img.shape[1], x + w + offset)
    y2 = min(img.shape[0], y + h + offset)

    imgCrop = img[y1:y2, x1:x2]

    if imgCrop.size == 0:
        return None, None

    aspectRatio = imgCrop.shape[0] / imgCrop.shape[1]
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if aspectRatio > 1:
        k = imgSize / imgCrop.shape[0]
        wCal = math.ceil(k * imgCrop.shape[1])
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / imgCrop.shape[1]
        hCal = math.ceil(k * imgCrop.shape[0])
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    return imgCrop, imgWhite

# === Main Capture Loop ===
def start_capture():
    cap = cv2.VideoCapture(0)
    detector = init_detector()
    counter = 0
    create_data_folder(folder)

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to read from camera")
            continue

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            imgCrop, imgWhite = crop_and_resize(img, hand['bbox'])

            if imgCrop is not None:
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord("s") and hands:
            counter += 1
            filename = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"✅ Saved {filename} | Total: {counter}")
        elif key == ord("q"):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# === Run Program ===
if __name__ == "__main__":
    start_capture()
