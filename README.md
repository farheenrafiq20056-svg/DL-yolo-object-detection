# Deep Learning Project #2 — Object Detection with YOLO

A beginner-friendly deep learning app that uses YOLOv8 (a real-time
object detection model) to find and label multiple objects in a photo.

## How this is different from Project #1
- Project #1 (image classifier): gives ONE label for the whole photo
  ("this is probably a dog").
- Project #2 (this one): finds MULTIPLE objects and draws a box around
  each one ("dog here, person here, car here"), with a confidence score
  for each detection.

## What it does
- Upload any JPG/PNG image
- YOLOv8n (the smallest, fastest YOLOv8 model) detects objects from
  80 everyday categories (person, car, dog, chair, laptop, etc.)
- Draws boxes + labels directly on the image
- Lists every detected object with its confidence %

## ⚠️ Important: Python version on Streamlit Cloud
YOLO (`ultralytics`) does not fully support Python 3.13+.
When you deploy this app on Streamlit Community Cloud:
1. Click "Advanced settings" **before** deploying.
2. In the "Python version" dropdown, choose **3.11 or 3.12**
   (NOT the default, which may be 3.13/3.14).
3. You cannot change this after deploying — you'd have to delete
   and redeploy the app to fix it, so set it correctly the first time.

## How to run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `packages.txt` to a GitHub repo.
2. Go to https://share.streamlit.io and connect your repo.
3. Set `app.py` as the main file.
4. Click "Advanced settings" and set Python version to 3.11 or 3.12.
5. Deploy. (First run downloads the YOLO model weights automatically —
   normal, only happens once.)

## What `packages.txt` is for
OpenCV (used internally by YOLO) sometimes needs a couple of system-level
graphics libraries on cloud servers. `packages.txt` tells Streamlit Cloud
to install them automatically — you don't need to do anything extra.

## How it works
- `YOLO("yolov8n.pt")` loads a pretrained object detection model.
- `model(image)` runs detection and returns boxes, class names, and
  confidence scores for everything it finds.
- `result.plot()` draws the boxes and labels directly onto the image.
- `@st.cache_resource` means the model loads into memory only once.
