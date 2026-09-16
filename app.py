import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# ---------------------------------------------------
# Deep Learning Project #2 - Object Detection with YOLO
# Beginner friendly version - easy to read and edit
#
# Difference from Project #1 (image classifier):
# - Classifier: "this whole photo is probably a dog" (1 label)
# - YOLO (object detection): "there's a dog HERE, a person HERE,
#   and a car HERE" - it finds and draws boxes around multiple
#   objects in the same image.
#
# YOLO ("You Only Look Once") is a real-time object detection
# deep learning model. We use a small pretrained version here
# (yolov8n = "nano", the smallest/fastest YOLOv8 model) so it
# runs quickly without a GPU.
# ---------------------------------------------------

st.set_page_config(page_title="YOLO Object Detector", page_icon="🎯")

st.title("🎯 Deep Learning Object Detection (YOLO)")
st.write(
    "Upload a photo and YOLOv8 (a pretrained deep learning model) will "
    "find and label every object it recognizes, with a box around each one."
)


# ---------------------------
# Step 1: Load the pretrained YOLO model (cached so it loads once)
# ---------------------------
@st.cache_resource
def load_model():
    # "yolov8n.pt" downloads automatically the first time this runs.
    # It's pretrained on the COCO dataset (80 everyday object types:
    # person, car, dog, chair, laptop, etc.)
    return YOLO("yolov8n.pt")


model = load_model()

# ---------------------------
# Step 2: Upload an image
# ---------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        # Run the image through YOLO
        results = model(np.array(image))
        result = results[0]

        # result.plot() draws the boxes and labels on the image for us
        annotated_image = result.plot()  # returns a BGR numpy array
        annotated_image = annotated_image[:, :, ::-1]  # convert BGR to RGB

    st.subheader("Detected Objects")
    st.image(annotated_image, use_container_width=True)

    # ---------------------------
    # Step 3: List out what was found, with confidence scores
    # ---------------------------
    if len(result.boxes) == 0:
        st.info("No objects were confidently detected in this image.")
    else:
        st.subheader("Details")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            st.write(f"**{class_name}** — {confidence:.1f}% confidence")
else:
    st.info("Upload an image above to see YOLO detect objects in it.")
