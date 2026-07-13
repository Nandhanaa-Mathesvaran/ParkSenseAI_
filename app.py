import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="ParkSense AI",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ ParkSense AI")
st.subheader("Smart Parking Occupancy Detection using YOLOv8")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-4/weights/best.pt")

model = load_model()

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(image, conf=0.25)[0]

    occupied = 0
    empty = 0

    # Draw boxes
    for box in results.boxes:

        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            color = (0,255,0)
            empty += 1
        else:
            color = (0,0,255)
            occupied += 1

        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)

    total = occupied + empty

    availability = 0

    if total != 0:
        availability = (empty/total)*100

    # Draw top panel
    cv2.rectangle(image,(10,10),(350,90),(255,255,255),-1)

    cv2.putText(
        image,
        f"Occupied : {occupied}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.putText(
        image,
        f"Empty : {empty}",
        (20,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,180,0),
        2
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1,col2 = st.columns([3,1])

    with col1:
        st.image(image_rgb,use_container_width=True)

    with col2:

        st.metric("Occupied", occupied)

        st.metric("Empty", empty)

        st.metric("Total Slots", total)

        st.metric("Availability", f"{availability:.1f}%")

        if availability > 50:
            st.success("Parking Available ✅")
        else:
            st.error("Parking Almost Full 🚗")

    _,buffer=cv2.imencode(".jpg",image)

    st.download_button(
        "⬇ Download Result",
        buffer.tobytes(),
        "prediction.jpg",
        "image/jpeg"
    )