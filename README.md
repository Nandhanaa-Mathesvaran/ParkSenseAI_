# 🅿️ ParkSense AI

An AI-powered Smart Parking Occupancy Detection System using **YOLOv8**, **OpenCV**, and **Streamlit**.

ParkSense AI detects parking spaces and classifies them as **Empty** or **Occupied** from parking lot images, helping automate parking availability monitoring.

Streamlit link: http://localhost:8501/ 

## 🚀 Features

- YOLOv8-based parking slot detection
- Detects empty and occupied parking spaces
- Bounding box visualization
- Displays:
  - Empty slot count
  - Occupied slot count
  - Parking availability
- Streamlit web interface

---

## 🧠 Model

**Algorithm:** YOLOv8 Object Detection

**Classes:**
- Empty
- Occupied

**Dataset:** PKLot Parking Dataset

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| Precision | 0.903 |
| Recall | 0.912 |
| mAP@50 | 0.877 |
| mAP@50-95 | 0.648 |

---

## 🛠 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- PyTorch

---

## 📂 Project Structure

```
ParkSenseAi/
│
├── app.py          # Streamlit Application
├── train.py        # Model Training
├── predict.py      # Prediction Script
├── data.yaml
├── dataset/
├── runs/
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ParkSenseAI.git
cd ParkSenseAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train Model

```bash
python train.py
```

---

## 🔍 Run Prediction

```bash
python predict.py
```

---

## 🌐 Run Web App

```bash
streamlit run app.py
```

---

## 🔮 Future Scope

- Real-time CCTV parking detection
- Mobile application integration
- Parking reservation system
- IoT-based smart parking
- Number plate recognition

---
