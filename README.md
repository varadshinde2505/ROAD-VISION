# 🚦 RoadVision AI
### Intelligent Traffic Monitoring & Vehicle Analytics System

RoadVision AI is a computer-vision-based traffic monitoring application designed to analyse road traffic videos using **YOLO11, ByteTrack, OpenCV, and Streamlit**.

It provides vehicle detection, object tracking, vehicle counting, traffic density estimation, and congestion analysis through an interactive dashboard.

## ✨ Features

- **Object Detection:** Detects cars, motorcycles, buses, trucks, bicycles, and people.
- **Object Tracking:** Tracks detected objects using ByteTrack IDs.
- **Vehicle Counting:** Counts tracked vehicles crossing a virtual line.
- **Traffic Density:** Estimates traffic density from detected vehicles.
- **Congestion Detection:** Estimates congestion using vehicle volume and movement.
- **Video Processing:** Processes uploaded or preloaded city traffic videos.
- **Interactive Dashboard:** Streamlit-based interface with processed video playback and download.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLO11 | Object detection |
| ByteTrack | Object tracking |
| OpenCV | Video processing |
| Streamlit | Interactive dashboard |
| NumPy | Numerical operations |
| FFmpeg | H.264 video encoding |

## 📁 Project Structure

```text
RoadVision-AI/
│
├── app.py
├── test_yolo.py
├── track_video.py
├── traffic_density.py
├── congestion_detection.py
├── convert_videos.py
├── yolo11n.pt
│
├── assets/
│   ├── images/
│   │   └── road_scene.png
│   └── videos/
│
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RoadVision-AI.git
cd RoadVision-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the YOLO11 model

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Ensure `yolo11n.pt` is available in the project root.

### 5. Add traffic videos

Place your video files inside:

```text
assets/videos/
```

Expected filenames:

- `mumbai.mp4`
- `delhi.mp4`
- `kolkata.mp4`

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚘 Dashboard Modules

1. Object Detection
2. Object Tracking
3. Vehicle Counting
4. Traffic Density Estimation
5. Congestion Detection

## 📊 Processing Pipeline

```text
Traffic Video
     |
     v
Frame Extraction
     |
     v
YOLO11 Object Detection
     |
     v
ByteTrack Object Tracking
     |
     v
Vehicle Analytics
     |
     v
Traffic Insights
     |
     v
Streamlit Dashboard
     |
     v
Processed Video Output
```

## 🖥️ Demo

Add your dashboard screenshot and processed traffic video here.

Example:

```markdown
![RoadVision AI Dashboard](assets/images/road_scene.png)
```

## 🔮 Future Improvements

- Real-time CCTV camera integration
- Automatic vehicle speed estimation
- Traffic signal optimisation
- Number plate recognition
- Multi-camera vehicle tracking
- AI-based traffic prediction
- City-wide traffic analytics

## ⚠️ Notes

- Detection performance depends on video quality, camera angle, lighting, and hardware.
- Traffic density and congestion outputs are estimates, not official traffic measurements.
- The project uses a balanced processing configuration to reduce inference time.
- Large video files and generated outputs are intentionally excluded from Git.

## 👨‍💻 Author

**Varad Shinde**

Electronics & Telecommunication Engineering

GitHub: https://github.com/varadshinde2505

---

*RoadVision AI — Building intelligent solutions for the future of urban mobility.*
