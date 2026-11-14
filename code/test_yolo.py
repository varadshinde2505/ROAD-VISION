
import os
from pathlib import Path

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# These paths are passed by app.py
INPUT_VIDEO = Path(
    os.environ.get(
        "ROADVISION_INPUT_VIDEO",
        str(BASE_DIR / "assets" / "videos" / "traffic.mp4")
    )
)

OUTPUT_VIDEO = Path(
    os.environ.get(
        "ROADVISION_OUTPUT_VIDEO",
        str(BASE_DIR / "outputs" / "first_detection.mp4")
    )
)

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

print("ROADVISION  - VIDEO OBJECT DETECTION")
print("Input:", INPUT_VIDEO)
print("Output:", OUTPUT_VIDEO)

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

model = YOLO(str(BASE_DIR / "yolo11n.pt"))

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} | FPS: {fps}")
print("Total frames:", total_frames)

# H.264 encoding for browser-compatible playback
writer = imageio_ffmpeg.write_frames(
    str(OUTPUT_VIDEO),
    (width, height),
    fps=fps,
    codec="libx264",
    pix_fmt_in="rgb24",
    pix_fmt_out="yuv420p",
    output_params=["-movflags", "+faststart"],
    ffmpeg_log_level="error"
)

writer.send(None)

classes = [0, 1, 2, 3, 5, 7]
processed = 0
total_detections = 0
class_counts = {}

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            conf=0.30,
            classes=classes,
            verbose=False
        )

        annotated = results[0].plot()

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            name = model.names[class_id]

            class_counts[name] = class_counts.get(name, 0) + 1
            total_detections += 1

        # Convert OpenCV BGR to RGB for FFmpeg
        rgb_frame = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        writer.send(rgb_frame)
        processed += 1

        if processed % 30 == 0:
            print(f"Processed {processed}/{total_frames} frames")

finally:
    cap.release()
    writer.close()

if processed == 0:
    raise RuntimeError("No video frames were processed.")

if not OUTPUT_VIDEO.exists() or OUTPUT_VIDEO.stat().st_size == 0:
    raise RuntimeError("Output video was not created correctly.")

print("\nVIDEO DETECTION COMPLETED")
print("Frames processed:", processed)
print("Total frame-level detections:", total_detections)

for name, count in class_counts.items():
    print(f"{name}: {count}")

print("Saved:", OUTPUT_VIDEO)
