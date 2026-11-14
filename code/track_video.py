
import cv2
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

video_path = "assets/videos/traffic.mp4"
output_path = "outputs/tracked_traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Traffic video not found!")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize large frames for faster processing
scale = min(1.0, 960 / width)
new_width = int(width * scale)
new_height = int(height * scale)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (new_width, new_height)
)

unique_ids = set()
frame_count = 0

print("Starting vehicle and pedestrian tracking...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        classes=[0, 1, 2, 3, 5, 7],
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):
            unique_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)

            label = f"{model.names[class_id]} ID:{track_id}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Tracked IDs: {len(unique_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    out.write(frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")

cap.release()
out.release()

print("\nTracking completed!")
print("Total frames processed:", frame_count)
print("Unique tracking IDs:", len(unique_ids))
print("Output saved:", output_path)
