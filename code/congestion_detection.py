
import cv2
import os
from collections import deque
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

video_path = "assets/videos/traffic.mp4"
output_path = "outputs/congestion_detection.mp4"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Traffic video not found!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale = min(1.0, 960 / width)
new_width = int(width * scale)
new_height = int(height * scale)

# Road monitoring region
roi_y1 = int(new_height * 0.40)
roi_y2 = int(new_height * 0.90)

# Previous object positions
previous_positions = {}

# Smooth measurements
vehicle_history = deque(maxlen=15)
motion_history = deque(maxlen=15)

vehicle_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (new_width, new_height)
)

frame_count = 0

print("Starting congestion detection...")

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
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]

    current_positions = {}
    current_count = 0
    movements = []

    # Draw monitoring region
    cv2.rectangle(
        frame,
        (0, roi_y1),
        (new_width, roi_y2),
        (255, 255, 0),
        2
    )

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            inside_roi = roi_y1 <= center_y <= roi_y2

            if not inside_roi:
                continue

            current_count += 1
            current_positions[track_id] = (center_x, center_y)

            # Estimate movement in pixels per frame
            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                distance = (
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                ) ** 0.5

                movements.append(distance)

            # Draw tracking information
            label = f"{vehicle_names[class_id]} ID:{track_id}"

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

    previous_positions = current_positions

    # Calculate smoothed measurements
    vehicle_history.append(current_count)

    average_count = sum(vehicle_history) / len(vehicle_history)

    average_motion = (
        sum(movements) / len(movements)
        if movements else 0
    )

    motion_history.append(average_motion)

    smoothed_motion = sum(motion_history) / len(motion_history)

    # Prototype congestion classification
    if average_count <= 3 or smoothed_motion >= 8:
        congestion = "LOW"
        color = (0, 255, 0)

    elif average_count > 7 and smoothed_motion < 3:
        congestion = "HIGH"
        color = (0, 0, 255)

    else:
        congestion = "MEDIUM"
        color = (0, 255, 255)

    # Dashboard
    cv2.rectangle(
        frame,
        (10, 10),
        (440, 165),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "ROADVISION ",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Congestion: {congestion}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {current_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {smoothed_motion:.2f} px/frame",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    out.write(frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Processed {frame_count} frames | "
            f"Vehicles: {current_count} | "
            f"Congestion: {congestion}"
        )

cap.release()
out.release()

print("\nCONGESTION DETECTION COMPLETED!")
print("Frames processed:", frame_count)
print("Output:", output_path)
