
import cv2
import os
from collections import deque
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

video_path = "assets/videos/traffic.mp4"
output_path = "outputs/traffic_density.mp4"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Traffic video not found!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize video for CPU performance
scale = min(1.0, 960 / width)
new_width = int(width * scale)
new_height = int(height * scale)

# Define road monitoring region
roi_x1 = 0
roi_x2 = new_width
roi_y1 = int(new_height * 0.40)
roi_y2 = int(new_height * 0.90)

# Vehicle classes
vehicle_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Smooth density changes over recent frames
density_history = deque(maxlen=15)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (new_width, new_height)
)

frame_count = 0

print("Starting traffic density estimation...")

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

    current_counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, class_id in zip(boxes, classes):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Count only objects inside road region
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and roi_y1 <= center_y <= roi_y2
            )

            if inside_roi and class_id in vehicle_names:

                vehicle_type = vehicle_names[class_id]
                current_counts[vehicle_type] += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # Total vehicles currently in road region
    total_vehicles = sum(current_counts.values())

    density_history.append(total_vehicles)

    # Average over recent frames
    average_vehicles = sum(density_history) / len(density_history)

    # Initial heuristic thresholds
    if average_vehicles <= 3:
        density = "LOW"
        density_color = (0, 255, 0)

    elif average_vehicles <= 7:
        density = "MEDIUM"
        density_color = (0, 255, 255)

    else:
        density = "HIGH"
        density_color = (0, 0, 255)

    # Draw road monitoring region
    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 255, 0),
        2
    )

    # Dashboard background
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 210),
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
        f"Traffic Density: {density}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        density_color,
        2
    )

    cv2.putText(
        frame,
        f"Vehicles in ROI: {total_vehicles}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y_offset = 145

    for vehicle, count in current_counts.items():

        cv2.putText(
            frame,
            f"{vehicle}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        y_offset += 18

    out.write(frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Processed {frame_count} frames | "
            f"Vehicles: {total_vehicles} | "
            f"Density: {density}"
        )

cap.release()
out.release()

print("\nTRAFFIC DENSITY ANALYSIS COMPLETED!")
print("Total frames:", frame_count)
print("Output saved:", output_path)

