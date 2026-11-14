
import cv2
import os
from ultralytics import YOLO

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

video_path = "assets/videos/traffic.mp4"
output_path = "outputs/vehicle_counting.mp4"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Traffic video not found!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize for better CPU performance
scale = min(1.0, 960 / width)
new_width = int(width * scale)
new_height = int(height * scale)

line_y = int(new_height * 0.60)

# Vehicle class names
vehicle_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Vehicle counters
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

# Tracking memory
previous_centers = {}
counted_ids = set()

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (new_width, new_height)
)

frame_count = 0

print("Starting vehicle counting...")

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

    # Draw virtual counting line
    cv2.line(
        frame,
        (0, line_y),
        (new_width, line_y),
        (255, 0, 0),
        3
    )

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            if class_id not in vehicle_names:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Calculate object centre
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            vehicle_type = vehicle_names[class_id]

            # Check whether object crossed the line
            if track_id in previous_centers and track_id not in counted_ids:

                previous_y = previous_centers[track_id]

                crossed_down = (
                    previous_y < line_y and center_y >= line_y
                )

                crossed_up = (
                    previous_y > line_y and center_y <= line_y
                )

                if crossed_down or crossed_up:
                    vehicle_counts[vehicle_type] += 1
                    counted_ids.add(track_id)

                    print(
                        f"Counted {vehicle_type}, ID: {track_id}"
                    )

            previous_centers[track_id] = center_y

            # Draw bounding box and ID
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{vehicle_type} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Display vehicle count dashboard
    y_offset = 35

    cv2.putText(
        frame,
        "ROADVISION AI - VEHICLE COUNTING",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    for vehicle, count in vehicle_counts.items():
        y_offset += 30

        cv2.putText(
            frame,
            f"{vehicle}: {count}",
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    out.write(frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print("\nVEHICLE COUNTING COMPLETED!")
print("--------------------------------")

for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

print("--------------------------------")
print("Total vehicles counted:", sum(vehicle_counts.values()))
print("Output saved:", output_path)
