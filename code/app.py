
import subprocess
import imageio_ffmpeg
from pathlib import Path
from collections import Counter, deque
import math

import cv2
import numpy as np
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadVision Traffic",
    page_icon="🚘",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "assets" / "videos"
OUTPUTS = ROOT / "outputs"
WEIGHTS = ROOT / "yolo11n.pt"

VIDEOS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

CITY_FILES = {
    "Mumbai": VIDEOS / "mumbai.mp4",
    "Delhi": VIDEOS / "delhi.mp4",
    "Kolkata": VIDEOS / "kolkata.mp4",
}

MODULES = [
    "Object Detection",
    "Object Tracking",
    "Vehicle Counting",
    "Traffic Density",
    "Congestion Detection",
]

CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

VEHICLES = {2, 3, 5, 7}
VEHICLE_NAMES = ["Car", "Motorcycle", "Bus", "Truck"]

# Balanced processing settings
FRAME_SKIP = 2
INFERENCE_WIDTH = 960
CONFIDENCE = 0.30
IMGSZ = 512
MAX_DETECTIONS = 100


# ============================================================
# 2. HTML HELPER
# ============================================================

def render_html(content):
    clean_html = "\n".join(
        line.strip()
        for line in content.splitlines()
        if line.strip()
    )
    st.markdown(clean_html, unsafe_allow_html=True)


# ============================================================
# 3. CUSTOM DASHBOARD CSS
# ============================================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background:
    radial-gradient(ellipse at 8% 5%, rgba(0,220,255,.10), transparent 34%),
    radial-gradient(ellipse at 92% 18%, rgba(120,50,255,.12), transparent 36%),
    #080b12;
    color: #edf5ff;
    font-family: Inter, sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#101827,#090d16);
    border-right: 1px solid rgba(0,220,255,.22);
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

.hero {
    border: 1px solid rgba(0,220,255,.25);
    border-radius: 18px;
    background: linear-gradient(
        120deg,
        rgba(0,220,255,.12),
        rgba(100,60,255,.10),
        #0c111b
    );
    padding: 25px;
    margin-bottom: 25px;
}

.brand {
    font-family: Orbitron, sans-serif;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 2px;
}

.brand span {
    color: #00e5ff;
}

.sub {
    color: #9aadc5;
    font-size: 11px;
    letter-spacing: 3px;
    margin-top: 10px;
}

.cyan {
    color: #00e5ff;
}

.section {
    font-family: Orbitron, sans-serif;
    font-size: 12px;
    color: #00e5ff;
    letter-spacing: 2px;
    margin: 28px 0 14px;
}

.panel {
    border: 1px solid rgba(0,220,255,.22);
    border-radius: 16px;
    background: #101827;
    padding: 22px;
    margin-bottom: 15px;
}

.metric {
    border: 1px solid rgba(0,220,255,.20);
    border-radius: 15px;
    background: #111a28;
    padding: 18px;
    min-height: 100px;
}

.metric small {
    color: #91a6c0;
    font-size: 12px;
}

.metric strong {
    display: block;
    color: #54eaff;
    font-family: Orbitron, sans-serif;
    font-size: 24px;
    margin-top: 10px;
}

.stButton > button {
    background: linear-gradient(90deg,#00bcd4,#6955ff);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}

.stButton > button:hover {
    border: 1px solid #00e5ff;
    color: white;
    box-shadow: 0 0 15px rgba(0,220,255,.25);
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,220,255,.4);
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""")


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def output_path(city, module):

    city_slug = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in city
    ).strip("_")

    module_slug = {
        "Object Detection": "detection",
        "Object Tracking": "tracking",
        "Vehicle Counting": "counting",
        "Traffic Density": "density",
        "Congestion Detection": "congestion",
    }[module]

    return OUTPUTS / f"{city_slug}_{module_slug}_balanced.mp4"


def draw_label(frame, text, x, y, color=(0, 235, 255)):

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.78
    thickness = 2
    padding_x = 12
    padding_y = 9

    (tw, th), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    box_w = tw + padding_x * 2
    box_h = th + baseline + padding_y * 2

    x = max(0, min(int(x), frame.shape[1] - box_w))
    y = max(box_h, int(y))

    cv2.rectangle(
        frame,
        (x, y - box_h),
        (x + box_w, y),
        color,
        -1
    )

    cv2.rectangle(
        frame,
        (x, y - box_h),
        (x + box_w, y),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        text,
        (x + padding_x, y - padding_y - baseline),
        font,
        scale,
        (8, 12, 20),
        thickness,
        cv2.LINE_AA
    )


def create_writer(path, fps, size):

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "Could not create output video. Check OpenCV codec support."
        )

    return writer


def metric(title, value):

    render_html(f"""
    <div class="metric">
        <small>{title}</small>
        <strong>{value}</strong>
    </div>
    """)


# ============================================================
# 5. VIDEO PROCESSING ENGINE
# ============================================================

def process_video(source, city, module, progress, status):

    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            "yolo11n.pt is missing from the project folder."
        )

    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(
            f"Input video not found: {source}"
        )

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {source.name}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps < 1:
        fps = 25.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    ok, first_frame = cap.read()

    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(
            "Video contains no readable frames."
        )

    original_h, original_w = first_frame.shape[:2]

    scale = min(
        1.0,
        INFERENCE_WIDTH / float(original_w)
    )

    width = max(2, int(original_w * scale))
    height = max(2, int(original_h * scale))

    width -= width % 2
    height -= height % 2

    final_path = output_path(city, module)

    temp_path = final_path.with_name(
        final_path.stem + "_working.mp4"
    )

    temp_path.unlink(missing_ok=True)

    writer = create_writer(
        temp_path,
        fps,
        (width, height)
    )

    frame_index = 0
    inference_frames = 0

    unique_vehicle_ids = set()

    unique_ids_by_class = {
        name: set() for name in VEHICLE_NAMES
    }

    all_track_ids = set()
    crossed_ids = set()

    previous_side = {}
    previous_centres = {}

    motion_history = deque(maxlen=100)
    sampled_vehicle_counts = []

    peak_vehicles = 0
    line_y = int(height * 0.62)

    last_annotated = None
    last_density = "LOW"
    last_congestion = "LOW"

    try:

        frame = first_frame

        while frame is not None:

            if (
                frame.shape[1] != width
                or frame.shape[0] != height
            ):
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )

            should_infer = (
                frame_index % FRAME_SKIP == 0
                or last_annotated is None
            )

            if should_infer:

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONFIDENCE,
                    iou=0.50,
                    imgsz=IMGSZ,
                    max_det=MAX_DETECTIONS,
                    classes=list(CLASSES.keys()),
                    verbose=False,
                    device="cpu"
                )[0]

                annotated = frame.copy()

                frame_counts = Counter()
                current_centres = {}

                boxes = result.boxes

                if boxes is not None:

                    ids_tensor = boxes.id

                    for box_index, box in enumerate(boxes):

                        class_id = int(
                            box.cls[0].item()
                        )

                        class_name = CLASSES.get(
                            class_id,
                            "Object"
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        x1, y1, x2, y2 = [
                            int(v)
                            for v in box.xyxy[0].tolist()
                        ]

                        x1 = max(0, min(x1, width - 1))
                        x2 = max(0, min(x2, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        y2 = max(0, min(y2, height - 1))

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        frame_counts[class_name] += 1

                        track_id = None

                        if (
                            ids_tensor is not None
                            and box_index < len(ids_tensor)
                        ):
                            track_id = int(
                                ids_tensor[box_index].item()
                            )

                        if track_id is not None:

                            all_track_ids.add(track_id)

                            if class_id in VEHICLES:

                                unique_vehicle_ids.add(track_id)

                                unique_ids_by_class[
                                    class_name
                                ].add(track_id)

                                current_centres[track_id] = (
                                    cx,
                                    cy
                                )

                        color = (
                            (0, 235, 110)
                            if class_id in VEHICLES
                            else (255, 190, 40)
                        )

                        display_text = (
                            f"{class_name} ID:{track_id}"
                            if track_id is not None
                            else f"{class_name} {confidence:.2f}"
                        )

                        # Bigger, thicker bounding boxes
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            4
                        )

                        # Bigger vehicle ID labels
                        draw_label(
                            annotated,
                            display_text,
                            x1,
                            max(y1, 55),
                            color
                        )

                # VEHICLE COUNTING

                if module == "Vehicle Counting":

                    cv2.line(
                        annotated,
                        (0, line_y),
                        (width, line_y),
                        (0, 210, 255),
                        4
                    )

                    for track_id, (cx, cy) in current_centres.items():

                        side = (
                            "above"
                            if cy < line_y
                            else "below"
                        )

                        old_side = previous_side.get(track_id)

                        if (
                            old_side is not None
                            and old_side != side
                        ):
                            crossed_ids.add(track_id)

                        previous_side[track_id] = side

                    draw_label(
                        annotated,
                        f"VEHICLES CROSSED: {len(crossed_ids)}",
                        20,
                        70,
                        (0, 210, 255)
                    )

                # MOTION TRACKING

                for track_id, centre in current_centres.items():

                    old_centre = previous_centres.get(track_id)

                    if old_centre is not None:
                        motion_history.append(
                            math.dist(centre, old_centre)
                        )

                    previous_centres[track_id] = centre

                # FRAME-WISE VEHICLE COUNT

                vehicle_count = sum(
                    frame_counts.get(CLASSES[c], 0)
                    for c in VEHICLES
                )

                sampled_vehicle_counts.append(
                    vehicle_count
                )

                peak_vehicles = max(
                    peak_vehicles,
                    vehicle_count
                )

                # DENSITY ESTIMATION

                density_percent = min(
                    100,
                    int((vehicle_count / 15.0) * 100)
                )

                last_density = (
                    "HIGH"
                    if density_percent >= 70
                    else "MEDIUM"
                    if density_percent >= 35
                    else "LOW"
                )

                # CONGESTION ESTIMATION

                average_motion = (
                    float(np.mean(motion_history))
                    if motion_history
                    else 0.0
                )

                last_congestion = (
                    "HIGH"
                    if vehicle_count >= 10 and average_motion < 3.0
                    else "MEDIUM"
                    if vehicle_count >= 5 and average_motion < 5.0
                    else "LOW"
                )

                # MODULE LABELS

                if module == "Object Detection":

                    draw_label(
                        annotated,
                        f"VEHICLES IN FRAME: {vehicle_count}",
                        20,
                        70,
                        (0, 220, 255)
                    )

                elif module == "Object Tracking":

                    draw_label(
                        annotated,
                        f"ACTIVE TRACKS: {len(current_centres)}",
                        20,
                        70,
                        (0, 220, 255)
                    )

                elif module == "Traffic Density":

                    draw_label(
                        annotated,
                        f"DENSITY: {last_density}",
                        20,
                        70,
                        (0, 220, 255)
                    )

                elif module == "Congestion Detection":

                    draw_label(
                        annotated,
                        f"CONGESTION: {last_congestion}",
                        20,
                        70,
                        (0, 220, 255)
                    )

                last_annotated = annotated
                inference_frames += 1

            # Export every original frame.
            # Detection runs on every second frame.

            if should_infer:
                writer.write(last_annotated)
            else:
                writer.write(frame)

            frame_index += 1

            if frame_index % 10 == 0:

                if total_frames > 0:
                    progress.progress(
                        min(frame_index / total_frames, 0.99)
                    )

                status.caption(
                    f"{city}: {frame_index:,} frames exported | "
                    f"{inference_frames:,} frames analysed"
                )

            ok, next_frame = cap.read()

            if not ok:
                break

            frame = next_frame

    finally:

        cap.release()
        writer.release()

    if (
        frame_index == 0
        or not temp_path.is_file()
        or temp_path.stat().st_size < 1000
    ):

        temp_path.unlink(missing_ok=True)

        raise RuntimeError(
            "No valid processed video was generated."
        )

    # Convert MP4V to browser-compatible H.264

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(temp_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(final_path)
    ]

    conversion = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if conversion.returncode != 0:

        temp_path.unlink(missing_ok=True)

        raise RuntimeError(
            "H.264 conversion failed: "
            + conversion.stderr[-1500:]
        )

    temp_path.unlink(missing_ok=True)

    progress.progress(1.0)

    average_vehicles = (
        round(float(np.mean(sampled_vehicle_counts)), 2)
        if sampled_vehicle_counts
        else 0
    )

    stats = {
        "output_frames": frame_index,
        "inference_frames": inference_frames,
        "unique_vehicles": len(unique_vehicle_ids),
        "unique_track_ids": len(all_track_ids),
        "crossed": len(crossed_ids),

        "unique_by_class": {
            name: len(unique_ids_by_class[name])
            for name in VEHICLE_NAMES
        },

        "average_vehicles": average_vehicles,
        "peak_vehicles": peak_vehicles,
        "density": last_density,
        "congestion": last_congestion,

        "size_mb": round(
            final_path.stat().st_size / (1024 * 1024),
            2
        ),

        "fps": round(fps, 2),

        "duration_seconds": round(
            frame_index / fps,
            2
        ),
    }

    return final_path, stats


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div class="brand">
        🚘 ROADVISION
    </div>
    <p class="sub">TRAFFIC COMMAND CENTER</p>
    """)

    st.divider()

    page = st.radio(
        "CONTROL CENTER",
        ["Overview"] + MODULES
    )

    st.divider()

    if WEIGHTS.is_file():
        st.success("YOLO11 weights found")
    else:
        st.error("yolo11n.pt missing")

    st.success("Balanced video pipeline ready")

    st.caption(
        f"BALANCED MODE · 1 analysis pass / {FRAME_SKIP} frames"
    )


# ============================================================
# 7. MAIN HEADER
# ============================================================

render_html("""
<div class="hero">
    <div class="brand">
        ROADVISION <span>TRAFFIC</span>
    </div>

    <div class="sub">
        INTELLIGENT TRAFFIC MONITORING · AUTONOMOUS MOBILITY
    </div>

    <br>

    <span style="color:#00efa5">
        ● BALANCED CITY VIDEO PROCESSING SYSTEM
    </span>
</div>
""")


# ============================================================
# 8. CITY SELECTION
# ============================================================

render_html("""
<div class="section">
    01 / SELECT CITY TRAFFIC FEED
</div>
""")

col_city, col_load = st.columns([3, 1])

with col_city:

    selected_city = st.selectbox(
        "SELECT CITY",
        list(CITY_FILES.keys())
    )

with col_load:

    st.write("")

    load_clicked = st.button(
        "LOAD CITY VIDEO",
        use_container_width=True
    )

if load_clicked:

    selected_source = CITY_FILES[selected_city]

    if (
        selected_source.is_file()
        and selected_source.stat().st_size > 0
    ):

        st.session_state["active_city"] = selected_city

        st.session_state["active_video"] = str(
            selected_source
        )

        st.session_state["last_result"] = None

        st.success(
            f"{selected_city} video loaded successfully."
        )

    else:

        st.error(
            f"{selected_source.name} not found in assets/videos/."
        )


# ============================================================
# 9. CUSTOM VIDEO UPLOAD
# ============================================================

with st.expander("UPLOAD YOUR OWN TRAFFIC VIDEO"):

    uploaded_video = st.file_uploader(
        "Choose MP4, MOV or AVI",
        type=["mp4", "mov", "avi"]
    )

    if (
        uploaded_video is not None
        and st.button("USE UPLOADED VIDEO")
    ):

        custom_path = VIDEOS / "custom_upload.mp4"

        custom_path.write_bytes(
            uploaded_video.getbuffer()
        )

        st.session_state["active_city"] = "Custom Upload"

        st.session_state["active_video"] = str(
            custom_path
        )

        st.session_state["last_result"] = None

        st.success("Custom video selected.")


# ============================================================
# 10. ACTIVE VIDEO
# ============================================================

active_city = st.session_state.get("active_city")

active_video = (
    Path(st.session_state["active_video"])
    if st.session_state.get("active_video")
    else None
)

if (
    active_city
    and active_video
    and active_video.is_file()
):

    render_html("""
    <div class="section">
        02 / SELECTED INPUT VIDEO
    </div>
    """)

    input_col, info_col = st.columns([3, 1])

    with input_col:

        st.video(str(active_video))

    with info_col:

        render_html(f"""
        <div class="panel">
            <small>ACTIVE CITY</small>
            <h2 class="cyan">{active_city.upper()}</h2>
            <small>{active_video.name}</small>
            <p style="color:#00efa5">
                ● INPUT CONNECTED
            </p>
        </div>
        """)

else:

    st.info(
        "Select Mumbai, Delhi or Kolkata and click LOAD CITY VIDEO."
    )


# ============================================================
# 11. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    render_html("""
    <div class="section">
        03 / SYSTEM OVERVIEW
    </div>
    """)

    columns = st.columns(4)

    overview_data = [
        ("Detection", "YOLO11n"),
        ("Tracking", "ByteTrack"),
        ("Video Engine", "OpenCV"),
        ("Processing", "BALANCED MODE"),
    ]

    for col, (title, value) in zip(columns, overview_data):

        with col:
            metric(title, value)

    st.markdown("### AVAILABLE MODULES")

    st.write(
        "Object Detection · Object Tracking · Vehicle Counting · "
        "Traffic Density · Congestion Detection"
    )

    st.info(
        "Balanced Mode analyses every second frame at up to 960px. "
        "Skipped frames are exported without detection overlays. "
        "The output retains the original video's FPS and duration."
    )


# ============================================================
# 12. ANALYSIS MODULES
# ============================================================

else:

    descriptions = {
        "Object Detection":
            "Detect road users and annotate the selected city's video.",

        "Object Tracking":
            "Track road users over time using persistent ByteTrack IDs.",

        "Vehicle Counting":
            "Count unique tracked vehicles crossing the virtual line.",

        "Traffic Density":
            "Estimate vehicle density using sampled video frames.",

        "Congestion Detection":
            "Estimate congestion using vehicle volume and movement.",
    }

    render_html(f"""
    <div class="section">
        03 / {page.upper()}
    </div>

    <div class="panel">
        <h3>{page.upper()}</h3>
        <p>{descriptions[page]}</p>
        <span class="cyan">
            YOLO11 NANO · BYTETRACK · BALANCED MODE
        </span>
    </div>
    """)

    if (
        not active_city
        or not active_video
        or not active_video.is_file()
    ):

        st.warning(
            "Load a city video before running this module."
        )

    else:

        st.success(
            f"Input source: {active_city} · {active_video.name}"
        )

        if st.button(
            f"▶ RUN {page.upper()}",
            use_container_width=True,
            key=f"run_{page}"
        ):

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:

                with st.spinner(
                    f"Processing {active_city} video..."
                ):

                    processed_path, result_stats = process_video(
                        active_video,
                        active_city,
                        page,
                        progress_bar,
                        status_text
                    )

                st.session_state["last_result"] = {
                    "city": active_city,
                    "module": page,
                    "path": str(processed_path),
                    "stats": result_stats
                }

                status_text.empty()

                st.success(
                    f"{page} completed for {active_city}."
                )

            except Exception as error:

                status_text.empty()

                st.error(
                    f"Processing failed: {error}"
                )


    # ========================================================
    # 13. PROCESSED OUTPUT
    # ========================================================

    render_html("""
    <div class="section">
        04 / PROCESSED OUTPUT
    </div>
    """)

    last_result = st.session_state.get("last_result")

    if (
        last_result
        and last_result.get("city") == active_city
        and last_result.get("module") == page
        and Path(last_result.get("path", "")).is_file()
    ):

        processed_file = Path(last_result["path"])
        stats = last_result["stats"]

        st.video(str(processed_file))

        st.success(
            f"Processed output ready: {processed_file.name}"
        )

        st.caption(
            f"{stats['output_frames']:,} output frames · "
            f"{stats['inference_frames']:,} analysed frames · "
            f"{stats['fps']} FPS · "
            f"{stats['duration_seconds']} seconds · "
            f"{stats['size_mb']} MB"
        )

        # MAIN METRICS

        top_metrics = st.columns(4)

        with top_metrics[0]:
            metric(
                "Unique Vehicles",
                stats["unique_vehicles"]
            )

        with top_metrics[1]:
            metric(
                "Vehicles Crossed",
                stats["crossed"]
            )

        with top_metrics[2]:
            metric(
                "Average Vehicles / Frame",
                stats["average_vehicles"]
            )

        with top_metrics[3]:
            metric(
                "Peak Vehicles / Frame",
                stats["peak_vehicles"]
            )

        # VEHICLE BREAKDOWN

        st.markdown("### UNIQUE VEHICLE BREAKDOWN")

        class_columns = st.columns(4)

        for col, vehicle_name in zip(
            class_columns,
            VEHICLE_NAMES
        ):

            with col:

                metric(
                    vehicle_name,
                    stats["unique_by_class"][vehicle_name]
                )

        # DENSITY RESULT

        if page == "Traffic Density":

            st.info(
                f"Final sampled-frame density estimate: "
                f"{stats['density']}. "
                "This is a relative visual estimate."
            )

        # CONGESTION RESULT

        if page == "Congestion Detection":

            st.info(
                f"Final sampled-frame congestion estimate: "
                f"{stats['congestion']}. "
                "This is a heuristic based on visible vehicle "
                "volume and movement."
            )

        # DOWNLOAD OUTPUT

        st.download_button(
            "⬇ DOWNLOAD PROCESSED VIDEO",
            data=processed_file.read_bytes(),
            file_name=processed_file.name,
            mime="video/mp4",
            use_container_width=True
        )

    else:

        st.info(
            "No output for this city and module yet. "
            "Click RUN to process the selected input video."
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

render_html("""
<p style="
    text-align:center;
    color:#71849d;
    font-family:Orbitron;
    letter-spacing:2px;
">
    ROADVISION · BUILT FOR THE FUTURE OF AUTONOMOUS MOBILITY
</p>
""")
