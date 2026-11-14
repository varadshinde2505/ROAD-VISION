
import subprocess
import imageio_ffmpeg
from pathlib import Path

output_dir = Path("outputs")
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

videos = [
    "tracked_traffic.mp4",
    "vehicle_counting.mp4",
    "traffic_density.mp4",
    "congestion_detection.mp4"
]

for filename in videos:

    source = output_dir / filename

    if not source.exists():
        print(f"Skipping missing file: {filename}")
        continue

    destination = output_dir / f"{source.stem}_web.mp4"

    print(f"\nConverting: {filename}")

    command = [
        ffmpeg,
        "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(destination)
    ]

    subprocess.run(command, check=True)

    print(f"Saved: {destination}")

print("\nAll available videos converted!")
