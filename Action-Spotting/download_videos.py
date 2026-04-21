import csv
import subprocess
import os

CSV_PATH = "data/tennis/videos.csv"
OUTPUT_DIR = "data/tennis/videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        name = row["name"]
        yt_id = row["yt_id"]

        url = f"https://www.youtube.com/watch?v={yt_id}"
        output_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")

        print(f"Downloading {name}...")

        fps = float(row["fps"])
        fps_int = int(round(fps))

        cmd = [
            "yt-dlp",
            "-f", f"bestvideo[height=1080][fps={fps_int}][vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height=1080][fps={fps_int}][vcodec^=avc]+bestaudio",
            "--merge-output-format", "mp4",
            "-o", os.path.join(os.path.abspath(OUTPUT_DIR), f"{name}.mp4"),
            url,
        ]
        subprocess.run(cmd)