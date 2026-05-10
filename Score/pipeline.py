#!/usr/bin/env python3
"""End-to-end tennis analysis pipeline."""

import os
import subprocess
import cv2
import tempfile
import argparse
import sys
import json
import gzip

# ---------------------------------------------------------------------------
# Config (AS = Action-Spotting)
# ---------------------------------------------------------------------------
AS_DIR = os.path.join('..', 'Action-Spotting')
TRACKING_DIR = os.path.join('..', 'Tracking')

VIDEO_DIR = 'data/tennis/videos'
FRAMES_DIR = 'data/tennis/frames'

TRACKING_YOLO_WEIGHTS = os.path.join(TRACKING_DIR, 'checkpoints', 'yolov8m.pt')
TRACKING_TRACKNET_WEIGHTS = os.path.join(TRACKING_DIR, 'checkpoints', 'tracknet', 'model_best.pt')
TRACKING_COURT_WEIGHTS = os.path.join(TRACKING_DIR, 'checkpoints', 'court', 'model_tennis_court_det.pt')
TRACKING_BALL_CHUNK = 4  # lower = less GPU memory

TRACKING_W, TRACKING_H = 1280, 720
AS_W, AS_H = 1920, 1080
# ---------------------------------------------------------------------------


def probe_video(video_path):
    """Return (num_frames, fps) from a video file using OpenCV."""
    vc = cv2.VideoCapture(video_path)
    assert vc.isOpened(), f'Could not open video: {video_path}'
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    vc.release()
    return num_frames, fps


def count_frames_on_disk(frame_dir):
    """Count actual extracted frames — more reliable than OpenCV metadata."""
    return len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])


def extract_frames(video_path, out_dir):
    _, fps = probe_video(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_out = os.path.join(out_dir, video_name)

    if os.path.exists(frame_out):
        actual_frames = count_frames_on_disk(frame_out)
        print(f'[Step 1/5] Frames already exist at {frame_out} ({actual_frames} frames), skipping extraction.')
        return frame_out, actual_frames, fps

    print(f'[Step 1/5] Extracting frames from {video_path} at {fps:.2f} fps...')
    subprocess.run([
        'python', os.path.join(AS_DIR, 'frames_as_jpg.py'),
        '--single_video', video_path,
        '--num_frames', str(999999),
        '--fps', str(fps),
        '-o', out_dir
    ], check=True, cwd=AS_DIR, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

    actual_frames = count_frames_on_disk(frame_out)
    print(f'[Step 1/5] Done. Extracted {actual_frames} frames to {frame_out}.')
    return frame_out, actual_frames, fps


def get_tracking_video(video_path):
    """
    If the video is already 720p, return it as-is.
    Otherwise, write a resized copy to a temp file and return that path.
    The caller is responsible for deleting the temp file when done.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if (w, h) == (TRACKING_W, TRACKING_H):
        print('Video already 720p, no resizing needed for tracking.')
        return video_path, False  # False = not a temp file

    print(f'Resizing {w}x{h} → {TRACKING_W}x{TRACKING_H} for tracking (in-process, no disk ffmpeg)...')
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()

    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (TRACKING_W, TRACKING_H)
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(cv2.resize(frame, (TRACKING_W, TRACKING_H)))
    cap.release()
    writer.release()
    print(f'Resized video written to temp file: {tmp_path}')
    return tmp_path, True  # True = caller should delete


def run_action_spotting(video_name, frame_dir, model_name, num_frames, fps, out_dir,
                        pred_file=None, force=False):
    """
    video_name: stem only, no extension (e.g. 'test_clip')
    frame_dir:  absolute path to the frames root directory
    out_dir:    absolute path where predictions will be saved
    force:      rerun even if output already exists
    """
    if pred_file is None:
        pred_file = os.path.join(out_dir, 'action_spotting')

    expected_output = pred_file + '.json'
    if os.path.exists(expected_output) and not force:
        print(f'[Step 2/5] Action spotting predictions already exist at {expected_output}, skipping.')
        return
    if os.path.exists(expected_output) and force:
        print(f'[Step 2/5] Action spotting already exists but --force-action-spotting set, recomputing...')

    cmd = [
        'python', 'test_e2e.py',
        os.path.join('checkpoints', model_name),
        frame_dir,
        '--single_video', video_name,
        '--num_frames', str(num_frames),
        '--fps', str(fps),
        '--save_as', pred_file,
    ]
    subprocess.run(cmd, check=True, cwd=AS_DIR, env={**os.environ, 'PYTHONUNBUFFERED': '1'})


def run_tracking(video_path, out_dir, ball_chunk=TRACKING_BALL_CHUNK,
                 force=False, no_players=False, no_ball=False, no_court=False):
    """
    Run the tracking step.

    force:      rerun even if tracking.json already exists
    no_players: skip player tracking
    no_ball:    skip ball tracking
    no_court:   skip court detection
    """
    output_path = os.path.join(out_dir, 'tracking.json')

    if os.path.exists(output_path) and not force:
        print(f'[Step 3/5] Tracking predictions already exist at {output_path}, skipping.')
        return output_path
    if os.path.exists(output_path) and force:
        print(f'[Step 3/5] Tracking already exists but --force-tracking set, recomputing...')

    print(f'[Step 3/5] Preparing tracking video (resizing to 720p if needed)...')
    tracking_video_path, is_temp = get_tracking_video(video_path)
    try:
        # Build which components to skip
        skip_flags = []
        if no_players:
            skip_flags.append('--no-players')
        if no_ball:
            skip_flags.append('--no-ball')
        if no_court:
            skip_flags.append('--no-court')

        skipped = ', '.join(f[2:] for f in skip_flags) if skip_flags else 'none'
        print(f'[Step 3/5] Running tracking on {tracking_video_path} (skipping: {skipped})...')

        cmd = [
            'python', '-m', 'scripts.track',
            '--video', tracking_video_path,
            '--output', output_path,
            '--yolo-weights', os.path.abspath(TRACKING_YOLO_WEIGHTS),
            '--tracknet-weights', os.path.abspath(TRACKING_TRACKNET_WEIGHTS),
            '--court-weights', os.path.abspath(TRACKING_COURT_WEIGHTS),
            '--ball-chunk', str(ball_chunk),
            '--original-video', video_path,
            '--original-width',  str(AS_W),
            '--original-height', str(AS_H),
            *skip_flags,
        ]
        subprocess.run(cmd, check=True, cwd=TRACKING_DIR,
                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        print(f'[Step 3/5] Tracking done. Results in {output_path}.')

        if (AS_W, AS_H) != (TRACKING_W, TRACKING_H):
            print(f'[Step 3/5] Scaling tracking coordinates to {AS_W}x{AS_H}...')
            scale_tracking_to_1080p(output_path,
                src_w=TRACKING_W, src_h=TRACKING_H,
                dst_w=AS_W, dst_h=AS_H)
    finally:
        if is_temp:
            os.remove(tracking_video_path)
            print(f'[Step 3/5] Removed temp tracking video: {tracking_video_path}')

    return output_path


def scale_tracking_to_1080p(tracking_json_path, src_w=1280, src_h=720, dst_w=1920, dst_h=1080):
    """Scale all pixel coordinates in the tracking JSON from 720p to 1080p in-place."""
    sx = dst_w / src_w
    sy = dst_h / src_h

    with open(tracking_json_path) as f:
        data = json.load(f)

    for frame in data['frames']:
        for player in frame['players']:
            x1, y1, x2, y2 = player['bbox']
            player['bbox'] = [x1*sx, y1*sy, x2*sx, y2*sy]

        ball = frame.get('ball')
        if ball:
            ball['x'] *= sx
            ball['y'] *= sy

    for frame in data['frames']:
        if frame.get("court_keypoints"):
            frame["court_keypoints"] = [
                None if kp is None else [kp[0] * sx, kp[1] * sy]
                for kp in frame["court_keypoints"]
            ]

    data['width'] = dst_w
    data['height'] = dst_h

    with open(tracking_json_path, 'w') as f:
        json.dump(data, f)

    print(f'Scaled tracking coordinates from {src_w}x{src_h} to {dst_w}x{dst_h}.')


def run_score(pred_dir, video_stem, out_dir,
              initial_sets=None, initial_games=None, initial_points=None, force=False):
    tracking_path = os.path.join(pred_dir, 'tracking.json')
    preds_path = os.path.join(pred_dir, 'action_spotting.json')
    output_path = os.path.join(out_dir, 'score.json')

    if os.path.exists(output_path) and not force:
        print(f'[Step 4/5] Score already exists at {output_path}, skipping. Use --force-score to recompute.')
        return output_path
    if os.path.exists(output_path) and force:
        print(f'[Step 4/5] Score already exists but --force-score set, recomputing...')
    else:
        print(f'[Step 4/5] Computing score...')

    cmd = [
        'python', '-m', 'scripts.score',
        '--tracking', tracking_path,
        '--preds', preds_path,
        '--output', output_path,
    ]
    if initial_sets:
        cmd += ['--initial-sets', initial_sets]
    if initial_games:
        cmd += ['--initial-games', initial_games]
    if initial_points:
        cmd += ['--initial-points', initial_points]
    subprocess.run(cmd, check=True, cwd=os.path.abspath('.'),
                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    print(f'[Step 4/5] Score done. Results in {output_path}.')
    return output_path

def run_visualization(pred_dir, video_stem, force=False):
    tracking_path = os.path.join(pred_dir, 'tracking.json')
    out_dir = pred_dir
    sample_frames_path = os.path.join(out_dir, 'sample_frames.png')

    if os.path.exists(sample_frames_path) and not force:
        print(f'[Step 5/5] Visualization already exists at {out_dir}, skipping. Use --force-viz to rerun.')
        return
    if os.path.exists(sample_frames_path) and force:
        print(f'[Step 5/5] Visualization already exists but --force-viz set, recomputing...')
    else:
        print(f'[Step 5/5] Running visualization...')

    cmd = [
        'python', '-m', 'scripts.visualize',
        '--tracks', tracking_path,
        '--out-dir', out_dir,
    ]
    subprocess.run(cmd, check=True, cwd=TRACKING_DIR,
                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    print(f'[Step 5/5] Visualization done. Results in {out_dir}.')

def check_cuda():
    """Abort early if no CUDA GPU is available."""
    import torch
    if not torch.cuda.is_available():
        print('ERROR: No CUDA GPU detected. Aborting.')
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU detected: {gpu_name} — OK.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', required=True,
                        help='Name of the video to process (e.g. match.mp4).')
    parser.add_argument('--as_model_name', required=True,
                        choices=['tennis_rny002gsm_gru_rgb', 'tennis_rny008gsm_gru_rgb'],
                        help='Name of the pre-trained model for Action-Spotting.')

    # Force flags
    parser.add_argument('--force-score', action='store_true',
                        help='Recompute score even if score.json already exists.')
    parser.add_argument('--force-tracking', action='store_true',
                        help='Rerun tracking even if tracking.json already exists.')
    parser.add_argument('--force-action-spotting', action='store_true',
                        help='Rerun action spotting even if action_spotting.json already exists.')

    # Selective tracking components
    parser.add_argument('--no-players', action='store_true',
                        help='Skip player tracking inside the tracking step.')
    parser.add_argument('--no-ball', action='store_true',
                        help='Skip ball tracking inside the tracking step.')
    parser.add_argument('--no-court', action='store_true',
                        help='Skip court detection inside the tracking step.')

    # Initial score state
    parser.add_argument('--initial-sets',   default=None,
                        help="Initial set score as FAR:NEAR, e.g. '1:0'")
    parser.add_argument('--initial-games',  default=None,
                        help="Initial game score as FAR:NEAR, e.g. '3:2'")
    parser.add_argument('--initial-points', default=None,
                        help="Initial point score as FAR:NEAR, e.g. '2:1'")

    # Visualization
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization step.')
    parser.add_argument('--force-viz', action='store_true',
                        help='Rerun visualization even if outputs already exist.')

    return parser.parse_args()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()

    check_cuda()

    video_path = os.path.abspath(os.path.join(VIDEO_DIR, args.video_name))
    video_stem = os.path.splitext(args.video_name)[0]

    print(f'\n{"="*60}')
    print(f'Pipeline started for: {args.video_name}')
    print(f'{"="*60}\n')

    pred_dir = os.path.abspath(os.path.join('outputs', 'predictions', video_stem))
    os.makedirs(pred_dir, exist_ok=True)
    print(f'Predictions will be saved to: {pred_dir}\n')

    frames_dir_abs = os.path.abspath(FRAMES_DIR)

    # Step 1: frame extraction
    frame_dir, num_frames, fps = extract_frames(video_path, frames_dir_abs)

    # Step 2: action spotting
    print(f'\n[Step 2/5] Running action spotting (model: {args.as_model_name})...')
    run_action_spotting(
        video_stem, frames_dir_abs, args.as_model_name, num_frames, fps,
        out_dir=pred_dir,
        force=args.force_action_spotting,
    )
    print(f'[Step 2/5] Action spotting done. Results in {pred_dir}.')

    # Step 3: tracking
    print(f'\n[Step 3/5] Running tracking...')
    run_tracking(
        video_path, pred_dir,
        force=args.force_tracking,
        no_players=args.no_players,
        no_ball=args.no_ball,
        no_court=args.no_court,
    )
    print(f'[Step 3/5] Tracking done. Results in {pred_dir}.')

    # Step 4: score computation
    run_score(
        pred_dir, video_stem, pred_dir,
        initial_sets=args.initial_sets,
        initial_games=args.initial_games,
        initial_points=args.initial_points,
        force=args.force_score,
    )

    # Step 5: visualization
    if not args.no_viz:
        run_visualization(
            pred_dir,
            video_stem,
            force=args.force_viz,
        )


    print(f'\n{"="*60}')
    print(f'Pipeline complete for: {args.video_name}')
    print(f'{"="*60}\n')