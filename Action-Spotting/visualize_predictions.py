#!/usr/bin/env python
# coding: utf-8

# # Prediction Visualizer
# Visualize action spotting predictions overlaid on video frames.

# In[11]:


import json
import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from collections import defaultdict


# ## Configuration — edit these paths

# In[12]:


VIDEO_PATH = "data/tennis/frames/wimbledon_2019_mens_semifinal_federer_nadal_1112_1716"
PRED_JSON     = "pred-single.40.json"                # path to predictions
DISPLAY_WINDOW_SEC = 0.5                           # seconds before/after event to show label
START_FRAME   = 0                                    # set to e.g. 1112 if clip starts at that frame


# ## Load predictions

# In[13]:


with open(PRED_JSON) as f:
    pred = json.load(f)[0]

fps    = pred['fps']
events = pred['events']

print(f"Video : {pred['video']}")
print(f"FPS   : {fps:.2f}")
print(f"Events: {len(events)}")


# ## Helper functions

# In[14]:


LABEL_COLORS = {
    'far_court_serve':   (255, 160,   0),
    'near_court_serve':  (255, 120,   0),
    'far_court_swing':   ( 50, 120, 255),
    'near_court_swing':  ( 50, 180, 255),
    'far_court_bounce':  ( 50, 200,  80),
    'near_court_bounce': ( 50, 150,  50),
}
DEFAULT_COLOR = (160, 160, 160)


def build_frame_label_map(events, fps, window_sec):
    """Map each frame number to the list of active (label, score, is_exact) tuples."""
    window_frames = int(window_sec * fps)
    frame_map = defaultdict(list)
    for event in events:
        label  = event['label']
        score  = event['score']
        center = event['frame']
        for f in range(center - window_frames, center + window_frames + 1):
            frame_map[f].append((label, score, center == f))
    return frame_map


def draw_overlay(frame_img, labels_at_frame, frame_num):
    img = frame_img.copy()

    # Frame counter
    cv2.putText(img, f"Frame: {frame_num}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    y_offset = 90
    for (label, score, is_exact) in labels_at_frame:
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        text  = f"{label}  {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        pad = 8
        x1, y1 = 15, y_offset - text_h - pad
        x2, y2 = 15 + text_w + pad * 2, y_offset + pad

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        border_thickness = 3 if is_exact else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, border_thickness)
        cv2.putText(img, text, (15 + pad, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        y_offset += text_h + pad * 2 + 10

    return img


def frame_to_jpeg_bytes(frame_bgr):
    """Convert a BGR OpenCV frame to JPEG bytes for display in the notebook."""
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


# Pre-load all frames into memory (fine for short clips)
def load_all_frames(frames_dir):
    import os
    frame_files = sorted([
        f for f in os.listdir(frames_dir) if f.endswith('.jpg')
    ])
    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
    print(f"Loaded {len(frames)} frames from {frames_dir}")
    return frames


frame_map = build_frame_label_map(events, fps, DISPLAY_WINDOW_SEC)
all_frames = load_all_frames(VIDEO_PATH)
total_frames = len(all_frames)


# ## Interactive player

# In[15]:


# --- Widgets ---
img_widget    = widgets.Image(format='jpeg', width='100%')
slider        = widgets.IntSlider(value=0, min=0, max=total_frames - 1,
                                  description='Frame', continuous_update=True,
                                  layout=widgets.Layout(width='100%'))
play_btn      = widgets.ToggleButton(value=False, description='▶ Play',
                                     button_style='success',
                                     layout=widgets.Layout(width='120px'))
prev_btn      = widgets.Button(description='◀ Prev', layout=widgets.Layout(width='100px'))
next_btn      = widgets.Button(description='Next ▶', layout=widgets.Layout(width='100px'))
event_dropdown = widgets.Dropdown(
    options=[('— jump to event —', -1)] +
            [(f"[{e['frame']}] {e['label']} ({e['score']:.2f})", e['frame']) for e in events],
    description='Jump to:',
    layout=widgets.Layout(width='400px')
)
info_label    = widgets.Label(value='')


def show_frame(frame_idx):
    abs_frame = frame_idx + START_FRAME
    raw = all_frames[frame_idx]
    labels = frame_map.get(abs_frame, [])
    annotated = draw_overlay(raw, labels, abs_frame)
    img_widget.value = frame_to_jpeg_bytes(annotated)
    active = list({l for l, _, _ in labels})
    info_label.value = f"Active labels: {', '.join(active)}" if active else "No active labels"


# Slider → update image
def on_slider_change(change):
    show_frame(change['new'])
slider.observe(on_slider_change, names='value')

# Prev / Next buttons
def on_prev(_):
    slider.value = max(0, slider.value - 1)
def on_next(_):
    slider.value = min(total_frames - 1, slider.value + 1)
prev_btn.on_click(on_prev)
next_btn.on_click(on_next)

# Jump-to-event dropdown
def on_event_select(change):
    target_frame = change['new']
    if target_frame >= 0:
        slider.value = max(0, min(total_frames - 1, target_frame - START_FRAME))
event_dropdown.observe(on_event_select, names='value')

# Play button (uses a Play widget internally for simplicity)
play = widgets.Play(value=0, min=0, max=total_frames - 1, step=1,
                    interval=int(1000 / fps)*3, description='Auto-play')
widgets.jslink((play, 'value'), (slider, 'value'))

# Layout
controls = widgets.HBox([play, prev_btn, next_btn, event_dropdown])
ui = widgets.VBox([img_widget, slider, controls, info_label])

show_frame(0)
display(ui)

# [MADE BY CLAUDE (Anthropic)]
