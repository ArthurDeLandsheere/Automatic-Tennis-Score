#!/usr/bin/env python
"""
Global evaluation script for tennis action spotting models.
Evaluates individual models and their ensemble across NMS windows.
"""

import os
import re
import subprocess
import itertools

# ── Configuration ────────────────────────────────────────────────────────────

CHECKPOINTS = [
    'checkpoints/tennis_rny002gsm_gru_rgb',
    'checkpoints/tennis_rny008gsm_gru_rgb',
]

SPLIT       = 'test'
DATASET     = 'tennis'
NMS_WINDOWS = [0, 1, 2, 3, 5]
TOLERANCES  = [0, 1, 2, 4]

# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70, flush=True)

def run(cmd):
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def find_score_file(ckpt, split):
    """Return the highest-epoch score file for a given split."""
    for pattern in [
        r'pred-{}\.(\d+)\.score\.json\.gz'.format(split),
        r'pred-{}\.(\d+)\.score\.json'.format(split),
    ]:
        regex = re.compile(pattern)
        candidates = [
            (os.path.join(ckpt, f), int(regex.match(f).group(1)))
            for f in os.listdir(ckpt) if regex.match(f)
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    return None

# ── Individual model evaluation ───────────────────────────────────────────────

for ckpt, nms in itertools.product(CHECKPOINTS, NMS_WINDOWS):
    model_name = ckpt.split('/')[-1]
    section(f'Model: {model_name} | NMS window: {nms}')
    run([
        'python', 'eval.py',
        ckpt,
        '-s', SPLIT,
        '--nms_window', str(nms),
        '-t', *map(str, TOLERANCES),
    ])
    

# ── Ensemble evaluation ───────────────────────────────────────────────────────

score_files = []
for ckpt in CHECKPOINTS:
    sf = find_score_file(ckpt, SPLIT)
    if sf is None:
        print(f'[WARN] No score file found for {ckpt} — skipping ensemble.')
        break
    score_files.append(sf)
else:
    for nms in NMS_WINDOWS:
        section(f'Ensemble (rny002 + rny008) | NMS window: {nms}')
        run([
            'python', 'eval_ensemble.py',
            DATASET,
            *score_files,
            '-s', SPLIT,
            '--nms_window', str(nms),
        ])

section('All evaluations complete.')