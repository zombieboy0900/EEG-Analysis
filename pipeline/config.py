"""
Shared constants for the ADHD EEG TDA pipeline.
Both pipeline_subject.py and pipeline_group.py import from here.
Changing a value here automatically invalidates caches (via hash in pipeline_subject.py).
"""

import numpy as np
import os

# ── Sampling / epoching ───────────────────────────────────────────────────────
SFREQ           = 128       # Hz
EPOCH_DURATION  = 2.0       # seconds
EPOCH_OVERLAP   = 0.0       # fraction (0 = no overlap)

# ── Channel selection ─────────────────────────────────────────────────────────
# 19 channels: full 10-20 set. Cz included — carries real signal (non-zero std
# confirmed in raw data) and is relevant to ADHD frontocentral midline activity.
TARGET_CHANNELS = [
    'C3', 'C4', 'Cz',
    'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
    'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz',
    'T7', 'T8',
]

# Standard 10-20 ordering as stored in the .mat files (index → name)
STANDARD_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1',  'O2',  'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz',  'Cz',  'Pz',
]

# ── Frequency bands ───────────────────────────────────────────────────────────
FREQ_BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# ── Pipeline feature flags ────────────────────────────────────────────────────
# Control which parts of the pipeline are active.
# Defaults reflect the current analysis scope (correlation-only, no spatial filter).

INTERPOLATE_BAD_CHANNELS  = False  # interpolate RANSAC bad channels before connectivity
COMPUTE_NO_SPATIAL_FILTER = True   # run pipeline on raw (no spatial filter) epochs
COMPUTE_SURFACE_LAPLACIAN = False  # run pipeline on CSD-transformed epochs
COMPUTE_COHERENCE         = False  # include coherence as a connectivity measure
COMPUTE_WPLI              = False  # include wPLI as a connectivity measure

# Active connectivity measures — derived from flags above, do not edit directly.
CONN_MEASURES = (
    (['coherence'] if COMPUTE_COHERENCE else []) +
    (['wpli']      if COMPUTE_WPLI      else []) +
    ['correlation']
)

# ── TDA ───────────────────────────────────────────────────────────────────────
N_FILT_STEPS = 100   # filtration resolution for Betti curves

# ── Graph theory ──────────────────────────────────────────────────────────────
GRAPH_DENSITY  = 0.25                           # proportional threshold for graph metrics
DENSITY_SWEEP  = list(np.arange(0.10, 0.51, 0.05))  # 9 points: 0.10, 0.15, …, 0.50

# ── ICA ───────────────────────────────────────────────────────────────────────
ICA_N_COMPONENTS = 15
ICA_METHOD       = 'infomax'
ICA_EXTENDED     = True
ICA_RANDOM_STATE = 97

# ICLabel automatic artifact rejection.
# Exclude a component only if ICLabel is >ICLABEL_THRESHOLD confident it's an artifact.
# Components whose label is in ICLABEL_KEEP are always retained.
ICLABEL_THRESHOLD = 0.80
ICLABEL_KEEP      = ('brain', 'other')

# ── AutoReject ────────────────────────────────────────────────────────────────
AUTOREJECT_N_INTERPOLATE = [1, 4, 32]
AUTOREJECT_RANDOM_STATE  = 42

# ── Surrogates (reference only — skipped in pipeline_subject.py) ──────────────
N_SURROGATES         = 200
SURROGATE_PERCENTILE = 95

# ── Quality control ───────────────────────────────────────────────────────────
# Subjects with fewer clean epochs after AutoReject are excluded.
# Set to 0 to disable the check.
MIN_CLEAN_EPOCHS = 30

# Max RANSAC bad channels before a subject is excluded.
# With 19-channel low-density EEG each electrode covers a whole brain region —
# stricter than the 20% rule used in high-density literature.
MAX_BAD_CHANNELS = 2

# ── Parallelism ───────────────────────────────────────────────────────────────
# i7-10750H: 6 physical cores / 12 threads. CPU <90% at n_jobs=9 → using all 12 threads.
# Override with --n-jobs flag in run_all.py.
N_JOBS = 12

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(_REPO_ROOT, 'results')
DATA_ROOT    = os.path.join(_REPO_ROOT, 'data')
METADATA_CSV = os.path.join(DATA_ROOT, 'metadata.csv')
RAW_DIR      = os.path.join(DATA_ROOT, 'raw')
