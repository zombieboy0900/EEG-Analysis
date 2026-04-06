"""
pipeline_subject.py — per-subject EEG TDA pipeline.

Process one subject end-to-end and write outputs to results/subjects/{subject_id}/.
Designed to be called from run_all.py via joblib.Parallel.

Usage:
    python pipeline/pipeline_subject.py data/raw/v10p.mat
    python pipeline/pipeline_subject.py data/raw/v10p.mat --force
    python pipeline/pipeline_subject.py data/raw/v10p.mat --figures
"""

import os
import sys
import json
import pickle
import hashlib
import inspect
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import scipy.io
import mne
from mne.io import RawArray
from mne.preprocessing import ICA
from autoreject import Ransac, AutoReject
from scipy.signal import coherence, csd, butter, filtfilt
from scipy.stats import kurtosis as sp_kurtosis
from statsmodels.tsa.stattools import kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings('ignore', category=InterpolationWarning)
from itertools import combinations
import networkx as nx
from ripser import ripser

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')

# Add pipeline/ to path so config.py is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS  (all scoped to a specific subject's cache_dir)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(cache_dir, name):
    return os.path.join(cache_dir, name)

def _cache_exists(cache_dir, *names):
    return all(os.path.exists(os.path.join(cache_dir, n)) for n in names)

def _invalidate(cache_dir, *names):
    for n in names:
        p = os.path.join(cache_dir, n)
        if os.path.exists(p):
            os.remove(p)

def _save_pkl(cache_dir, name, obj):
    """Atomic pickle write: write to .tmp then rename to prevent corruption."""
    os.makedirs(cache_dir, exist_ok=True)
    tmp = os.path.join(cache_dir, name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, os.path.join(cache_dir, name))

def _load_pkl(cache_dir, name):
    with open(os.path.join(cache_dir, name), 'rb') as f:
        return pickle.load(f)

def _save_npz(cache_dir, stem, d):
    """Atomic npz write."""
    os.makedirs(cache_dir, exist_ok=True)
    tmp = os.path.join(cache_dir, stem + '.tmp.npz')
    np.savez_compressed(tmp, **{'__'.join(k): v for k, v in d.items()})
    os.replace(tmp, os.path.join(cache_dir, stem + '.npz'))

def _load_npz(cache_dir, stem):
    data = np.load(os.path.join(cache_dir, stem + '.npz'))
    return {tuple(k.split('__')): data[k] for k in data.files}

def _make_cache_key(fn_list, cfg_dict):
    """Hash function source code + config values → cache key string."""
    src = ''.join(inspect.getsource(f) for f in fn_list)
    cfg = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256((src + cfg).encode()).hexdigest()

def _read_key(cache_dir, name):
    p = os.path.join(cache_dir, name + '.key')
    return open(p).read().strip() if os.path.exists(p) else None

def _write_key(cache_dir, name, key):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, name + '.key'), 'w') as f:
        f.write(key)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logger(subject_id, out_dir):
    """File logger (run.log) + stdout. Returns logger."""
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(subject_id)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(os.path.join(out_dir, 'run.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(f'[{subject_id}] %(message)s'))
    logger.addHandler(sh)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(mat_path, subject_id, logger):
    """Load .mat, clean, epoch, return clean Epochs object."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find {mat_path}")

    mat  = scipy.io.loadmat(mat_path)

    # The variable name inside the .mat file matches the subject_id (e.g. 'v10p').
    if subject_id not in mat:
        # Fallback: use the first non-metadata key
        keys = [k for k in mat if not k.startswith('_')]
        if not keys:
            raise KeyError(f"No data variable found in {mat_path}. Keys: {list(mat.keys())}")
        key = keys[0]
        logger.warning(f"Key '{subject_id}' not in mat file — using '{key}' instead")
    else:
        key = subject_id

    data = mat[key]
    if data.shape[0] > data.shape[1]:
        data = data.T   # ensure (n_channels, n_times)

    # ── Channel setup ─────────────────────────────────────────────────────────
    ch_names_generic = [f'EEG{i:03d}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names_generic, sfreq=config.SFREQ, ch_types='eeg')
    # Data is in µV; MNE expects volts internally.
    raw = RawArray(data * 1e-6, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    if len(raw.ch_names) == len(config.STANDARD_19):
        mapping = {raw.ch_names[i]: config.STANDARD_19[i]
                   for i in range(len(config.STANDARD_19))}
        raw.rename_channels(mapping)
        raw.set_montage(montage)
    else:
        logger.warning(f"Expected {len(config.STANDARD_19)} channels, got {len(raw.ch_names)}. Skipping rename.")
        raw.set_montage(montage, on_missing='ignore')

    # ── Filtering ─────────────────────────────────────────────────────────────
    raw.notch_filter(freqs=60.0)
    raw.filter(l_freq=1.0, h_freq=45.0, phase='zero', fir_design='firwin')

    # ── RANSAC bad channel detection ──────────────────────────────────────────
    epochs_temp = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    rsc = Ransac(n_jobs=1, verbose=False)
    rsc.fit(epochs_temp)
    raw.info['bads'] = rsc.bad_chs_
    logger.info(f"RANSAC bad channels: {rsc.bad_chs_}")
    del epochs_temp

    # Interpolate bad channels before ICA so the fit uses a complete channel set.
    # rsc.bad_chs_ is still returned for diagnostics — info['bads'] is cleared here
    # so both no-filter and surface laplacian pipelines downstream see the same base data.
    if rsc.bad_chs_:
        raw.interpolate_bads(reset_bads=True)
        logger.info(f"Interpolated {len(rsc.bad_chs_)} bad channels before ICA")

    # Average reference applied after bad channel interpolation so the mean
    # is not contaminated by any broken electrodes.
    raw.set_eeg_reference('average', projection=False)
    logger.info("Re-referenced to average (after bad channel interpolation)")

    # ── ICA with ICLabel ──────────────────────────────────────────────────────
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = ICA(n_components=config.ICA_N_COMPONENTS,
              method=config.ICA_METHOD,
              fit_params=dict(extended=config.ICA_EXTENDED),
              random_state=config.ICA_RANDOM_STATE)
    ica.fit(raw_for_ica)

    _ICLABEL_CLASS_NAMES = [
        'brain', 'muscle artifact', 'eye blink', 'heart beat',
        'line noise', 'channel noise', 'other',
    ]

    # Import guard is separate so runtime errors inside label_components
    # are not silently mistaken for a missing package.
    # We call iclabel_label_components directly to get the full (n_components, 7)
    # probability matrix; label_components (public API) only returns the winning prob.
    try:
        from mne_icalabel.iclabel import iclabel_label_components as _iclabel_fn
        _icalabel_available = True
    except ImportError:
        _icalabel_available = False

    if not _icalabel_available:
        ica.exclude = [0, 1]
        logger.warning("mne-icalabel not installed — using hardcoded ica.exclude=[0,1]. "
                       "Run: pip install mne-icalabel")
        ica_report = None
    else:
        try:
            # Returns (n_components, 7) — one probability per ICLabel class.
            # Force onnx backend: torch has multiprocessing issues on Windows
            # with 12 loky workers; onnxruntime is more reliable.
            probas = np.array(_iclabel_fn(raw_for_ica, ica, backend='onnx'))  # (n_components, 7)
            labels = [_ICLABEL_CLASS_NAMES[int(np.argmax(p))] for p in probas]

            ica.exclude = [
                i
                for i, (label, proba) in enumerate(zip(labels, probas))
                if label not in config.ICLABEL_KEEP
                and float(proba.max()) > config.ICLABEL_THRESHOLD
            ]

            # Per-component log: every IC with label, max prob, full class breakdown
            logger.info(f"ICLabel — {len(labels)} components "
                        f"(threshold={config.ICLABEL_THRESHOLD}, keep={config.ICLABEL_KEEP}):")
            for i, (label, proba) in enumerate(zip(labels, probas)):
                tag      = " [EXCLUDED]" if i in ica.exclude else ""
                prob_str = "  ".join(
                    f"{cn.split()[0][:3]}={float(p):.2f}"
                    for cn, p in zip(_ICLABEL_CLASS_NAMES, proba)
                )
                logger.info(f"  IC{i:02d}: {label:<18s} max={float(proba.max()):.3f}  "
                            f"{prob_str}{tag}")

            logger.info(f"ICLabel excluded {len(ica.exclude)} components: {ica.exclude}")

            ica_report = {
                'labels':          labels,
                'probas':          probas,
                'class_names':     _ICLABEL_CLASS_NAMES,
                'excluded':        list(ica.exclude),
                'excluded_labels': [labels[i] for i in ica.exclude],
                'excluded_probas': [float(probas[i].max()) for i in ica.exclude],
                'n_components':    len(labels),
            }

        except Exception as e:
            ica.exclude = []
            logger.warning(f"ICLabel failed ({type(e).__name__}: {e}) — "
                           "no components excluded.")
            ica_report = None

    ica.apply(raw)
    del raw_for_ica

    # ── Channel selection ─────────────────────────────────────────────────────
    available = [ch for ch in config.TARGET_CHANNELS if ch in raw.ch_names]
    raw.pick(available)

    # ── Epoching + AutoReject ─────────────────────────────────────────────────
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=config.EPOCH_DURATION,
        overlap=config.EPOCH_OVERLAP,
        preload=True,
    )
    ar = AutoReject(
        n_interpolate=config.AUTOREJECT_N_INTERPOLATE,
        random_state=config.AUTOREJECT_RANDOM_STATE,
        n_jobs=1,
        verbose=False,
    )
    epochs_clean = ar.fit_transform(epochs)
    logger.info(f"Epochs: {len(epochs)} -> {len(epochs_clean)} after AutoReject "
                f"({len(epochs) - len(epochs_clean)} dropped)")

    # ── Minimum epochs check ──────────────────────────────────────────────────
    if config.MIN_CLEAN_EPOCHS > 0 and len(epochs_clean) < config.MIN_CLEAN_EPOCHS:
        raise ValueError(
            f"Only {len(epochs_clean)} clean epochs remain "
            f"(threshold: {config.MIN_CLEAN_EPOCHS}). Subject excluded."
        )

    # ── KPSS stationarity (diagnostic only — no epochs dropped) ──────────────
    _log_stationarity(epochs_clean, logger)

    return epochs_clean, rsc.bad_chs_, ica.exclude, ica_report


def _log_stationarity(epochs_clean, logger, significance=0.05):
    """Run KPSS per channel/epoch and log summary. Does not drop anything."""
    data = epochs_clean.get_data()
    n_epochs, n_ch, _ = data.shape
    fail_mask = np.zeros((n_epochs, n_ch), dtype=bool)

    for ep in range(n_epochs):
        for ch in range(n_ch):
            try:
                _, p_value, _, _ = kpss(data[ep, ch], regression='c', nlags='auto')
                if p_value < significance:
                    fail_mask[ep, ch] = True
            except Exception:
                pass

    total_tests = n_epochs * n_ch
    total_fail  = int(fail_mask.sum())
    fail_pct    = 100 * total_fail / total_tests if total_tests > 0 else 0.0
    logger.info(f"KPSS stationarity: {total_fail}/{total_tests} tests failed "
                f"({fail_pct:.1f}%) — diagnostic only, no epochs dropped")


# ══════════════════════════════════════════════════════════════════════════════
# SURFACE LAPLACIAN (CSD)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_csd(epochs_clean, logger):
    """Apply Current Source Density transform. Returns CSD epochs or None."""
    try:
        from mne.preprocessing import compute_current_source_density
        epochs_for_csd = epochs_clean.copy()
        if epochs_for_csd.info['bads']:
            epochs_for_csd.interpolate_bads()
        epochs_for_csd.info['bads'] = []
        epochs_csd = compute_current_source_density(epochs_for_csd)
        logger.info("CSD transform applied.")
        return epochs_csd
    except Exception as e:
        logger.warning(f"CSD transform failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════════════

def _compute_coherence(epochs_data, sfreq, fmin, fmax):
    """Return per-epoch coherence matrices stacked as (n_epochs, n_ch, n_ch)."""
    n_epochs, n_ch, n_times = epochs_data.shape
    nperseg = min(128, n_times)
    all_conn = np.zeros((n_epochs, n_ch, n_ch))
    for ep in range(n_epochs):
        conn = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                f, cxy = coherence(epochs_data[ep, i], epochs_data[ep, j],
                                   fs=sfreq, nperseg=nperseg)
                mask = (f >= fmin) & (f <= fmax)
                if mask.any():
                    conn[i, j] = conn[j, i] = float(np.mean(cxy[mask]))
        np.fill_diagonal(conn, 1.0)
        all_conn[ep] = conn
    return all_conn


def _compute_wpli(epochs_data, sfreq, fmin, fmax):
    """Return per-epoch wPLI matrices stacked as (n_epochs, n_ch, n_ch).
    Each epoch is treated as an independent estimate (single-epoch wPLI)."""
    n_epochs, n_ch, n_times = epochs_data.shape
    nperseg = min(128, n_times)
    all_conn = np.zeros((n_epochs, n_ch, n_ch))
    for ep in range(n_epochs):
        conn = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                f, sxy = csd(epochs_data[ep, i], epochs_data[ep, j],
                             fs=sfreq, nperseg=nperseg)
                mask = (f >= fmin) & (f <= fmax)
                if not mask.any():
                    continue
                im = np.imag(sxy[mask])
                imag_sum     = float(np.sum(im))
                imag_abs_sum = float(np.sum(np.abs(im)))
                imag_sq_sum  = float(np.sum(im ** 2))
                if imag_abs_sum > 0:
                    num = imag_sum ** 2 - imag_sq_sum
                    den = imag_abs_sum ** 2 - imag_sq_sum
                    if den > 0:
                        conn[i, j] = conn[j, i] = abs(num / den)
        np.fill_diagonal(conn, 1.0)
        all_conn[ep] = conn
    return all_conn


def _compute_correlation(epochs_data, sfreq, fmin, fmax):
    """Return per-epoch correlation matrices stacked as (n_epochs, n_ch, n_ch)."""
    n_epochs, n_ch, n_times = epochs_data.shape
    nyq  = sfreq / 2.0
    low  = fmin / nyq
    high = min(fmax / nyq, 0.99)
    b, a = butter(4, [low, high], btype='band')
    all_conn = np.zeros((n_epochs, n_ch, n_ch))
    for ep in range(n_epochs):
        filtered = np.array([filtfilt(b, a, epochs_data[ep, ch]) for ch in range(n_ch)])
        conn = np.abs(np.corrcoef(filtered))
        np.fill_diagonal(conn, 1.0)
        all_conn[ep] = conn
    return all_conn


def _compute_all_connectivity(epochs_clean, logger):
    """
    Compute per-epoch connectivity matrices for all active measures and bands.

    Returns
    -------
    conn_per_epoch : dict {(measure, band): ndarray (n_epochs, n_ch, n_ch)}
    dist_per_epoch : dict {(measure, band): ndarray (n_epochs, n_ch, n_ch)}
    ch_names       : list of str
    """
    epochs_data = epochs_clean.get_data()
    sfreq       = epochs_clean.info['sfreq']
    ch_names    = list(epochs_clean.ch_names)
    n_epochs, n_ch, _ = epochs_data.shape

    fns = {}
    if config.COMPUTE_COHERENCE:
        fns['coherence'] = _compute_coherence
    if config.COMPUTE_WPLI:
        fns['wpli'] = _compute_wpli
    fns['correlation'] = _compute_correlation

    conn_per_epoch = {}
    dist_per_epoch = {}
    total = len(fns) * len(config.FREQ_BANDS)
    count = 0

    for m_name, m_func in fns.items():
        for b_name, (fmin, fmax) in config.FREQ_BANDS.items():
            count += 1
            logger.debug(f"Connectivity [{count}/{total}] {m_name}/{b_name}...")
            try:
                # all_conn shape: (n_epochs, n_ch, n_ch)
                all_conn = m_func(epochs_data, sfreq, fmin, fmax)
                all_conn = np.clip(np.nan_to_num(all_conn, nan=0.0), 0, 1)
                all_dist = np.clip(1.0 - all_conn, 0, 1)
                for ep in range(n_epochs):
                    np.fill_diagonal(all_dist[ep], 0.0)
                conn_per_epoch[(m_name, b_name)] = all_conn
                dist_per_epoch[(m_name, b_name)] = all_dist
            except Exception as e:
                logger.warning(f"Connectivity {m_name}/{b_name} FAILED: {e}")
                dummy = np.zeros((n_epochs, n_ch, n_ch))
                conn_per_epoch[(m_name, b_name)] = dummy
                dist_per_epoch[(m_name, b_name)] = dummy

    return conn_per_epoch, dist_per_epoch, ch_names


# ══════════════════════════════════════════════════════════════════════════════
# TDA
# ══════════════════════════════════════════════════════════════════════════════

def _compute_persistence(dist_matrix, ch_names):
    n = len(ch_names)
    if np.all(dist_matrix == 0) or np.all(dist_matrix == 1):
        return [np.array([]), np.array([])], {0: [], 1: []}

    try:
        result      = ripser(dist_matrix, maxdim=1, distance_matrix=True, do_cocycles=True)
        raw_cocycles = result.get('cocycles', [[], []])
    except TypeError:
        result      = ripser(dist_matrix, maxdim=1, distance_matrix=True)
        raw_cocycles = [[], []]

    dgms       = result['dgms']
    generators = {0: [], 1: []}

    # H0 via MST
    try:
        G_full = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G_full.add_edge(i, j, weight=float(dist_matrix[i, j]))
        mst       = nx.minimum_spanning_tree(G_full, weight='weight')
        mst_edges = sorted(
            [(d['weight'], u, v) for u, v, d in mst.edges(data=True)],
            key=lambda e: e[0],
        )
        for birth, death in (dgms[0] if dgms else []):
            if not np.isfinite(death):
                generators[0].append([])
            else:
                best = min(mst_edges, key=lambda e: abs(e[0] - death))
                generators[0].append([ch_names[best[1]], ch_names[best[2]]])
    except Exception:
        generators[0] = [[] for _ in range(len(dgms[0]) if dgms else 0)]

    # H1 via cocycles
    try:
        h1_dgm      = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)
        cocycles_h1 = raw_cocycles[1] if len(raw_cocycles) > 1 else []
        for k in range(len(h1_dgm)):
            if k < len(cocycles_h1):
                cocycle  = np.asarray(cocycles_h1[k])
                involved = set()
                for edge in cocycle:
                    u, v = int(edge[0]), int(edge[1])
                    if u < n: involved.add(ch_names[u])
                    if v < n: involved.add(ch_names[v])
                generators[1].append(sorted(involved))
            else:
                generators[1].append([])
    except Exception:
        generators[1] = [[] for _ in range(len(dgms[1]) if len(dgms) > 1 else 0)]

    return dgms, generators


def _dist_to_rho_mapper(dist_matrix):
    """
    Build a function that converts distance values → edge density ρ.

    ρ = k / C(N,2), where k = number of edges with distance ≤ d.
    Edges are added in order of increasing distance (= decreasing correlation),
    so ρ = 0 means no edges, ρ = 1 means fully connected.
    """
    n = dist_matrix.shape[0]
    n_edges = n * (n - 1) // 2
    upper = dist_matrix[np.triu_indices(n, k=1)]
    sorted_dists = np.sort(upper)

    def mapper(d):
        return float(np.searchsorted(sorted_dists, d, side='right')) / n_edges

    return mapper


def _compute_betti_curves(diagrams, rho_mapper):
    """
    Compute Betti curves with edge density ρ as the filtration parameter.

    Parameters
    ----------
    diagrams   : list of ndarray — ripser persistence diagrams
    rho_mapper : callable — maps distance value → ρ ∈ [0, 1]
    """
    rho_values = np.linspace(0, 1.0, config.N_FILT_STEPS)
    betti = {}
    for dim, dgm in enumerate(diagrams):
        counts = np.zeros(config.N_FILT_STEPS)
        if dgm.size > 0:
            for birth_eps, death_eps in dgm:
                birth_rho = rho_mapper(birth_eps)
                death_rho = 1.0 if not np.isfinite(death_eps) else rho_mapper(death_eps)
                counts[(rho_values >= birth_rho) & (rho_values < death_rho)] += 1
        betti[dim] = counts
    return rho_values, betti


def _compute_landscape(dgm_rho, rho_values, n_levels=5):
    """
    Compute the persistent landscape for one persistence diagram in ρ space.

    For each (birth_ρ, death_ρ) pair, build a tent function:
        tent_i(t) = max(0, min(t − birth_ρ, death_ρ − t))
    The k-th landscape λ_k(t) is the k-th largest tent value at each t.

    Parameters
    ----------
    dgm_rho   : ndarray (n_pairs, 2) — birth/death already in ρ units
    rho_values: ndarray (N_FILT_STEPS,) — evaluation grid
    n_levels  : int — number of landscape levels to keep (λ_1 … λ_n)

    Returns
    -------
    landscape : ndarray (n_levels, N_FILT_STEPS)
    """
    n_steps   = len(rho_values)
    landscape = np.zeros((n_levels, n_steps))

    if dgm_rho.size == 0:
        return landscape

    # Build tent values: shape (n_pairs, n_steps)
    births = dgm_rho[:, 0][:, None]   # (n_pairs, 1)
    deaths = dgm_rho[:, 1][:, None]
    t      = rho_values[None, :]       # (1, n_steps)
    tents  = np.maximum(0, np.minimum(t - births, deaths - t))  # (n_pairs, n_steps)

    # At each ρ step, sort tent heights descending and take top n_levels
    for s in range(n_steps):
        col = np.sort(tents[:, s])[::-1]
        take = min(n_levels, len(col))
        landscape[:take, s] = col[:take]

    return landscape


def _compute_betti_features(eps_values, betti_curve):
    if np.all(betti_curve == 0) or len(betti_curve) == 0:
        return {'auc': 0.0, 'slope': 0.0, 'kurtosis': 0.0}
    auc = float(np.trapz(betti_curve, eps_values))
    try:
        kurt = float(sp_kurtosis(betti_curve, fisher=True))
        if np.isnan(kurt):
            kurt = 0.0
    except Exception:
        kurt = 0.0
    if np.unique(betti_curve).size > 1:
        try:
            slope = float(np.polyfit(eps_values, betti_curve, 1)[0])
        except np.linalg.LinAlgError:
            slope = 0.0
    else:
        slope = 0.0
    return {'auc': auc, 'slope': slope, 'kurtosis': kurt}


def _run_tda_pipeline(dist_per_epoch, ch_names):
    """
    Run TDA on per-epoch distance matrices.

    For each (measure, band):
      - Run ripser on each epoch's distance matrix
      - Convert filtration parameter from distance ε → edge density ρ
      - Compute Betti curves per epoch in ρ space
      - Average Betti curves across epochs

    Parameters
    ----------
    dist_per_epoch : dict {(measure, band): ndarray (n_epochs, n_ch, n_ch)}
    ch_names       : list of str
    """
    rho_values  = np.linspace(0, 1.0, config.N_FILT_STEPS)
    tda_results = {}

    n_levels = config.N_LANDSCAPE_LEVELS

    for key, dist_stack in dist_per_epoch.items():
        n_epochs      = dist_stack.shape[0]
        betti_sum     = {0: np.zeros(config.N_FILT_STEPS),
                         1: np.zeros(config.N_FILT_STEPS)}
        landscape_sum = {0: np.zeros((n_levels, config.N_FILT_STEPS)),
                         1: np.zeros((n_levels, config.N_FILT_STEPS))}
        valid_epochs  = 0

        for ep in range(n_epochs):
            dist_matrix = dist_stack[ep]
            if np.all(dist_matrix == 0) or np.all(dist_matrix == 1):
                continue

            rho_mapper = _dist_to_rho_mapper(dist_matrix)
            dgms, _    = _compute_persistence(dist_matrix, ch_names)
            _, betti   = _compute_betti_curves(dgms, rho_mapper)

            for dim in [0, 1]:
                if dim in betti:
                    betti_sum[dim] += betti[dim]

                # Convert this epoch's diagram to ρ space then compute landscape
                raw_dgm = dgms[dim] if dim < len(dgms) else np.array([]).reshape(0, 2)
                if raw_dgm.size > 0:
                    dgm_rho = np.column_stack([
                        [rho_mapper(b) for b in raw_dgm[:, 0]],
                        [1.0 if not np.isfinite(d) else rho_mapper(d)
                         for d in raw_dgm[:, 1]],
                    ])
                else:
                    dgm_rho = np.zeros((0, 2))
                landscape_sum[dim] += _compute_landscape(dgm_rho, rho_values, n_levels)

            valid_epochs += 1

        if valid_epochs == 0:
            mean_betti     = {0: np.zeros(config.N_FILT_STEPS),
                              1: np.zeros(config.N_FILT_STEPS)}
            mean_landscape = {0: np.zeros((n_levels, config.N_FILT_STEPS)),
                              1: np.zeros((n_levels, config.N_FILT_STEPS))}
        else:
            mean_betti     = {dim: betti_sum[dim]     / valid_epochs for dim in [0, 1]}
            mean_landscape = {dim: landscape_sum[dim] / valid_epochs for dim in [0, 1]}

        features = {
            f'B{dim}': _compute_betti_features(rho_values, mean_betti[dim])
            for dim in [0, 1]
        }
        # Add landscape AUC (λ_1 only — dominant landscape level) to features
        for dim in [0, 1]:
            lam1_auc = float(np.trapz(mean_landscape[dim][0], rho_values))
            features[f'B{dim}']['landscape_auc'] = lam1_auc

        tda_results[key] = {
            'rho_values':  rho_values,
            'betti':       mean_betti,
            'landscape':   mean_landscape,
            'n_epochs':    valid_epochs,
            'features':    features,
        }

    return tda_results


def _extract_betti_features(tda_results):
    """Collapse tda_results into a compact per-subject feature dict for group analysis."""
    subject_features = {}
    for key, r in tda_results.items():
        feats    = r['features']
        b0_feats = feats.get('B0', {'auc': 0.0, 'slope': 0.0, 'kurtosis': 0.0})
        b1_feats = feats.get('B1', {'auc': 0.0, 'slope': 0.0, 'kurtosis': 0.0})
        subject_features[key] = {
            # Betti curve scalar features
            'auc_b0':           b0_feats['auc'],
            'slope_b0':         b0_feats['slope'],
            'kurtosis_b0':      b0_feats['kurtosis'],
            'auc_b1':           b1_feats['auc'],
            'slope_b1':         b1_feats['slope'],
            'kurtosis_b1':      b1_feats['kurtosis'],
            # Landscape AUC (λ_1, the dominant level)
            'landscape_auc_b0': b0_feats.get('landscape_auc', 0.0),
            'landscape_auc_b1': b1_feats.get('landscape_auc', 0.0),
            # Full curves (for visualisation)
            'betti_0':          np.array(r['betti'].get(0, [])),
            'betti_1':          np.array(r['betti'].get(1, [])),
            'landscape_0':      r.get('landscape', {}).get(0, np.array([])),
            'landscape_1':      r.get('landscape', {}).get(1, np.array([])),
            'rho':              np.array(r['rho_values']),
            'n_epochs':         r.get('n_epochs', 0),
        }
    return subject_features


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH THEORY
# ══════════════════════════════════════════════════════════════════════════════

def _compute_graph_metrics(conn_matrix, ch_names, density=None):
    if density is None:
        density = config.GRAPH_DENSITY
    n        = len(ch_names)
    off_diag = conn_matrix[np.triu_indices(n, k=1)]
    threshold = np.percentile(off_diag, 100 * (1 - density))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if conn_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=conn_matrix[i, j])

    strength    = dict(G.degree(weight='weight'))
    clustering  = nx.clustering(G, weight='weight')

    G_dist = nx.Graph()
    G_dist.add_nodes_from(range(n))
    for i, j, d in G.edges(data=True):
        if d['weight'] > 0:
            G_dist.add_edge(i, j, weight=1.0 / d['weight'])

    if nx.is_connected(G_dist):
        cpl = nx.average_shortest_path_length(G_dist, weight='weight')
    else:
        cpls = [
            nx.average_shortest_path_length(G_dist.subgraph(c), weight='weight')
            for c in nx.connected_components(G_dist)
            if len(c) > 1
        ]
        cpl = float(np.mean(cpls)) if cpls else 0.0

    betweenness = nx.betweenness_centrality(G_dist, weight='weight')
    try:
        efficiency = nx.global_efficiency(G_dist, weight='weight')
    except TypeError:
        efficiency = nx.global_efficiency(G_dist)
    try:
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        modularity  = nx.community.modularity(G, communities, weight='weight')
    except Exception:
        modularity = 0.0

    return {
        'strength':    float(np.mean(list(strength.values()))),
        'clustering':  float(np.mean(list(clustering.values()))),
        'cpl':         float(cpl),
        'betweenness': float(np.mean(list(betweenness.values()))),
        'efficiency':  float(efficiency),
        'modularity':  float(modularity),
    }


def _run_graph_pipeline(conn_matrices, ch_names):
    return {key: _compute_graph_metrics(conn, ch_names)
            for key, conn in conn_matrices.items()}


def _run_density_sweep(conn_matrices, ch_names):
    densities = np.array(config.DENSITY_SWEEP)
    metrics   = ['strength', 'clustering', 'cpl', 'betweenness', 'efficiency', 'modularity']
    sweep     = {}
    for key, conn in conn_matrices.items():
        sweep[key] = {m: np.zeros(len(densities)) for m in metrics}
        for d_idx, d in enumerate(densities):
            result = _compute_graph_metrics(conn, ch_names, density=d)
            for m in metrics:
                sweep[key][m][d_idx] = result[m]
    return sweep, densities


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_subject(mat_path: str, force: bool = False, figures: bool = False):
    """
    Process one subject end-to-end.

    Parameters
    ----------
    mat_path : str
        Path to the subject's .mat file (e.g. 'data/raw/v10p.mat').
    force : bool
        If True, ignore all caches and recompute from scratch.
    figures : bool
        If True, generate per-subject visualizations (slow; skip for batch runs).

    Returns
    -------
    dict | None
        betti_features dict on success, None on failure.
    """
    mat_path   = str(mat_path)
    subject_id = Path(mat_path).stem
    out_dir    = os.path.join(config.RESULTS_ROOT, 'subjects', subject_id)
    cache_dir  = os.path.join(out_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)

    logger = _setup_logger(subject_id, out_dir)
    logger.info(f"=== Starting {subject_id} (force={force}, figures={figures}) ===")

    try:
        # ── Cache key computation ─────────────────────────────────────────────
        preproc_key = _make_cache_key(
            [_preprocess, _apply_csd],
            {
                'SFREQ': config.SFREQ,
                'EPOCH_DURATION': config.EPOCH_DURATION,
                'EPOCH_OVERLAP': config.EPOCH_OVERLAP,
                'TARGET_CHANNELS': config.TARGET_CHANNELS,
                'STANDARD_19': config.STANDARD_19,
                'ICA_N_COMPONENTS': config.ICA_N_COMPONENTS,
                'ICA_METHOD': config.ICA_METHOD,
                'ICA_EXTENDED': config.ICA_EXTENDED,
                'ICA_RANDOM_STATE': config.ICA_RANDOM_STATE,
                'ICLABEL_THRESHOLD': config.ICLABEL_THRESHOLD,
                'ICLABEL_KEEP': list(config.ICLABEL_KEEP),
                'AUTOREJECT_N_INTERPOLATE': config.AUTOREJECT_N_INTERPOLATE,
                'AUTOREJECT_RANDOM_STATE': config.AUTOREJECT_RANDOM_STATE,
                'MIN_CLEAN_EPOCHS': config.MIN_CLEAN_EPOCHS,
            },
        )
        conn_key = _make_cache_key(
            [_compute_all_connectivity, _compute_coherence, _compute_wpli, _compute_correlation],
            {'FREQ_BANDS': config.FREQ_BANDS, 'CONN_MEASURES': config.CONN_MEASURES},
        )
        tda_key = _make_cache_key(
            [_run_tda_pipeline, _compute_persistence, _compute_betti_curves,
             _compute_betti_features, _extract_betti_features],
            {'N_FILT_STEPS': config.N_FILT_STEPS},
        )
        graph_key = _make_cache_key(
            [_run_graph_pipeline, _compute_graph_metrics, _run_density_sweep],
            {'GRAPH_DENSITY': config.GRAPH_DENSITY, 'DENSITY_SWEEP': config.DENSITY_SWEEP},
        )

        # ── Cascade invalidation ──────────────────────────────────────────────
        preproc_changed = force or (_read_key(cache_dir, 'preprocess') != preproc_key)
        conn_changed    = preproc_changed or (_read_key(cache_dir, 'connectivity') != conn_key)
        tda_changed     = conn_changed or (_read_key(cache_dir, 'tda') != tda_key)
        graph_changed   = conn_changed or (_read_key(cache_dir, 'graph') != graph_key)

        if preproc_changed:
            logger.info("Preprocessing changed — invalidating all caches")
            _invalidate(cache_dir,
                        'epochs_clean.fif', 'ch_names.pkl',
                        'conn_matrices.npz', 'dist_matrices.npz',
                        'conn_csd.npz', 'dist_csd.npz',
                        'tda_results.pkl', 'tda_csd.pkl',
                        'graph_results.pkl', 'graph_results_csd.pkl',
                        'density_sweep.pkl', 'density_sweep_csd.pkl',
                        'betti_features.pkl')
        elif conn_changed:
            logger.info("Connectivity changed — invalidating connectivity and downstream")
            _invalidate(cache_dir,
                        'conn_matrices.npz', 'dist_matrices.npz',
                        'conn_csd.npz', 'dist_csd.npz',
                        'tda_results.pkl', 'tda_csd.pkl',
                        'graph_results.pkl', 'graph_results_csd.pkl',
                        'density_sweep.pkl', 'density_sweep_csd.pkl',
                        'betti_features.pkl')
        elif tda_changed or graph_changed:
            logger.info("TDA/Graph changed — invalidating TDA and graph caches")
            _invalidate(cache_dir,
                        'tda_results.pkl', 'tda_csd.pkl',
                        'graph_results.pkl', 'graph_results_csd.pkl',
                        'density_sweep.pkl', 'density_sweep_csd.pkl',
                        'betti_features.pkl')

        # ── Step 1: Preprocessing ─────────────────────────────────────────────
        if _cache_exists(cache_dir, 'epochs_clean.fif', 'ch_names.pkl'):
            logger.info("[1/5] Preprocessing — loading from cache")
            epochs_clean = mne.read_epochs(
                _cache_path(cache_dir, 'epochs_clean.fif'), preload=True, verbose=False)
            ch_names = _load_pkl(cache_dir, 'ch_names.pkl')
        else:
            logger.info("[1/5] Preprocessing...")
            epochs_clean, bad_chs, ica_excluded, ica_report = _preprocess(
                mat_path, subject_id, logger)
            ch_names = list(epochs_clean.ch_names)
            epochs_clean.save(_cache_path(cache_dir, 'epochs_clean.fif'),
                              overwrite=True, verbose=False)
            _save_pkl(cache_dir, 'ch_names.pkl', ch_names)
            _save_pkl(cache_dir, 'bad_channels.pkl', bad_chs)
            if ica_report is not None:
                _save_pkl(cache_dir, 'ica_report.pkl', ica_report)
                _save_pkl(out_dir,   'ica_report.pkl', ica_report)
            _write_key(cache_dir, 'preprocess', preproc_key)
            logger.info(f"Saved epochs_clean.fif  ({len(epochs_clean)} epochs, "
                        f"{len(ch_names)} channels)")

        # ── Step 2: Connectivity ──────────────────────────────────────────────
        n_measures = len(config.CONN_MEASURES) * len(config.FREQ_BANDS)
        if _cache_exists(cache_dir, 'conn_matrices.npz', 'dist_matrices.npz'):
            logger.info("[2/5] Connectivity — loading from cache")
            conn_per_epoch = _load_npz(cache_dir, 'conn_matrices')
            dist_per_epoch = _load_npz(cache_dir, 'dist_matrices')
        else:
            logger.info(f"[2/5] Computing connectivity ({n_measures} measure/band combinations)...")
            conn_per_epoch, dist_per_epoch, ch_names = _compute_all_connectivity(epochs_clean, logger)
            _save_npz(cache_dir, 'conn_matrices', conn_per_epoch)
            _save_npz(cache_dir, 'dist_matrices', dist_per_epoch)
            _write_key(cache_dir, 'connectivity', conn_key)

        # Mean matrices (epoch-average) used for graph theory
        conn_matrices = {k: v.mean(axis=0) for k, v in conn_per_epoch.items()}

        # Surface Laplacian connectivity
        conn_sl = dist_sl = dist_per_epoch_sl = None
        if config.COMPUTE_SURFACE_LAPLACIAN:
            if _cache_exists(cache_dir, 'conn_csd.npz', 'dist_csd.npz'):
                logger.debug("  Surface Laplacian connectivity — loading from cache")
                conn_per_epoch_sl = _load_npz(cache_dir, 'conn_csd')
                dist_per_epoch_sl = _load_npz(cache_dir, 'dist_csd')
            else:
                epochs_sl = _apply_csd(epochs_clean, logger)
                if epochs_sl is not None:
                    logger.info("  Computing Surface Laplacian connectivity...")
                    conn_per_epoch_sl, dist_per_epoch_sl, _ = _compute_all_connectivity(epochs_sl, logger)
                    _save_npz(cache_dir, 'conn_csd', conn_per_epoch_sl)
                    _save_npz(cache_dir, 'dist_csd', dist_per_epoch_sl)
            if dist_per_epoch_sl is not None:
                conn_sl = {k: v.mean(axis=0) for k, v in conn_per_epoch_sl.items()}
                dist_sl = {k: v.mean(axis=0) for k, v in dist_per_epoch_sl.items()}

        # ── Step 3: TDA ───────────────────────────────────────────────────────
        if _cache_exists(cache_dir, 'tda_results.pkl'):
            logger.info("[3/5] TDA — loading from cache")
            tda_results = _load_pkl(cache_dir, 'tda_results.pkl')
        else:
            logger.info("[3/5] Running TDA pipeline (per-epoch, then averaging Betti curves)...")
            tda_results = _run_tda_pipeline(dist_per_epoch, ch_names)
            _save_pkl(cache_dir, 'tda_results.pkl', tda_results)
            _write_key(cache_dir, 'tda', tda_key)

        # TDA on Surface Laplacian
        tda_sl = None
        if config.COMPUTE_SURFACE_LAPLACIAN and dist_per_epoch_sl is not None:
            if _cache_exists(cache_dir, 'tda_csd.pkl'):
                logger.debug("  Surface Laplacian TDA — loading from cache")
                tda_sl = _load_pkl(cache_dir, 'tda_csd.pkl')
            else:
                logger.info("  Running TDA on Surface Laplacian data...")
                tda_sl = _run_tda_pipeline(dist_per_epoch_sl, ch_names)
                _save_pkl(cache_dir, 'tda_csd.pkl', tda_sl)

        # ── Step 4: Betti features (group analysis) ───────────────────────────
        if _cache_exists(cache_dir, 'betti_features.pkl'):
            logger.info("[4/5] Betti features — loading from cache")
            betti_features = _load_pkl(cache_dir, 'betti_features.pkl')
        else:
            logger.info("[4/5] Extracting Betti features...")
            betti_features = _extract_betti_features(tda_results)
            _save_pkl(cache_dir, 'betti_features.pkl', betti_features)
            _save_pkl(out_dir, 'betti_features.pkl', betti_features)
            logger.info("Saved betti_features.pkl")

        # Surface Laplacian betti features
        if config.COMPUTE_SURFACE_LAPLACIAN and tda_sl is not None:
            if _cache_exists(cache_dir, 'betti_features_csd.pkl'):
                logger.debug("  Surface Laplacian Betti features — loading from cache")
            else:
                logger.info("  Extracting Surface Laplacian Betti features...")
                betti_features_sl = _extract_betti_features(tda_sl)
                _save_pkl(cache_dir, 'betti_features_csd.pkl', betti_features_sl)
                _save_pkl(out_dir, 'betti_features_csd.pkl', betti_features_sl)
                logger.info("Saved betti_features_csd.pkl")

        # ── Step 5: Graph theory ──────────────────────────────────────────────
        if _cache_exists(cache_dir, 'graph_results.pkl'):
            logger.info("[5/5] Graph theory — loading from cache")
            graph_results = _load_pkl(cache_dir, 'graph_results.pkl')
        else:
            logger.info("[5/5] Running graph theory pipeline...")
            graph_results = _run_graph_pipeline(conn_matrices, ch_names)
            _save_pkl(cache_dir, 'graph_results.pkl', graph_results)

        if _cache_exists(cache_dir, 'density_sweep.pkl'):
            logger.debug("  Density sweep — loading from cache")
        else:
            logger.info("  Running density sweep (0.10 -> 0.50, 9 steps)...")
            sweep, densities = _run_density_sweep(conn_matrices, ch_names)
            _save_pkl(cache_dir, 'density_sweep.pkl', (sweep, densities))
            _write_key(cache_dir, 'graph', graph_key)

        # Surface Laplacian graph theory + density sweep
        if config.COMPUTE_SURFACE_LAPLACIAN and conn_sl is not None:
            if _cache_exists(cache_dir, 'graph_results_csd.pkl'):
                logger.debug("  Surface Laplacian graph results — loading from cache")
            else:
                logger.info("  Running Surface Laplacian graph theory pipeline...")
                graph_results_sl = _run_graph_pipeline(conn_sl, ch_names)
                _save_pkl(cache_dir, 'graph_results_csd.pkl', graph_results_sl)

            if _cache_exists(cache_dir, 'density_sweep_csd.pkl'):
                logger.debug("  Surface Laplacian density sweep — loading from cache")
            else:
                logger.info("  Running Surface Laplacian density sweep (0.10 -> 0.50, 9 steps)...")
                sweep_sl, densities_sl = _run_density_sweep(conn_sl, ch_names)
                _save_pkl(cache_dir, 'density_sweep_csd.pkl', (sweep_sl, densities_sl))

        # ── Figures (optional) ────────────────────────────────────────────────
        if figures:
            # Mean conn/dist matrices passed to figures (averaged across epochs)
            dist_matrices_mean = {k: v.mean(axis=0) for k, v in dist_per_epoch.items()}
            logger.info("Generating figures...")
            _generate_figures(
                out_dir, ch_names,
                conn_matrices, dist_matrices_mean, tda_results, graph_results,
                conn_sl=conn_sl, dist_sl=dist_sl,
                tda_sl=tda_sl,
            )

        logger.info("STATUS: complete")
        return betti_features

    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed:\n{traceback.format_exc()}")
        logger.info(f"STATUS: failed — {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES  (only called when figures=True)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_figures(out_dir, ch_names,
                      conn_matrices, dist_matrices, tda_results, graph_results,
                      conn_sl=None, dist_sl=None, tda_sl=None):
    """Import and run all plot functions from TDADHD_copy.py."""
    import importlib.util
    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proto_path = os.path.join(repo_root, 'TDADHD_copy.py')

    spec   = importlib.util.spec_from_file_location('TDADHD_copy', proto_path)
    proto  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)

    no_filter_dir = os.path.join(out_dir, 'no_spatial_filter')
    sl_dir = os.path.join(out_dir, 'surface_laplacian')
    os.makedirs(no_filter_dir, exist_ok=True)
    os.makedirs(sl_dir, exist_ok=True)

    # Load density sweep results for plotting
    cache_dir = os.path.join(out_dir, '.cache')
    sweep, densities = _load_pkl(cache_dir, 'density_sweep.pkl')

    # Fixed colormap ceilings for network_viz — anchors vmin=0 and caps the
    # color scale at a consistent value so ADHD and TDC plots are directly
    # comparable.  Adjust if the edge_weight_distributions figure shows that
    # most subjects' values are clipping at (or far below) these ceilings.
    # Surface Laplacian attenuates volume-conduction-driven coherence/correlation so its
    # ceilings are lower than the no-filter pipeline.
    VMAX_NF = {'coherence': 0.80, 'wpli': 0.30, 'correlation': 0.80}
    VMAX_SL = {'coherence': 0.50, 'wpli': 0.20, 'correlation': 0.60}

    lbl = 'No Filter'
    proto.plot_connectivity_heatmaps(conn_matrices, ch_names, out_dir=no_filter_dir, pipeline_label=lbl)
    proto.plot_betti_curves(tda_results, out_dir=no_filter_dir, pipeline_label=lbl)
    proto.plot_persistence_diagrams(tda_results, out_dir=no_filter_dir, pipeline_label=lbl)
    proto.plot_graph_metrics(graph_results, out_dir=no_filter_dir, pipeline_label=lbl)
    proto.plot_density_sweep(sweep, densities, out_dir=no_filter_dir, pipeline_label=lbl)
    proto.plot_network_viz(conn_matrices, ch_names, out_dir=no_filter_dir, pipeline_label=lbl,
                           vmax_override=VMAX_NF)
    proto.plot_edge_weight_distributions(conn_matrices, ch_names, out_dir=no_filter_dir, pipeline_label=lbl)

    if conn_sl is not None and tda_sl is not None:
        lbl = 'Surface Laplacian'
        graph_results_sl = _load_pkl(cache_dir, 'graph_results_csd.pkl')
        sweep_sl, densities_sl = _load_pkl(cache_dir, 'density_sweep_csd.pkl')
        proto.plot_connectivity_heatmaps(conn_sl, ch_names, out_dir=sl_dir, pipeline_label=lbl)
        proto.plot_betti_curves(tda_sl, out_dir=sl_dir, pipeline_label=lbl)
        proto.plot_graph_metrics(graph_results_sl, out_dir=sl_dir, pipeline_label=lbl)
        proto.plot_density_sweep(sweep_sl, densities_sl, out_dir=sl_dir, pipeline_label=lbl)
        proto.plot_network_viz(conn_sl, ch_names, out_dir=sl_dir, pipeline_label=lbl,
                               vmax_override=VMAX_SL)
        proto.plot_edge_weight_distributions(conn_sl, ch_names, out_dir=sl_dir, pipeline_label=lbl)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process one EEG subject through the TDA pipeline.')
    parser.add_argument('mat_path',          help='Path to subject .mat file (e.g. data/raw/v10p.mat)')
    parser.add_argument('--force',   action='store_true', help='Ignore all caches and recompute')
    parser.add_argument('--figures', action='store_true', help='Generate per-subject visualizations')
    args = parser.parse_args()

    result = run_subject(args.mat_path, force=args.force, figures=args.figures)
    sys.exit(0 if result is not None else 1)
