import os, re
import numpy as np
import pandas as pd
import heartpy as hp
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, roc_auc_score, precision_score, recall_score

# Signal settings
SR = 100
CHUNK = 12 * SR
DC_WIN_SEC = 2.0
USE_BANDPASS_FOR_DET = True

FEATURES = ['PI', 'Amp_PeakMean', 'RMSSD', 'BPM']
FEATS_Z  = [f'{f}_z' for f in FEATURES]

# Manifest
@dataclass
class TrialPair:
    subject_id: str
    trial_id: str
    baseline_path: str
    holding_path: str

def load_manifest_trials(manifest_path: str) -> List[TrialPair]:
    m = pd.read_csv(manifest_path)

    # column name 호환
    rename_map = {}
    if 'baseline_csv' in m.columns: rename_map['baseline_csv'] = 'baseline_path'
    if 'holding_csv'  in m.columns: rename_map['holding_csv']  = 'holding_path'
    m = m.rename(columns=rename_map)

    req = {'subject_id','trial_id','baseline_path','holding_path'}
    if not req.issubset(m.columns):
        raise ValueError(f"manifest 열 누락: {sorted(req)}")

    out = []
    for _, r in m.iterrows():
        out.append(TrialPair(
            subject_id=str(r['subject_id']).strip(),
            trial_id=str(r['trial_id']).strip(),
            baseline_path=str(r['baseline_path']).strip(),
            holding_path=str(r['holding_path']).strip(),
        ))
    return out

# raw loader: txt numbers
def load_values_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    toks = re.split(r'[,\s]+', s.strip())

    v = pd.to_numeric(pd.Series(toks), errors='coerce').dropna().astype(float).values
    if v.size == 0:
        raise ValueError(f"No numeric data parsed from: {path}")
    return v.astype(float)

def to_chunks(vec: np.ndarray, chunk: int, overlap: float = 0.2) -> list:
    step = int(chunk * (1 - overlap))
    return [vec[i:i+chunk] for i in range(0, len(vec) - chunk + 1, step)]

# preprocessing + peak detection
def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    c = np.cumsum(xp, dtype=float); c[win:] = c[win:] - c[:-win]
    ma = c[win-1:] / float(win)
    return ma[:len(x)]

def bandpass_sos(x: np.ndarray, sr: float, lo=0.5, hi=8.0, order=2):
    nyq = 0.5 * sr
    sos = butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def refine_peaks(peaks: List[int], sr: float, min_rr_sec=0.30) -> np.ndarray:
    if len(peaks) < 2: return np.array(peaks, dtype=int)
    out = [peaks[0]]
    for p in peaks[1:]:
        if (p - out[-1]) / sr >= min_rr_sec:
            out.append(p)
    return np.array(out, dtype=int)

def detect_peaks_ir(raw_chunk: np.ndarray, sr: int):
    if len(raw_chunk) < CHUNK or np.all(np.isnan(raw_chunk)):
        return False, np.array([], dtype=int), np.nan, raw_chunk, raw_chunk, {}
    dc = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac = raw_chunk - dc
    sig_det = bandpass_sos(ac, sr) if USE_BANDPASS_FOR_DET else ac
    try:
        wd, m = hp.process(sig_det, sample_rate=sr)
        raw_peaks = [p for p in wd.get('peaklist', []) if p not in wd.get('removed_beats', [])]
        peaks = refine_peaks(raw_peaks, sr, min_rr_sec=0.30)
        bpm = float(m.get('bpm', np.nan))
    except Exception:
        return False, np.array([], dtype=int), np.nan, sig_det, ac, {}
    sec = len(raw_chunk) / sr
    valid = (30 < bpm < 200) and (len(peaks) > 8) and (len(peaks) > sec / 2.0)
    return bool(valid), peaks, bpm, sig_det, ac, m

def compute_PI(ac_full: np.ndarray, dc_full: np.ndarray) -> float:
    dc_mean = float(np.nanmean(dc_full)) if np.isfinite(dc_full).any() else np.nan
    ac_rms  = float(np.sqrt(np.nanmean((ac_full - np.nanmean(ac_full))**2))) if np.isfinite(ac_full).any() else np.nan
    return 100.0 * ac_rms / dc_mean if (dc_mean and np.isfinite(dc_mean)) else np.nan

def extract_features_from_chunk(raw_chunk: np.ndarray, sr: int) -> Optional[dict]:
    valid, peaks, bpm, sig_det, ac, m = detect_peaks_ir(raw_chunk, sr)
    if not valid:
        return None

    dc_full = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac_full = raw_chunk - dc_full

    PI   = compute_PI(ac_full, dc_full)
    RMSSD = float(m.get('rmssd', np.nan)) if m.get('rmssd') is not None else np.nan
    SDNN  = float(m.get('sdnn',  np.nan)) if m.get('sdnn')  is not None else np.nan
    BPM   = float(m.get('bpm',   np.nan)) if m.get('bpm')   is not None else np.nan

    # Amp_PeakMean: peak 위치의 raw 신호 진폭 평균
    if len(peaks) >= 2:
        Amp_PeakMean = float(np.mean(raw_chunk[peaks]))
    else:
        Amp_PeakMean = np.nan

    feat = {
        'PI': PI,
        'Amp_PeakMean': Amp_PeakMean,
        'RMSSD': RMSSD,
        'SDNN': SDNN,
        'BPM': BPM,
    }

    if not all(np.isfinite(v) for v in feat.values()):
        return None
    return feat

# build features from ONE row (trial pair)
def build_features_df_from_trial(pair: TrialPair, sr: int = SR, chunk: int = CHUNK) -> pd.DataFrame:
    rb = load_values_txt(pair.baseline_path)
    rh = load_values_txt(pair.holding_path)

    rows = []
    for cond, raw in [('baseline', rb), ('holding', rh)]:
        for chunk_idx, ch in enumerate(to_chunks(raw, chunk)):
            feat = extract_features_from_chunk(ch, sr)
            if feat is None:
                continue
            rows.append({
                'subject_id': pair.subject_id,
                'trial_id': pair.trial_id,
                'condition': cond,
                'chunk_idx': chunk_idx,
                **feat
            })
    return pd.DataFrame(rows)

def build_features_df_from_manifest(manifest_path: str) -> pd.DataFrame:
    pairs = load_manifest_trials(manifest_path)
    frames = []
    for p in pairs:
        df = build_features_df_from_trial(p)
        if len(df) == 0:
            # trial 단위로 유효 청크가 0이면 그냥 skip
            continue
        frames.append(df)
    if not frames:
        raise ValueError("유효 청크가 있는 trial이 없습니다.")
    return pd.concat(frames, ignore_index=True)

# normalization: subject baseline only (robust)
def _robust_mu_sd(g: pd.DataFrame, feats) -> Tuple[pd.Series, pd.Series]:
    mu = {}; sd = {}
    for f in feats:
        x = g[f].dropna().values
        if x.size == 0:
            mu[f], sd[f] = np.nan, np.nan
            continue
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        sd_robust = 1.4826 * mad
        mu[f], sd[f] = med, (sd_robust if sd_robust > 0 else np.nan)
    return pd.Series(mu), pd.Series(sd)

def normalize_by_subject_baseline(df: pd.DataFrame, feats=FEATURES, eps=1e-6, robust: bool=True) -> pd.DataFrame:
    base = df[df['condition']=='baseline'].copy()
    if robust:
        stats_rows = []
        for sid, g in base.groupby('subject_id'):
            mu, sd = _robust_mu_sd(g, feats)
            stats_rows.append({
                'subject_id': sid,
                **{f'{f}_mean': mu[f] for f in feats},
                **{f'{f}_std' : sd[f] for f in feats},
            })
        stats = pd.DataFrame(stats_rows).set_index('subject_id')
    else:
        stats = base.groupby('subject_id')[feats].agg(['mean','std'])
        stats.columns = [f'{a}_{b}' for a,b in stats.columns]

    out = df.merge(stats, left_on='subject_id', right_index=True, how='left')
    for f in feats:
        mu = out[f'{f}_mean']
        sd = out[f'{f}_std'].replace(0, np.nan)
        out[f'{f}_z'] = (out[f] - mu) / (sd.fillna(eps))
    return out

# undersampling baseline per subject
def undersample_baseline_per_subject(df: pd.DataFrame, ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    rs = np.random.RandomState(random_state)
    kept = []
    for sid, g in df.groupby('subject_id'):
        g_h = g[g['condition']=='holding']
        g_b = g[g['condition']=='baseline']
        n_h = len(g_h)
        if n_h == 0 or len(g_b) == 0:
            kept.append(g); continue
        n_b_keep = int(np.ceil(n_h * ratio))
        if len(g_b) > n_b_keep:
            g_b = g_b.sample(n=n_b_keep, random_state=rs)
        kept.append(pd.concat([g_b, g_h], ignore_index=True))
    return pd.concat(kept, ignore_index=True)

# model pipeline
def make_classifier_pipeline(use_poly2: bool, logreg_c: float, random_state: int = 42) -> Pipeline:
    steps = []
    if use_poly2:
        steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
    steps += [
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced',
                                      random_state=random_state, C=logreg_c,
                                      penalty='l2', solver='lbfgs'))
    ]
    return Pipeline(steps)

# threshold tuning (subject-level)
def subject_level_aggregate(y_true, y_prob, subjects, conditions, thr):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob,
                       'subject': subjects, 'cond': conditions})
    g = (df.groupby(['subject','cond'])
           .agg(y_true=('y_true','mean'), y_prob=('y_prob','mean'))
           .reset_index())
    y_true_s = g['y_true'].astype(int).values
    y_pred_s = (g['y_prob'] >= thr).astype(int).values
    return y_true_s, y_pred_s, g

def best_threshold_subjectF1(y_true, y_prob, subjects, conditions):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0: return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in thr:
        yS_true, yS_pred, _ = subject_level_aggregate(y_true, y_prob, subjects, conditions, t)
        f1 = f1_score(yS_true, yS_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

def best_threshold_fbeta(y_true, y_prob, beta=0.5):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0: return 0.5, 0.0
    fb = (1+beta*beta) * (p*r) / (beta*beta*p + r + 1e-12)
    idx = int(np.nanargmax(fb[:-1]))
    return float(thr[idx]), float(fb[idx])

def best_threshold_target_precision(y_true, y_prob, target_prec=0.9):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0: return 0.5, 0.0
    idxs = np.where(p[:-1] >= target_prec)[0]
    if len(idxs) == 0:
        return best_threshold_fbeta(y_true, y_prob, beta=1.0)
    t = float(thr[idxs[-1]])
    f1 = (2*p[idxs[-1]]*r[idxs[-1]]) / (p[idxs[-1]] + r[idxs[-1]] + 1e-12)
    return t, float(f1)

def pick_threshold(y_tr, y_trp, subjects, conditions, strategy: str, f_beta: float, target_prec: float):
    if strategy == "subject_f1":
        return best_threshold_subjectF1(y_tr, y_trp, subjects, conditions)
    if strategy == "fbeta":
        return best_threshold_fbeta(y_tr, y_trp, beta=f_beta)
    if strategy == "target_precision":
        return best_threshold_target_precision(y_tr, y_trp, target_prec=target_prec)
    return best_threshold_fbeta(y_tr, y_trp, beta=1.0)

# metrics helpers
def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    return acc, roc, f1, prec, rec, y_pred