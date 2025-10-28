import os, re, numpy as np, pandas as pd
import heartpy as hp
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve
import joblib

SR = 100
CHUNK = 12 * SR
DC_WIN_SEC = 2.0
USE_BANDPASS_FOR_DET = True
RANDOM_STATE = 42

# 데이터/피처
RATIO = 1.0                       # holding : baseline = 1 : RATIO (언더샘플)
FEATURES = ['PI', 'RMSSD']
FEATS_Z  = [f'{f}_z' for f in FEATURES]

# 정규화
ROBUST_ZSCORE = True              # True면 median/MAD 기반 z-score
ZSCORE_CLIP   = 5.0               # z-score를 [-5, 5]로 클립 (None이면 비활성)

# 임계치 선택
TUNE_STRATEGY = "subject_f1"      # 'subject_f1' | 'fbeta' | 'target_precision'
F_BETA        = 0.5               # TUNE_STRATEGY='fbeta'일 때 사용(정밀도 강조하려면 0.5 추천)
TARGET_PREC   = 0.90              # TUNE_STRATEGY='target_precision'일 때 목표 정밀도

# 분류기 보수화 정도
USE_POLY2     = False            # True면 2차항 추가(PI^2, RMSSD^2, 교차항)
LOGREG_C      = 0.5              # 작을수록 보수적(과검출 줄이는 방향)

MANIFEST_PATH = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\dataset\manifest_v1.csv"
MODEL_OUT     = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\model_version\ir_model_v1.joblib"

@dataclass
class SubjectPair:
    subject_id: str
    baseline_path: str
    holding_path: str

def load_manifest(manifest_path: str) -> List[SubjectPair]:
    m = pd.read_csv(manifest_path)
    if {'baseline_path','holding_path'}.issubset(m.columns) is False and \
       {'baseline_csv','holding_csv'}.issubset(m.columns):
        m = m.rename(columns={'baseline_csv':'baseline_path','holding_csv':'holding_path'})
    req = {'subject_id','baseline_path','holding_path'}
    if not req.issubset(m.columns):
        raise ValueError(f"manifest 열 누락: {sorted(req)}")
    return [SubjectPair(str(r['subject_id']), str(r['baseline_path']), str(r['holding_path']))
            for _, r in m.iterrows()]

def load_values(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    v = np.fromstring(s, sep=' ')
    if v.size == 0:
        toks = re.split(r'\s+', s.strip())
        v = pd.to_numeric(pd.Series(toks), errors='coerce').dropna().astype(float).values
    return v.astype(float)

def to_chunks(vec: np.ndarray, chunk: int) -> list:
    return [vec[i:i+chunk] for i in range(0, len(vec), chunk)]

# 신호 전처리/유효성/피처
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

def detect_peaks_ir(raw_chunk: np.ndarray, sr: int) -> Tuple[bool, np.ndarray, float, np.ndarray, np.ndarray, dict]:
    """
    반환: valid, peaks, bpm, sig_det, ac, m
    - DC 제거(2s) → (선택) bandpass(0.5–8 Hz) → HeartPy process
    - 유효성: 30<BPM<200, 피크수>8, 피크수>구간초/2
    """
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

def extract_PI_RMSSD_from_chunk(raw_chunk: np.ndarray, sr: int) -> Optional[dict]:
    """유효 청크만 {PI, RMSSD} 반환"""
    valid, _, _, _, _, m = detect_peaks_ir(raw_chunk, sr)
    if not valid:
        return None
    dc_full = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac_full = raw_chunk - dc_full
    PI = compute_PI(ac_full, dc_full)
    RMSSD = float(m.get('rmssd', np.nan)) if m.get('rmssd', None) is not None else np.nan
    if not np.isfinite(PI) or not np.isfinite(RMSSD):
        return None
    return {'PI': PI, 'RMSSD': RMSSD}

def build_features_df_from_raw(subject_id: str, baseline_path: str, holding_path: str) -> pd.DataFrame:
    """raw txt 2개를 읽어 12s 청크 단위로 유효성 검사 후 PI/RMSSD를 산출하여 DF 반환"""
    rb = load_values(baseline_path)
    rh = load_values(holding_path)
    rows = []
    for cond, raw in [('baseline', rb), ('holding', rh)]:
        for ch in to_chunks(raw, CHUNK):
            feat = extract_PI_RMSSD_from_chunk(ch, SR)
            if feat is None:  # invalid skip
                continue
            rows.append({'subject_id': subject_id, 'condition': cond, **feat})
    if not rows:
        raise ValueError(f"{subject_id}: 유효 청크가 없습니다.")
    return pd.DataFrame(rows)

# 정규화/학습
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
            row = {**{f'{f}_mean': mu[f] for f in feats},
                   **{f'{f}_std' : sd[f] for f in feats},
                   'subject_id': sid}
            stats_rows.append(row)
        stats = pd.DataFrame(stats_rows).set_index('subject_id')
    else:
        stats = (base.groupby('subject_id')[feats].agg(['mean','std']))
        stats.columns = [f'{a}_{b}' for a,b in stats.columns]

    out = df.merge(stats, left_on='subject_id', right_index=True, how='left')
    for f in feats:
        mu = out[f'{f}_mean']
        sd = out[f'{f}_std'].replace(0, np.nan)
        out[f'{f}_z'] = (out[f] - mu) / (sd.fillna(eps))
    return out

def undersample_baseline_per_subject(df: pd.DataFrame, ratio: float = 1.0, rng=RANDOM_STATE) -> pd.DataFrame:
    rs = np.random.RandomState(rng)
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

def make_classifier_pipeline(random_state=RANDOM_STATE) -> Pipeline:
    steps = []
    if USE_POLY2:
        steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
    steps += [
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced',
                                      random_state=random_state, C=LOGREG_C, penalty='l2', solver='lbfgs'))
    ]
    return Pipeline(steps)

# 임계치 선택
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
        yS_true, yS_pred, _ = subject_level_aggregate(
            y_true, y_prob, subjects, conditions, t
        )
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
        # 목표 정밀도를 못 맞추면 그냥 F1 기준으로
        return best_threshold_fbeta(y_true, y_prob, beta=1.0)
    t = float(thr[idxs[-1]])  # 정밀도 유지하며 recall 최대화
    # 리포트용 F1
    f1 = (2*p[idxs[-1]]*r[idxs[-1]]) / (p[idxs[-1]] + r[idxs[-1]] + 1e-12)
    return t, float(f1)

def pick_threshold(y_tr, y_trp, subjects, conditions):
    if TUNE_STRATEGY == "subject_f1":
        return best_threshold_subjectF1(y_tr, y_trp, subjects, conditions)
    elif TUNE_STRATEGY == "fbeta":
        return best_threshold_fbeta(y_tr, y_trp, beta=F_BETA)
    elif TUNE_STRATEGY == "target_precision":
        return best_threshold_target_precision(y_tr, y_trp, target_prec=TARGET_PREC)
    else:
        return best_threshold_fbeta(y_tr, y_trp, beta=1.0)

def build_training_df_from_manifest(manifest_path: str) -> pd.DataFrame:
    pairs = load_manifest(manifest_path)
    frames = [build_features_df_from_raw(p.subject_id, p.baseline_path, p.holding_path) for p in pairs]
    return pd.concat(frames, ignore_index=True)

def train_eval_with_undersampling_from_manifest(manifest_path: str, ratio=1.0,
                                                random_state=RANDOM_STATE):
    # raw → 유효 청크 → PI/RMSSD
    df = build_training_df_from_manifest(manifest_path)
    # subject-baseline z-score
    df_norm = normalize_by_subject_baseline(df, feats=FEATURES, robust=ROBUST_ZSCORE)
    # z-score 클리핑(극단치 완화)
    if ZSCORE_CLIP is not None:
        df_norm[FEATS_Z] = df_norm[FEATS_Z].clip(-abs(ZSCORE_CLIP), abs(ZSCORE_CLIP))

    use_cols = FEATS_Z + ['condition','subject_id']
    df_used = df_norm.dropna(subset=use_cols).copy()

    # [추가] 학습에 실제 사용할 수 있는 행 수(정규화/NaN 제거 후) 요약
    _print_counts(df_used, "USABLE ROWS AFTER NORMALIZATION/DROPNA")

    X_all = df_used[FEATS_Z].values
    y_all = (df_used['condition']=='holding').astype(int).values
    groups = df_used['subject_id'].values

    # GroupKFold CV (사람 단위)
    subjects = np.unique(groups)
    n_splits = min(5, len(subjects))
    if n_splits < 2:
        raise ValueError("피험자 수가 2 미만이면 GroupKFold 불가.")
    gkf = GroupKFold(n_splits=n_splits)

    clf = make_classifier_pipeline(random_state)
    accs_c, rocs_c, f1s_c, fold_thresholds = [], [], [], []

    for tr_idx, te_idx in gkf.split(X_all, y_all, groups=groups):
        train_df = df_used.iloc[tr_idx].copy()
        test_df  = df_used.iloc[te_idx].copy()

        # 훈련 폴드: baseline 언더샘플링
        train_df = undersample_baseline_per_subject(train_df, ratio=ratio, rng=random_state)

        X_tr = train_df[FEATS_Z].values
        y_tr = (train_df['condition']=='holding').astype(int).values
        X_te = test_df[FEATS_Z].values
        y_te = (test_df['condition']=='holding').astype(int).values

        clf.fit(X_tr, y_tr)

        # 임계치 선택
        y_trp = clf.predict_proba(X_tr)[:, 1]
        thr, _ = pick_threshold(
            y_tr, y_trp,
            subjects=train_df['subject_id'].values,
            conditions=train_df['condition'].values
        )
        fold_thresholds.append(thr)

        # 테스트
        y_tep = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_tep >= thr).astype(int)
        accs_c.append(accuracy_score(y_te, y_pred))
        rocs_c.append(roc_auc_score(y_te, y_tep))
        f1s_c.append(f1_score(y_te, y_pred))

    thr_final = float(np.median(fold_thresholds))
    print(f"[TRAIN CV | ratio={ratio} | C={LOGREG_C} | poly2={USE_POLY2} | tune={TUNE_STRATEGY}] "
          f"Chunk: Acc {np.mean(accs_c):.3f}±{np.std(accs_c):.3f} | "
          f"ROC {np.mean(rocs_c):.3f}±{np.std(rocs_c):.3f} | "
          f"F1 {np.mean(f1s_c):.3f}±{np.std(f1s_c):.3f} | thr≈{thr_final:.2f}")

    # 전체 학습
    train_all = undersample_baseline_per_subject(df_used, ratio=ratio, rng=random_state)

    # [추가] 최종 모델 학습에 실제 투입된 데이터 수(언더샘플 후) 요약
    _print_counts(train_all, f"FINAL TRAIN DATA (undersample ratio={ratio})")
    X_all = train_all[FEATS_Z].values
    y_all = (train_all['condition']=='holding').astype(int).values
    clf.fit(X_all, y_all)

    return {'model': clf, 'threshold': thr_final}

# === 추가: 간단 통계 출력 헬퍼 ===
def _print_counts(df: pd.DataFrame, title: str):
    n_subj = df['subject_id'].nunique()
    n_all  = len(df)
    n_b    = int((df['condition']=='baseline').sum())
    n_h    = int((df['condition']=='holding').sum())
    print(f"\n[{title}]")
    print(f"- subjects: {n_subj}")
    print(f"- rows total: {n_all}  | baseline: {n_b}  | holding: {n_h}")

    # subject × condition 카운트 (요약)
    wide = (df.groupby(['subject_id','condition'])
              .size().unstack(fill_value=0).rename(columns={'baseline':'baseline','holding':'holding'}))
    # 너무 길면 상위 몇 명만
    preview = wide.reset_index().sort_values(['holding','baseline'], ascending=False).head(26)
    if not preview.empty:
        print("  (top 10 by holding count)")
        print(preview.to_string(index=False))

# === 추가: manifest에서 바로 '유효 청크' 개수만 미리 확인하고 싶을 때 호출용 ===
def summarize_from_manifest(manifest_path: str):
    """
    manifest를 읽어 유효 청크 기반 피처 DF를 만든 뒤, 학습 전 단계의 데이터 개수를 요약 출력
    (build_features_df_from_raw에서 invalid 청크는 이미 제외됨)
    """
    df = build_training_df_from_manifest(manifest_path)
    _print_counts(df, "VALID CHUNKS (features built from raw)")
    return df


if __name__ == "__main__":
    bundle = train_eval_with_undersampling_from_manifest(MANIFEST_PATH, ratio=RATIO)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({'bundle': bundle, 'features': FEATS_Z}, MODEL_OUT)
    print(f"✅ 모델 저장: {MODEL_OUT}")
