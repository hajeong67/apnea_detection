"""
단일 IR CSV(value 열) → 12초 청크 → (유효성+HeartPy) PI/RMSSD → subject baseline z-score
→ 모델 예측(유효 청크만) → 플롯:
  - 전체 AC 파형 회색
  - invalid(유효 아님) 청크 파란색
  - holding(True) 청크 빨간색
per-chunk CSV 함께 저장
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import heartpy as hp
from typing import Optional
from scipy.signal import butter, sosfiltfilt

MODEL_PATH = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\model_version\ir_model_v1.joblib"

INPUT_RAW_CSV      = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\dataset\holding_breath\ir_all\hajeong_ppg_ir.csv"
OPTIONAL_BASELINE  = None

OUT_DIR  = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\output\test_run\v1"
OUT_PNG  = os.path.join(OUT_DIR, "test_raw_highlight.png")
OUT_CSV  = os.path.join(OUT_DIR, "test_raw_per_chunk.csv")

# 데이터/전처리
SR            = 100
CHUNK_SEC     = 12.0
DC_WIN_SEC    = 2.0
DETECT_MODE   = "bp"
BP_LO, BP_HI  = 0.5, 8.0
MIN_RR_SEC    = 0.30
BASELINE_WINDOW_SEC = 120  # baseline CSV 없을 때 앞 구간으로 μ,σ 추정
THR_OVERRIDE = None
DELTA_THR = 0.09

# 플롯
Y_MIN, Y_MAX  = None, None   # AC 자동 스케일 권장
WIDTH_PER_MIN = 4.0
ANNOTATE      = False

ROBUST_BASE_STATS = True               # median/MAD 기반
WINSOR = None                 # (0~1) 극단치 컷, 끄려면 None
MIN_BASE_VALID = 6                     # 유효 베이스라인 청크 최소개수(≥72초)
SD_FLOOR = {'PI': 0.02, 'RMSSD': 3.0}   # σ 하한
ZSCORE_CLIP = 5.0                      # z-score clip [-5,5], 끄려면 None

CALIBRATE_FROM_BASE = False             # 앞 120초 확률분포로 임계치 하한 보정
BASE_ALPHA = 0.05                      # baseline에서 허용 FP 비율(상위 95%를 하한으로)

HYSTERESIS_ON = False                   # 연속성 필터
HYS_MIN_CONSEC_ON = 2                  # 최소 2청크 연속 True일 때만 유지
HYS_THR_GAP = 0.05                     # off 임계치 = thr_on - gap

FEATURES = ['PI', 'RMSSD']
FEATS_Z  = [f"{f}_z" for f in FEATURES]

def load_values(path: str, value_col: str = "value") -> np.ndarray:
    """CSV에서 value 열(없으면 후보/마지막 숫자열)만 1D float로 로드."""
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    if df.empty:
        raise ValueError(f"빈 CSV: {path}")
    cols_norm = [str(c).strip() for c in df.columns]
    df.columns = cols_norm
    cols_lower = [c.lower() for c in cols_norm]

    target = None
    if value_col and value_col.lower() in cols_lower:
        target = cols_norm[cols_lower.index(value_col.lower())]
    else:
        for name in ["value", "val", "ir", "ppg", "signal", "y"]:
            if name in cols_lower:
                target = cols_norm[cols_lower.index(name)]
                break
    if target is None:
        num_cols = []
        for c in cols_norm:
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().sum() >= max(1, int(0.6*len(ser))):
                num_cols.append(c)
        if not num_cols:
            raise ValueError(f"value/숫자열 없음. columns={cols_norm}")
        target = num_cols[-1]
    vals = pd.to_numeric(df[target], errors="coerce").dropna().astype(float).values
    if vals.size == 0:
        raise ValueError(f"선택 열 '{target}'에서 숫자 없음")
    return vals

def to_chunks(vec: np.ndarray, chunk_len: int) -> list:
    return [vec[i:i+chunk_len] for i in range(0, len(vec), chunk_len)]

def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(float).copy()
    pad = win // 2
    xp = np.pad(x.astype(float), (pad, pad), mode='edge')
    c = np.cumsum(xp, dtype=float); c[win:] = c[win:] - c[:-win]
    ma = c[win-1:] / float(win)
    return ma[:len(x)]

def remove_dc(raw: np.ndarray, sr: int, win_sec: float = 2.0):
    win = max(1, int(win_sec * sr))
    dc = moving_avg(raw.astype(float), win)
    ac = raw - dc
    return dc, ac

def bandpass_sos(x: np.ndarray, sr: float, lo=0.5, hi=8.0, order=2):
    nyq = 0.5 * sr
    sos = butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def refine_peaks(peaks: list, sr: float, min_rr_sec=0.30) -> np.ndarray:
    if len(peaks) < 2: return np.array(peaks, dtype=int)
    out = [peaks[0]]
    for p in peaks[1:]:
        if (p - out[-1]) / sr >= min_rr_sec:
            out.append(p)
    return np.array(out, dtype=int)

def detect_peaks_ir(raw_chunk: np.ndarray, sr: int):
    """전처리 → HeartPy → 유효성 판정(훈련과 동일)."""
    dc = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac = raw_chunk - dc
    sig_det = ac if DETECT_MODE == "ac" else bandpass_sos(ac, sr, lo=BP_LO, hi=BP_HI, order=2)
    try:
        wd, m = hp.process(sig_det, sample_rate=sr)
        raw_peaks = [p for p in wd.get('peaklist', []) if p not in wd.get('removed_beats', [])]
        peaks = refine_peaks(raw_peaks, sr, min_rr_sec=MIN_RR_SEC)
        bpm = float(m.get('bpm', np.nan))
    except Exception:
        return False, np.array([], dtype=int), np.nan, sig_det, ac, {}

    sec = len(raw_chunk) / sr
    valid = (30 < bpm < 200) and (len(peaks) > 8) and (len(peaks) > sec/2.0)
    return bool(valid), peaks, bpm, sig_det, ac, m

def compute_features_per_chunk(raw: np.ndarray, sr: int, chunk_sec: float) -> pd.DataFrame:
    """청크별 PI/RMSSD/BPM/valid 계산."""
    chunk_len = int(round(chunk_sec * sr))
    rows = []
    for idx, ch in enumerate(to_chunks(raw, chunk_len)):
        s = idx * chunk_len
        e = s + len(ch)
        t_center = ((s + e) / 2.0) / float(sr)

        valid, peaks, bpm, sig_det, ac, m = detect_peaks_ir(ch, sr)

        # PI
        dc_full = moving_avg(ch.astype(float), int(DC_WIN_SEC * sr))
        ac_full = ch - dc_full
        dc_mean = float(np.nanmean(dc_full)) if np.isfinite(dc_full).any() else np.nan
        ac_rms  = float(np.sqrt(np.nanmean((ac_full - np.nanmean(ac_full))**2))) if np.isfinite(ac_full).any() else np.nan
        PI = 100.0 * ac_rms / dc_mean if (pd.notna(dc_mean) and dc_mean != 0.0) else np.nan

        # RMSSD
        RMSSD = float(m.get('rmssd', np.nan)) if m.get('rmssd', None) is not None else np.nan

        rows.append({
            "chunk_idx": idx, "start": s, "end": e, "t_center": t_center,
            "valid": bool(valid), "BPM": bpm,
            "PI": PI, "RMSSD": RMSSD
        })
    return pd.DataFrame(rows)

# baseline μ,σ 추정
def _winsorize(x, lo=0.05, hi=0.95):
    if x.size == 0 or lo is None or hi is None: return x
    ql, qh = np.nanpercentile(x, [lo*100, hi*100])
    return np.clip(x, ql, qh)

def _robust_mean_sd_frame(df: pd.DataFrame):
    mu, sd_used, sd_raw = {}, {}, {}
    for col in FEATURES:
        x = df[col].dropna().values
        if x.size == 0:
            mu[col], sd_used[col], sd_raw[col] = np.nan, np.nan, np.nan
            continue
        # robust mean/sd
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        s_raw = 1.4826 * mad
        s_used = max(s_raw, SD_FLOOR[col])
        mu[col] = med
        sd_raw[col] = s_raw
        sd_used[col] = s_used
    return pd.Series(mu), pd.Series(sd_used), pd.Series(sd_raw)

def estimate_baseline_stats(baseline_csv: Optional[str],
                            test_raw: np.ndarray,
                            sr: int,
                            chunk_sec: float,
                            baseline_window_sec: int = 120) -> dict:
    """
    baseline CSV가 있으면 그 파일의 유효 청크만, 없으면 test_raw 앞 구간의 유효 청크만 사용.
    유효 청크가 부족하면 전체에서 폴백.
    """
    def _stats_from_raw(raw):
        dfb = compute_features_per_chunk(raw, sr, chunk_sec)
        use = dfb[dfb['valid']][FEATURES].dropna()
        if len(use) < MIN_BASE_VALID:
            use = dfb[FEATURES].dropna()
        if ROBUST_BASE_STATS:
            mu, sd_used, sd_raw = _robust_mean_sd_frame(use)
        else:
            mu = use.mean(numeric_only=True)
            sd_raw = use.std(numeric_only=True).replace(0, np.nan)  # raw std
            sd_used = sd_raw.copy()
            for k in FEATURES:
                if pd.isna(sd_used[k]) or sd_used[k] < SD_FLOOR[k]:
                    sd_used[k] = SD_FLOOR[k]

        return {"mu": mu, "sd": sd_used, "sd_raw": sd_raw, "df_base_used": dfb}

    if baseline_csv and os.path.exists(baseline_csv):
        base_raw = load_values(baseline_csv)
        return _stats_from_raw(base_raw)
    else:
        n_use = max(1, int(baseline_window_sec // chunk_sec))
        head = test_raw[: int(n_use * chunk_sec * sr)]
        return _stats_from_raw(head)

def zscore_with_stats(df: pd.DataFrame, stats: dict, eps=1e-6) -> pd.DataFrame:
    out = df.copy()
    for f in FEATURES:
        mu = stats["mu"].get(f, np.nan)
        sd = stats["sd"].get(f, np.nan)
        out[f"{f}_z"] = (out[f] - mu) / (sd if (pd.notna(sd) and sd != 0.0) else eps)
    if ZSCORE_CLIP is not None:
        out[[f"{f}_z" for f in FEATURES]] = out[[f"{f}_z" for f in FEATURES]].clip(-abs(ZSCORE_CLIP), abs(ZSCORE_CLIP))
    return out

def unpack_model_bundle(obj):
    """joblib에서 로드한 객체에서 (Pipeline, threshold) 추출."""
    if isinstance(obj, dict) and 'bundle' in obj:
        obj = obj['bundle']
    if isinstance(obj, dict) and 'model' in obj and 'threshold' in obj:
        return obj['model'], float(obj['threshold'])
    if hasattr(obj, 'predict_proba'):
        return obj, 0.5
    raise TypeError("지원하지 않는 모델 포맷")

def calibrate_threshold_from_baseline(clf, df_feat_full, stats, base_thr, sr, chunk_sec,
                                      base_alpha=0.05):
    head_end = int(120.0 * sr)
    base_chunks = df_feat_full[(df_feat_full['end'] <= head_end) & (df_feat_full['valid'])]
    if base_chunks.empty:
        return base_thr
    tmp = base_chunks.copy()
    for f in FEATURES:
        mu = stats['mu'].get(f, np.nan); sd = stats['sd'].get(f, np.nan) or 1e-6
        tmp[f'{f}_z'] = (tmp[f] - mu) / sd
    if ZSCORE_CLIP is not None:
        tmp[[f"{f}_z" for f in FEATURES]] = tmp[[f"{f}_z" for f in FEATURES]].clip(-abs(ZSCORE_CLIP), abs(ZSCORE_CLIP))

    mask = ~tmp[[f"{f}_z" for f in FEATURES]].isna().any(axis=1)
    if not mask.any(): return base_thr
    Xb = tmp.loc[mask, [f"{f}_z" for f in FEATURES]].values
    p_base = clf.predict_proba(Xb)[:, 1]
    percl = float(np.nanpercentile(p_base, 100*(1.0 - base_alpha)))  # 상위 1-α 퍼센타일
    return min(max(base_thr, percl), 0.99)

def hysteresis_labels(y_prob, thr_on, thr_off=None, min_consec_on=2, min_consec_off=1):
    if thr_off is None:
        thr_off = max(0.0, thr_on - 0.05)
    state = False
    on_streak = off_streak = 0
    out = np.zeros(len(y_prob), dtype=bool)

    for i, p in enumerate(y_prob):
        if not np.isfinite(p):
            out[i] = False
            continue
        if not state:  # OFF
            on_streak = on_streak + 1 if p >= thr_on else 0
            if on_streak >= min_consec_on:
                state, on_streak = True, 0
        else:          # ON
            off_streak = off_streak + 1 if p < thr_off else 0
            if off_streak >= min_consec_off:
                state, off_streak = False, 0
        out[i] = state
    return out

def predict_and_plot_highlights(model_path: str,
                                input_raw_csv: str,
                                out_png: str,
                                out_csv: str,
                                optional_baseline_csv: Optional[str] = None,
                                sr: int = 100,
                                chunk_sec: float = 12.0):
    # 모델
    obj = joblib.load(model_path)
    clf, thr = unpack_model_bundle(obj)
    base_thr = float(thr)

    # 데이터 & 피처 (전체)
    raw = load_values(input_raw_csv)
    df_feat_full = compute_features_per_chunk(raw, sr, chunk_sec)

    # baseline μ,σ → z-score
    stats = estimate_baseline_stats(optional_baseline_csv, raw, sr, chunk_sec, BASELINE_WINDOW_SEC)
    df_z = zscore_with_stats(df_feat_full, stats, eps=1e-6)

    # 예측: 유효 청크만 사용
    mask_pred = df_z['valid'] & (~df_z[[f"{f}_z" for f in FEATURES]].isna().any(axis=1))
    y_prob = np.full(len(df_z), np.nan, dtype=float)
    if mask_pred.any():
        X = df_z.loc[mask_pred, [f"{f}_z" for f in FEATURES]].values
        y_prob[mask_pred.values] = clf.predict_proba(X)[:, 1]

    # 임계치 결정
    if THR_OVERRIDE is not None:
        thr_used = float(THR_OVERRIDE)
    else:
        thr_used = min(0.99, base_thr + DELTA_THR)
        if CALIBRATE_FROM_BASE:
            thr_cal = calibrate_threshold_from_baseline(clf, df_feat_full, stats, thr_used, sr, chunk_sec, BASE_ALPHA)
            thr_used = max(thr_used, thr_cal)
    print(f"[threshold] base={base_thr:.3f} | used={thr_used:.3f}")

    # 이진 판정
    if HYSTERESIS_ON:
        y_pred_bool = hysteresis_labels(y_prob, thr_on=thr_used, thr_off=max(0.0, thr_used - HYS_THR_GAP),
                                        min_consec_on=HYS_MIN_CONSEC_ON)
    else:
        y_pred_bool = (y_prob >= thr_used)
    y_pred = np.where(np.isnan(y_prob), False, y_pred_bool)
    used_for_pred = mask_pred.values.astype(bool)

    # AC 파형 플롯
    _, ac_full = remove_dc(raw, sr, win_sec=DC_WIN_SEC)
    n = len(ac_full); t = np.arange(n, dtype=float) / float(sr)
    dur_min = max(1e-6, (n/sr)/60.0)
    figsize = (max(12.0, dur_min*WIDTH_PER_MIN), 4.0)

    fig, ax = plt.subplots(figsize=figsize, dpi=180, constrained_layout=True)
    ax.plot(t, ac_full, lw=0.6, color='0.5')

    def _auto_ylim(x):
        x = x[np.isfinite(x)]
        if x.size == 0: return (-1.0, 1.0)
        lo, hi = np.nanpercentile(x, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        return (lo - pad, hi + pad)

    if (Y_MIN is None) or (Y_MAX is None):
        ax.set_ylim(*_auto_ylim(ac_full))
    else:
        rmin, rmax = float(np.nanmin(ac_full)), float(np.nanmax(ac_full))
        ax.set_ylim(*( _auto_ylim(ac_full) if (rmax < Y_MIN) or (rmin > Y_MAX) else (Y_MIN, Y_MAX) ))

    chunk_len = int(round(chunk_sec * sr))
    for _, r in df_z[~df_z['valid']].iterrows():
        s, e = int(r['start']), int(r['end'])
        if s >= e: continue
        ax.plot(t[s:e], ac_full[s:e], lw=1.0, color='tab:blue')
    for _, r in df_z.iterrows():
        if bool(y_pred[int(r['chunk_idx'])]):
            s, e = int(r['start']), int(r['end'])
            if s >= e: continue
            ax.plot(t[s:e], ac_full[s:e], lw=1.0, color='red')
            if ANNOTATE:
                cx = 0.5*(t[s] + t[e-1])
                ax.text(cx, 0.98, f"#{int(r['chunk_idx'])}", transform=ax.transAxes,
                        va='top', ha='center', fontsize=8)

    ax.set_xlabel("Time(s)")
    ax.set_ylabel("IR AC")
    ax.set_title(
        f"IR PPG Apnea Prediction | thr={thr_used:.2f}",
        fontsize=30,
    )
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=18, prune='both'))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='0.5', lw=2.0, label='AC (all)'),
        Line2D([0], [0], color='tab:blue', lw=3.0, label='Invalid segment'),
        Line2D([0], [0], color='red', lw=3.0, label='Predicted event'),
        # 필요하면 GT를 실제로 그릴 때 열고 사용 (노랑):
        Line2D([0], [0], color='gold',    lw=13.0, label='Ground truth', alpha=0.3),
    ]

    leg = ax.legend(
        handles=handles,
        loc='upper right',
        fontsize=18,  # ← 범례 글자 크기
        title_fontsize=16,  # 선택: 제목 크기
        handlelength=3.0,  # 선 길이
        labelspacing=0.6,  # 라벨 간격
        borderpad=0.6,  # 테두리-내용 간격
        frameon=True,
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

    # per-chunk 리포트
    out_df = df_z.copy()
    out_df['used_for_pred'] = used_for_pred
    out_df['p_holding'] = y_prob
    out_df['y_pred']    = np.where(used_for_pred, y_pred.astype(int), np.nan)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 디버그 출력
    print("μ:", stats["mu"].to_dict(), "σ:", stats["sd"].to_dict())
    print("p quantiles:", np.nanpercentile(y_prob[np.isfinite(y_prob)], [10,50,90,95,99]))
    print("positive_rate@", thr_used, "=", float(np.nanmean(y_prob >= thr_used)))
    print("μ_used:", stats["mu"].to_dict(),
          "σ_raw:", stats["sd_raw"].to_dict(),
          "σ_used:", stats["sd"].to_dict())

    print(f"✅ 저장: {out_png}")
    print(f"🧾 저장: {out_csv}")
    print(f"유효 청크 수: {int(df_z['valid'].sum())}/{len(df_z)} | holding=True: {int(np.nansum(out_df['y_pred']==1))}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_RAW_CSV):
        raise FileNotFoundError(f"INPUT_RAW_CSV 없음: {INPUT_RAW_CSV}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH 없음: {MODEL_PATH}")

    predict_and_plot_highlights(
        model_path=MODEL_PATH,
        input_raw_csv=INPUT_RAW_CSV,
        out_png=OUT_PNG,
        out_csv=OUT_CSV,
        optional_baseline_csv=OPTIONAL_BASELINE,
        sr=SR,
        chunk_sec=CHUNK_SEC
    )
