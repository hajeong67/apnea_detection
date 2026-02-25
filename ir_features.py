import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import ttest_ind

SR = 100
CHUNK = 12 * SR               # 12초 청크(1200샘플)
DC_WIN_SEC = 2.0              # DC(바탕선) 이동평균 윈도우(초)
USE_BANDPASS_FOR_DETECTION = True  # 검출용 0.5–8 Hz BP 적용(플롯은 AC)
Y_MIN, Y_MAX = -3000, 4000         # y축 고정 범위
ALPHA = 0.1

BASELINE_PATH = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\dataset\holding_breath\ir\0910_ir_jiyeon_baseline.txt"
HOLDING_PATH  = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\dataset\holding_breath\ir\0910_ir_jiyeon_holding3.txt"

OUT_DIR = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\ir_plot\ir_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# 유틸
def load_values(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    v = np.fromstring(s, sep=' ')
    if v.size == 0:
        toks = re.split(r'\s+', s.strip())
        v = pd.to_numeric(pd.Series(toks), errors='coerce').dropna().astype(float).values
    return v.astype(float)

def to_chunks(vec: np.ndarray, chunk: int) -> list[np.ndarray]:
    return [vec[i:i+chunk] for i in range(0, len(vec), chunk)]

def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
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

def refine_peaks(peaks: list[int], sr: float, min_rr_sec=0.30) -> np.ndarray:
    if len(peaks) < 2: return np.array(peaks, dtype=int)
    out = [peaks[0]]
    for p in peaks[1:]:
        if (p - out[-1]) / sr >= min_rr_sec:
            out.append(p)
    return np.array(out, dtype=int)

def bandpower_welch(sig: np.ndarray, sr: float, f_lo: float, f_hi: float) -> float:
    if len(sig) < sr * 2:
        return np.nan
    f, pxx = welch(sig - np.nanmean(sig), fs=sr, nperseg=min(512, len(sig)))
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.trapz(pxx[m], f[m])) if np.any(m) else np.nan

def signal_envelope(sig: np.ndarray, sr: float) -> np.ndarray:
    import scipy.signal as ss
    env = np.abs(ss.hilbert(sig))
    # 0.3초 이동평균으로 살짝 평활
    win = max(1, int(0.3 * sr))
    pad = win // 2
    xp = np.pad(env, (pad, pad), mode='edge')
    c = np.cumsum(xp, dtype=float); c[win:] = c[win:] - c[:-win]
    ma = c[win-1:] / float(win)
    return ma[:len(env)]

# 전처리 + 유효성 + 피크 검출
def detect_peaks_ir(raw_chunk: np.ndarray, sr: int):
    """
    반환: valid(bool), peaks(np.ndarray), bpm(float), sig_det(np.ndarray), ac(np.ndarray), m(dict)
    - 전처리: DC 제거(2s MA) -> (선택) bandpass(0.5–8 Hz)로 검출
    - 유효성: 30<BPM<200, 피크수>8, 피크수>구간초/2
    """
    dc = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac = raw_chunk - dc
    sig_det = bandpass_sos(ac, sr) if USE_BANDPASS_FOR_DETECTION else ac

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

# AC 3x3 플롯
def plot_ac_grid_no_dots_with_tags(raw: np.ndarray, sr: int, chunk_len: int, title_prefix: str, save_prefix: str, group=9):
    chunks = to_chunks(raw, chunk_len)
    for i in range(0, len(chunks), group):
        sub = chunks[i:i+group]; n = len(sub)
        nrows = int(np.ceil(n / 3)) if n > 0 else 1
        fig, axs = plt.subplots(nrows, 3, figsize=(15, 2.4*nrows), sharex=False, sharey=True)
        axs = axs.flat if hasattr(axs, 'flat') else [axs]
        for j, ax in enumerate(axs):
            if j < n:
                raw_chunk = sub[j]
                valid, peaks, bpm, sig_det, ac, _ = detect_peaks_ir(raw_chunk, sr)
                t = np.arange(len(ac))/sr
                ax.plot(t, ac, linewidth=0.9)
                tag = f"{'valid' if valid else 'invalid'}"
                bpm_txt = f"{bpm:.1f} bpm" if np.isfinite(bpm) else "— bpm"
                ax.set_title(f"Chunk {i+j}  ({tag}, {bpm_txt})")
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_ylim(Y_MIN, Y_MAX)  # 🔒축 고정
            else:
                ax.axis('off')
        plt.suptitle(f"{title_prefix} {i}–{i+n-1}", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        sp = os.path.join(OUT_DIR, f"{save_prefix}_{i:03d}.png")
        plt.savefig(sp, dpi=200); plt.close()
        print(f"📦 저장: {sp}")

# 피처 추출
def extract_features_ir_chunk(raw_chunk: np.ndarray, sr: int):
    """
    유효 청크만 피처 계산:
    - PI, Amp_PeakMean, dPPG_max, PW50, RIIV_Power, BPM, SDNN, RMSSD
    """
    if len(raw_chunk) < CHUNK or np.all(np.isnan(raw_chunk)):
        return {'valid': False}

    valid, peaks, bpm, sig_det, ac, m = detect_peaks_ir(raw_chunk, sr)
    if not valid or len(peaks) == 0 or not np.isfinite(bpm):
        return {'valid': False}

    # DC / AC 다시 계산 (PI용)
    dc = moving_avg(raw_chunk.astype(float), int(DC_WIN_SEC * sr))
    ac_full = raw_chunk - dc

    # 1) PI = 100 * AC_RMS / mean(DC)
    dc_mean = float(np.nanmean(dc)) if np.isfinite(dc).any() else np.nan
    ac_rms  = float(np.sqrt(np.nanmean((ac_full - np.nanmean(ac_full))**2))) if np.isfinite(ac_full).any() else np.nan
    PI = 100.0 * ac_rms / dc_mean if (dc_mean and np.isfinite(dc_mean)) else np.nan

    # 2) Amp_PeakMean (검출 신호 기준)
    Amp_PeakMean = float(np.mean(sig_det[peaks])) if len(peaks) else np.nan

    # 3) dPPG_max
    d1 = np.diff(sig_det)
    dPPG_max = float(np.nanmax(d1)) if d1.size > 0 else np.nan

    # 4) PW50 (half-amplitude pulse width, 초)
    pw_list = []
    for i, pk in enumerate(peaks):
        left = peaks[i-1] if i > 0 else 0
        right = peaks[i+1] if i < len(peaks)-1 else len(sig_det)-1
        if right - left < 5:
            continue
        base_val = np.min(sig_det[left:right])
        peak_val = sig_det[pk]
        half = base_val + 0.5*(peak_val - base_val)
        # 좌/우 half 레벨 교차
        li = None
        for k in range(pk, left, -1):
            a, b = sig_det[k-1], sig_det[k]
            if (a <= half <= b) or (b <= half <= a): li = k; break
        ri = None
        for k in range(pk, right-1):
            a, b = sig_det[k], sig_det[k+1]
            if (a <= half <= b) or (b <= half <= a): ri = k; break
        if li is not None and ri is not None and ri > li:
            pw_list.append((ri - li) / sr)
    PW50 = float(np.nanmean(pw_list)) if pw_list else np.nan

    # 5) RIIV_Power(0.1–0.4 Hz) — 엔벨로프 기반
    env = signal_envelope(sig_det, sr)
    riiv = env - np.nanmean(env)
    RIIV_Power = bandpower_welch(riiv, sr, 0.1, 0.4)

    # 6~8) BPM/SDNN/RMSSD (HeartPy 결과)
    SDNN  = float(m.get('sdnn', np.nan)) if m.get('sdnn', None) is not None else np.nan
    RMSSD = float(m.get('rmssd', np.nan)) if m.get('rmssd', None) is not None else np.nan

    return {
        'valid': True,
        'PI': PI,
        'Amp_PeakMean': Amp_PeakMean,
        'dPPG_max': dPPG_max,
        'PW50': PW50,
        'RIIV_Power': RIIV_Power,
        'BPM': bpm,
        'SDNN': SDNN,
        'RMSSD': RMSSD,
    }

def extract_features_ir(chunks: list[np.ndarray], sr: int) -> pd.DataFrame:
    rows = []
    for ch in chunks:
        feat = extract_features_ir_chunk(ch, sr)
        if feat.get('valid', False):
            rows.append(feat)
    return pd.DataFrame(rows)

# 두 집단 비교 박스플롯
def compare_groups_boxplot(df_base: pd.DataFrame, df_hold: pd.DataFrame, title='IR'):
    feats = ['PI','Amp_PeakMean','dPPG_max','PW50','RIIV_Power','BPM','SDNN','RMSSD']

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, f in enumerate(feats):
        ax = axes[i]
        a = df_base[f].dropna().values
        b = df_hold[f].dropna().values

        # Welch t-test (표본 부족시 NA)
        if len(a) >= 2 and len(b) >= 2:
            _, p = ttest_ind(a, b, equal_var=False)
            label = '(significant)' if (p < ALPHA) else '(ns)'
        else:
            p = np.nan
            label = '(NA)'

        # 박스플롯
        bp = ax.boxplot([a, b], patch_artist=True, labels=['baseline','holding'])
        colors = ['tab:blue', 'tab:orange']
        for patch, c in zip(bp['boxes'], colors): patch.set_facecolor(c)
        for elem in ['medians','whiskers','caps']:
            for obj in bp[elem]: obj.set_color('black')

        ax.set_title(f'{f} {label}')
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[-1].axis('off')

    plt.suptitle(f'{title}: Feature Boxplots (baseline vs holding)', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    save_path = os.path.join(OUT_DIR, f'{title}_boxplots.png')
    plt.savefig(save_path, dpi=200); plt.close()
    print(f'저장: {save_path}')

# 실행
if __name__ == "__main__":
    raw_base = load_values(BASELINE_PATH)
    raw_hold = load_values(HOLDING_PATH)

    base_name = os.path.splitext(os.path.basename(BASELINE_PATH))[0]
    hold_name = os.path.splitext(os.path.basename(HOLDING_PATH))[0]

    # AC 3x3 플롯 (각각, 점 없음 + 타이틀에 valid/BPM)
    plot_ac_grid_no_dots_with_tags(
        raw_base, sr=SR, chunk_len=CHUNK,
        title_prefix=f"{base_name}: AC Chunks",
        save_prefix=f"{base_name}_ac_grid_nodots"
    )
    plot_ac_grid_no_dots_with_tags(
        raw_hold, sr=SR, chunk_len=CHUNK,
        title_prefix=f"{hold_name}: AC Chunks",
        save_prefix=f"{hold_name}_ac_grid_nodots"
    )

    # 피처 추출 (유효 청크만)
    df_base = extract_features_ir(to_chunks(raw_base, CHUNK), sr=SR)
    df_hold = extract_features_ir(to_chunks(raw_hold, CHUNK), sr=SR)

    # 라벨링 & CSV 저장
    df_base['label'] = 'baseline'
    df_hold['label'] = 'holding'
    df_base.to_csv(os.path.join(OUT_DIR, f"{base_name}_features_ir.csv"), index=False)
    df_hold.to_csv(os.path.join(OUT_DIR, f"{hold_name}_features_ir.csv"), index=False)
    print(f"유효 청크 수: baseline={len(df_base)}, holding={len(df_hold)}")

    # 두 집단 비교 박스플롯 저장
    compare_groups_boxplot(df_base, df_hold, title='IR')
