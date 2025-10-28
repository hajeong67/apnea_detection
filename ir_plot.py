import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt

SR = 100                  # 샘플레이트(Hz)
CHUNK = 1200             # 12초
DC_WIN_SEC = 2.0         # DC(바탕선) 이동평균 윈도우(초) - 필요시 3.0
INPUT_PATH = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\dataset\holding_breath\ir\ir_hajeong_l_2.txt"
OUT_DIR = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\ir_plot\remove_dc"
SHOW_ZSCORE = True       # 시각화용 z-score(필터 아님) 추가 플롯

os.makedirs(OUT_DIR, exist_ok=True)

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

def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)

# DC 제거(이동평균)
def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    """same-length moving average with edge padding"""
    if win <= 1: return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    c = np.cumsum(xp, dtype=float)
    c[win:] = c[win:] - c[:-win]
    ma = c[win-1:] / float(win)
    return ma[:len(x)]

def remove_dc(raw: np.ndarray, sr: int, win_sec: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """returns (dc, ac) where ac = raw - dc"""
    win = max(1, int(win_sec * sr))
    dc = moving_avg(raw.astype(float), win)
    ac = raw - dc
    return dc, ac

# plot
def plot_raw_whole(raw: np.ndarray, title: str, save_name: str):
    plt.figure(figsize=(16, 4))
    plt.plot(raw, linewidth=1.0)
    plt.title(title)
    plt.xlabel('Sample'); plt.ylabel('Amplitude')
    plt.ylim(-3000, 4000)
    plt.grid(True, linestyle='--', alpha=0.4)
    sp = os.path.join(OUT_DIR, save_name)
    plt.savefig(sp, dpi=200); plt.close()
    print(f"저장: {sp}")

def plot_raw_vs_dc(raw: np.ndarray, dc: np.ndarray, title: str, save_name: str):
    plt.figure(figsize=(16, 4))
    plt.plot(raw, label='RAW', linewidth=1.0)
    plt.plot(dc,  label=f'DC (moving avg {DC_WIN_SEC:.1f}s)', linewidth=1.0)
    plt.title(title); plt.legend()
    plt.xlabel('Sample'); plt.ylabel('Amplitude')
    plt.ylim(-3000, 4000)
    plt.grid(True, linestyle='--', alpha=0.4)
    sp = os.path.join(OUT_DIR, save_name)
    plt.savefig(sp, dpi=200); plt.close()
    print(f"저장: {sp}")

def plot_grid(data: np.ndarray, chunk=300, group=9, title_prefix="Chunks", save_prefix="grid"):
    chunks = to_chunks(data, chunk)
    for i in range(0, len(chunks), group):
        sub = chunks[i:i+group]
        n = len(sub)
        nrows = int(np.ceil(n / 3)) if n > 0 else 1
        fig, axs = plt.subplots(nrows, 3, figsize=(15, 2.4*nrows), sharex=False, sharey=True)
        axs = axs.flat if hasattr(axs, 'flat') else [axs]
        for j, ax in enumerate(axs):
            if j < n:
                ax.plot(sub[j], linewidth=0.9)
                ax.set_title(f"Chunk {i+j}")
                ax.grid(True, linestyle='--', alpha=0.3)
            else:
                ax.axis('off')
        plt.suptitle(f"{title_prefix} {i}–{i+n-1}", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.ylim(-3000, 4000)
        sp = os.path.join(OUT_DIR, f"{save_prefix}_{i:03d}.png")
        plt.savefig(sp, dpi=200); plt.close()
        print(f"저장: {sp}")

if __name__ == "__main__":
    raw = load_values(INPUT_PATH)
    if raw.size < 10:
        raise SystemExit("데이터가 너무 짧거나 비었습니다.")

    base = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    # DC 제거
    dc, ac = remove_dc(raw, sr=SR, win_sec=DC_WIN_SEC)

    first_raw = raw[:CHUNK]
    first_dc  = dc[:CHUNK]
    first_ac  = ac[:CHUNK]

    # RAW vs DC(바탕선) 오버레이 (첫 청크)
    plot_raw_vs_dc(
        first_raw, first_dc,
        title=f"{base}: RAW vs DC (first {CHUNK/SR:.0f}s, MA {DC_WIN_SEC:.1f}s)",
        save_name=f"{base}_raw_vs_dc_first_chunk.png"
    )

    # AC(=RAW-DC) (첫 청크)
    plot_raw_whole(
        first_ac,
        title=f"{base}: AC = RAW - DC (first {CHUNK/SR:.0f}s)",
        save_name=f"{base}_ac_first_chunk.png"
    )

    # AC 12초 단위 그리드 (전체)
    plot_grid(
        ac, chunk=CHUNK, group=9,
        title_prefix=f"{base}: AC Chunks",
        save_prefix=f"{base}_ac_grid"
    )