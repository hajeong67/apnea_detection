"""
PI, RMSSD만 사용한 baseline vs holding 분류 (TEST-ONLY)
- 모델: 저장된 joblib 번들을 로드하여 사용(학습 없음)
- 정규화: 사람별 baseline μ, σ로 z-score (baseline/holding 동일 변환)
- 평가: 학습과 분리된 입력 지원
    1) 단일 외부 pair (EVAL_BASELINE_CSV/EVAL_HOLDING_CSV/EVAL_SUBJECT_ID)
    2) 외부 평가 manifest_v0.csv (EVAL_MANIFEST_PATH)
- 시각화: 외부 manifest 기준 오버레이/히스토그램/산점도/박스/바이올린 플롯
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix
)

# ===================== 사용자 설정 =====================
FEATURES = ['PI', 'RMSSD']                # 사용할 피처
FEATS_Z  = [f'{f}_z' for f in FEATURES]

# 이미 학습되어 저장된 모델 경로
MODEL_OUT = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\model_version\ir_model_v1.joblib"

# 외부 평가용 manifest 및 리포트 출력 폴더
EVAL_MANIFEST_PATH    = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\dataset\manifest_test_v1.csv"
EVAL_MANIFEST_OUT_DIR = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\neurokit2\output\model_report\v1"

# (선택) 단일 pair 평가를 환경변수로 받을 수 있음
# EVAL_SUBJECT_ID, EVAL_BASELINE_CSV, EVAL_HOLDING_CSV
# ======================================================

@dataclass
class SubjectPair:
    subject_id: str
    baseline_csv: str
    holding_csv: str

def load_manifest(manifest_path: str) -> List[SubjectPair]:
    m = pd.read_csv(manifest_path)
    required = {'subject_id', 'baseline_csv', 'holding_csv'}
    if not required.issubset(m.columns):
        raise ValueError(f"manifest.csv에 필요한 열이 없습니다: {sorted(required)}")
    pairs = []
    for _, r in m.iterrows():
        pairs.append(SubjectPair(
            subject_id=str(r['subject_id']),
            baseline_csv=str(r['baseline_csv']),
            holding_csv=str(r['holding_csv'])
        ))
    return pairs

def _load_pair_to_df(subject_id: str, baseline_csv: str, holding_csv: str) -> pd.DataFrame:
    if not os.path.exists(baseline_csv):
        raise FileNotFoundError(f"baseline_csv 없음: {baseline_csv}")
    if not os.path.exists(holding_csv):
        raise FileNotFoundError(f"holding_csv 없음: {holding_csv}")
    df_b = pd.read_csv(baseline_csv).assign(subject_id=subject_id, condition='baseline')
    df_h = pd.read_csv(holding_csv).assign(subject_id=subject_id, condition='holding')
    use_cols = ['subject_id','condition'] + [c for c in FEATURES if c in df_b.columns and c in df_h.columns]
    missing = [c for c in FEATURES if c not in use_cols]
    if missing:
        raise ValueError(f"{subject_id}: 필요한 피처 없음: {missing}")
    return pd.concat([df_b[use_cols], df_h[use_cols]], ignore_index=True)

def normalize_by_subject_baseline(df: pd.DataFrame, feats=FEATURES, eps=1e-6) -> pd.DataFrame:
    # 각 subject baseline으로 μ, σ 계산 → baseline/holding 모두 z변환
    stats = (df[df['condition']=='baseline']
             .groupby('subject_id')[feats]
             .agg(['mean','std']))
    stats.columns = [f'{a}_{b}' for a,b in stats.columns]
    out = df.merge(stats, left_on='subject_id', right_index=True, how='left')
    for f in feats:
        mu = out[f'{f}_mean']
        sd = out[f'{f}_std'].replace(0, np.nan)
        out[f'{f}_z'] = (out[f] - mu) / (sd.fillna(eps))
    return out

def unpack_model_bundle(obj):
    """
    obj가 dict(bundle) 또는 Pipeline일 수 있으니 안전하게 풀어준다.
    반환: (clf, threshold)
    """
    # joblib에서 {'bundle': bundle, 'features': ...}로 저장된 경우
    if isinstance(obj, dict) and 'bundle' in obj:
        obj = obj['bundle']
    # 표준 bundle: {'model': clf, 'threshold': float}
    if isinstance(obj, dict) and 'model' in obj and 'threshold' in obj:
        return obj['model'], float(obj['threshold'])
    # 과거 버전: 그냥 Pipeline만 있는 경우 -> 기본 임계치 0.5
    if hasattr(obj, 'predict_proba'):
        return obj, 0.5
    raise TypeError("지원하지 않는 모델 객체 형식입니다. (bundle dict 또는 sklearn Pipeline이어야 함)")

def subject_level_aggregate(y_true, y_prob, subjects, conditions, thr):
    """
    (subject_id, condition) 단위로 확률 평균 → 임계치로 라벨링
    반환: (y_true_s, y_pred_s, df_group)
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob,
        'subject': subjects,
        'cond': conditions
    })
    g = (df.groupby(['subject','cond'])
           .agg(y_true=('y_true','mean'),   # 그룹 내 동일 라벨이라 mean==그 라벨
                y_prob=('y_prob','mean'))
           .reset_index())
    y_true_s = g['y_true'].astype(int).values
    y_pred_s = (g['y_prob'] >= thr).astype(int).values
    return y_true_s, y_pred_s, g

def save_confusion_matrix_plot(y_true, y_pred, path_png,
                               labels=('baseline','holding'),
                               normalize=False, title=None, dpi=200):
    """
    혼동행렬을 이미지로 저장(plt.imshow). normalize=True면 행 정규화(%).
    """
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_plot = cm.astype(float)
    fmt = 'd'
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        cm_plot = cm_plot / row_sum
        fmt = '.2f'

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels([f'pred_{labels[0]}', f'pred_{labels[1]}'])
    ax.set_yticklabels([f'true_{labels[0]}', f'true_{labels[1]}'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')

    # 셀에 값 표기
    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.5
    for i in range(2):
        for j in range(2):
            text_val = f"{cm_plot[i, j]:.2f}" if fmt == '.2f' else f"{int(cm_plot[i, j])}"
            ax.text(j, i, text_val,
                    ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=12)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=dpi)
    plt.close()

def evaluate_external_pair(model_bundle,
                           subject_id: str,
                           baseline_csv: str,
                           holding_csv: str,
                           report_out: Optional[str] = None) -> dict:
    """학습과 분리된 외부 단일 pair 평가 + 혼동행렬 PNG 저장."""
    clf, thr = unpack_model_bundle(model_bundle)

    df_sub = _load_pair_to_df(subject_id, baseline_csv, holding_csv)
    df_norm = normalize_by_subject_baseline(df_sub, feats=FEATURES)

    X = df_norm[FEATS_Z].values
    y = (df_norm['condition'] == 'holding').astype(int).values

    y_proba = clf.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= thr).astype(int)

    acc = accuracy_score(y, y_pred)
    roc = roc_auc_score(y, y_proba)
    f1  = f1_score(y, y_pred)
    print(f"[EVAL external pair {subject_id}] Acc {acc:.3f} | ROC {roc:.3f} | F1 {f1:.3f} "
          f"(thr={thr:.2f}, N={len(y)}, base={(y==0).sum()}, hold={(y==1).sum()})")

    if report_out:
        base, _ = os.path.splitext(report_out)
        os.makedirs(os.path.dirname(report_out), exist_ok=True)

        # 리포트(청크별) 저장
        out = df_norm.copy()
        out['y_true'] = y
        out['p_holding'] = y_proba
        out['y_pred'] = y_pred
        out.to_csv(report_out, index=False)

        # 혼동행렬(청크 단위) PNG
        save_confusion_matrix_plot(
            y_true=y, y_pred=y_pred,
            path_png=base + "_cm_chunk.png",
            labels=('baseline','holding'),
            normalize=False,
            title=f"{subject_id} - Confusion Matrix (Chunk)"
        )
        save_confusion_matrix_plot(
            y_true=y, y_pred=y_pred,
            path_png=base + "_cm_chunk_norm.png",
            labels=('baseline','holding'),
            normalize=True,
            title=f"{subject_id} - Confusion Matrix (Chunk, row-normalized)"
        )
        print(f"🧾 저장: {report_out}, "
              f"{base+'_cm_chunk.png'}, {base+'_cm_chunk_norm.png'}")

    return {'acc': acc, 'roc': roc, 'f1': f1}

def evaluate_external_manifest(model_bundle,
                               manifest_path: str,
                               out_dir: Optional[str] = None) -> pd.DataFrame:
    """외부 평가 manifest_v0.csv: 다수 피험자 일괄 평가 + 전체 혼동행렬 PNG 저장."""
    clf, thr = unpack_model_bundle(model_bundle)
    pairs = load_manifest(manifest_path)

    rows = []
    y_all_true, y_all_pred, y_all_proba = [], [], []
    yS_true_all, yS_pred_all = [], []

    for p in pairs:
        rep_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            rep_path = os.path.join(out_dir, f"eval_{p.subject_id}.csv")
        m = evaluate_external_pair(model_bundle, p.subject_id, p.baseline_csv, p.holding_csv, report_out=rep_path)
        rows.append({'subject_id': p.subject_id, **m})

        # 전체 혼동행렬용 누적
        df_sub = _load_pair_to_df(p.subject_id, p.baseline_csv, p.holding_csv)
        df_norm = normalize_by_subject_baseline(df_sub, feats=FEATURES)
        X = df_norm[FEATS_Z].values
        y = (df_norm['condition']=='holding').astype(int).values
        y_proba = clf.predict_proba(X)[:, 1]
        y_pred  = (y_proba >= thr).astype(int)

        y_all_true.append(y); y_all_pred.append(y_pred)

        # 사람 단위 집계(평균 확률 → 임계치)
        yS_true, yS_pred, _ = subject_level_aggregate(
            y_true=y,
            y_prob=y_proba,
            subjects=df_norm['subject_id'].values,
            conditions=df_norm['condition'].values,
            thr=thr
        )
        yS_true_all.append(yS_true); yS_pred_all.append(yS_pred)

    df_summary = pd.DataFrame(rows)
    if out_dir:
        df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        y_all_true = np.concatenate(y_all_true)
        y_all_pred = np.concatenate(y_all_pred)
        save_confusion_matrix_plot(
            y_true=y_all_true, y_pred=y_all_pred,
            path_png=os.path.join(out_dir, "confusion_matrix_chunk.png"),
            labels=('baseline','holding'),
            normalize=False,
            title="All Subjects - Confusion Matrix (Chunk)"
        )

        yS_true_all = np.concatenate(yS_true_all)
        yS_pred_all = np.concatenate(yS_pred_all)
        save_confusion_matrix_plot(
            y_true=yS_true_all, y_pred=yS_pred_all,
            path_png=os.path.join(out_dir, "confusion_matrix_subject.png"),
            labels=('baseline','holding'),
            normalize=False,
            title="All Subjects - Confusion Matrix (Subject agg.)"
        )

        print("📊 외부 평가 요약/혼동행렬 PNG 저장 완료")

    return df_summary

# ========= 시각화 유틸(외부 manifest 기준) =========
def load_all_df_from_manifest(manifest_path: str) -> pd.DataFrame:
    pairs = load_manifest(manifest_path)
    frames = [ _load_pair_to_df(p.subject_id, p.baseline_csv, p.holding_csv) for p in pairs ]
    return pd.concat(frames, ignore_index=True)

def add_chunk_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['chunk_idx'] = df.groupby(['subject_id', 'condition']).cumcount()
    return df

def plot_train_overlays_lines(df_norm: pd.DataFrame, out_dir: str, feats=('PI_z', 'RMSSD_z')):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(feats), figsize=(6*len(feats), 4), dpi=200, constrained_layout=True)
    if len(feats) == 1:
        axes = [axes]
    for ax, f in zip(axes, feats):
        for (sid, cond), g in df_norm.groupby(['subject_id', 'condition']):
            color = 'tab:blue' if cond == 'baseline' else 'tab:orange'
            gg = g.dropna(subset=[f, 'chunk_idx'])
            if gg.empty: continue
            ax.plot(gg['chunk_idx'].values, gg[f].values, color=color, alpha=0.25, lw=1.0)
        ax.set_title(f"Overlay lines: {f}")
        ax.set_xlabel("chunk index")
        ax.set_ylabel(f)
        ax.grid(True, ls='--', alpha=0.3)
    path = os.path.join(out_dir, "train_overlays_lines.png")
    fig.savefig(path); plt.close(fig)
    print(f"🖼 저장: {path}")

def plot_train_overlays_hist(df_norm: pd.DataFrame, out_dir: str, feats=('PI_z', 'RMSSD_z'), bins=40):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(feats), figsize=(6*len(feats), 4), dpi=200, constrained_layout=True)
    if len(feats) == 1:
        axes = [axes]
    for ax, f in zip(axes, feats):
        a = df_norm.loc[df_norm['condition']=='baseline', f].dropna().values
        b = df_norm.loc[df_norm['condition']=='holding',  f].dropna().values
        ax.hist(a, bins=bins, alpha=0.5, label='baseline', color='tab:blue', density=True)
        ax.hist(b, bins=bins, alpha=0.5, label='holding',  color='tab:orange', density=True)
        ax.set_title(f"Histogram: {f}")
        ax.set_xlabel(f); ax.set_ylabel("density")
        ax.legend(); ax.grid(True, ls='--', alpha=0.3)
    path = os.path.join(out_dir, "train_overlays_hist.png")
    fig.savefig(path); plt.close(fig)
    print(f"🖼 저장: {path}")

def plot_train_boxplots(df_norm: pd.DataFrame, out_dir: str, feats=('PI_z', 'RMSSD_z')):
    os.makedirs(out_dir, exist_ok=True)
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4), dpi=200, constrained_layout=True)
    if n == 1: axes = [axes]
    for ax, f in zip(axes, feats):
        a = df_norm.loc[df_norm['condition']=='baseline', f].dropna().values
        b = df_norm.loc[df_norm['condition']=='holding',  f].dropna().values
        bp = ax.boxplot([a, b],
                        labels=['baseline', 'holding breath'],
                        showfliers=False, patch_artist=True, widths=0.6)
        colors = ['#4C78A8', '#F58518']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(1.0)
        for median in bp['medians']:
            median.set_color('black'); median.set_linewidth(1.5)
        ax.set_title(f'Boxplot: {f}'); ax.set_ylabel(f)
        ax.grid(True, linestyle='--', alpha=0.3)
    path = os.path.join(out_dir, "train_boxplots.png")
    fig.savefig(path); plt.close(fig)
    print(f"🖼 저장: {path}")

# ======================= MAIN (TEST ONLY) =======================
if __name__ == "__main__":
    # 1) 저장된 모델 로드
    if not os.path.exists(MODEL_OUT):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_OUT}")
    loaded = joblib.load(MODEL_OUT)
    print(f"✅ 모델 로드: {MODEL_OUT}")

    # 2) (옵션) 단일 pair 평가: 환경변수로 지정 시 수행
    env_sid = os.getenv("EVAL_SUBJECT_ID")
    env_base = os.getenv("EVAL_BASELINE_CSV")
    env_hold = os.getenv("EVAL_HOLDING_CSV")
    if env_sid and env_base and env_hold:
        pair_out = None
        if EVAL_MANIFEST_OUT_DIR:
            os.makedirs(EVAL_MANIFEST_OUT_DIR, exist_ok=True)
            pair_out = os.path.join(EVAL_MANIFEST_OUT_DIR, f"eval_{env_sid}.csv")
        evaluate_external_pair(loaded, env_sid, env_base, env_hold, report_out=pair_out)

    # 3) 외부 manifest 일괄 평가 + 혼동행렬 저장
    if EVAL_MANIFEST_PATH and os.path.exists(EVAL_MANIFEST_PATH):
        evaluate_external_manifest(loaded,
                                   manifest_path=EVAL_MANIFEST_PATH,
                                   out_dir=EVAL_MANIFEST_OUT_DIR)

        # 4) 시각화(외부 manifest 기준)
        df_all = load_all_df_from_manifest(EVAL_MANIFEST_PATH)
        df_norm = normalize_by_subject_baseline(df_all, feats=FEATURES)
        df_norm = add_chunk_index(df_norm)

        feats_z = [f'{f}_z' for f in FEATURES]
        plot_train_overlays_lines(df_norm, out_dir=EVAL_MANIFEST_OUT_DIR, feats=feats_z)
        plot_train_overlays_hist(df_norm,  out_dir=EVAL_MANIFEST_OUT_DIR, feats=feats_z, bins=40)
        plot_train_boxplots(df_norm,      out_dir=EVAL_MANIFEST_OUT_DIR, feats=feats_z)

    else:
        print("⚠️ EVAL_MANIFEST_PATH가 비어있거나 파일이 없습니다. manifest 평가/시각화를 건너뜁니다.")
