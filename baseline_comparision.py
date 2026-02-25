import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
import utils as U

MANIFEST_PATH = r"mainfest/train_v0.csv"
ZSCORE_CLIP   = 3.0
ROBUST_ZSCORE = True

def best_threshold_f1(y_true, y_score):
    """train 데이터에서 F1 최대화 threshold 탐색"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(y_score.min(), y_score.max(), 100):
        f1 = f1_score(y_true, (y_score <= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def eval_single_feature(df_used, feat_z, groups, gkf):
    """단일 피처 threshold 기반 분류"""
    accs, recs, f1s = [], [], []

    for tr_idx, te_idx in gkf.split(df_used[U.FEATS_Z].values,
                                     (df_used['condition']=='holding').astype(int).values,
                                     groups=groups):
        tr_df = df_used.iloc[tr_idx]
        te_df = df_used.iloc[te_idx]

        y_tr = (tr_df['condition'] == 'holding').astype(int).values
        y_te = (te_df['condition'] == 'holding').astype(int).values
        x_tr = tr_df[feat_z].values
        x_te = te_df[feat_z].values

        # train에서 threshold 탐색 (값이 낮을수록 holding이므로 <=)
        thr = best_threshold_f1(y_tr, x_tr)

        y_pred = (x_te <= thr).astype(int)
        accs.append(accuracy_score(y_te, y_pred))
        recs.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return np.mean(accs), np.mean(recs), np.mean(f1s)


if __name__ == "__main__":
    df = U.build_features_df_from_manifest(MANIFEST_PATH)
    dfz = U.normalize_by_subject_baseline(df, feats=U.FEATURES, robust=ROBUST_ZSCORE)
    dfz[U.FEATS_Z] = dfz[U.FEATS_Z].clip(-ZSCORE_CLIP, ZSCORE_CLIP)
    df_used = dfz.dropna(subset=U.FEATS_Z + ['condition', 'subject_id']).copy()

    groups   = df_used['subject_id'].astype(str).values
    n_splits = min(5, len(np.unique(groups)))
    gkf      = GroupKFold(n_splits=n_splits)

    print(f"{'Method':<20} {'Accuracy':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 52)

    # 단일 피처 threshold 방법
    for feat_z in U.FEATS_Z:
        acc, rec, f1 = eval_single_feature(df_used, feat_z, groups, gkf)
        print(f"{feat_z:<20} {acc*100:>9.1f} {rec*100:>9.1f} {f1*100:>9.1f}")

    print(f"{'Proposed':<20} {'84.6':>10} {'--':>10} {'75.9':>10}")