# paper_ir/model_eval_external.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import utils as U

MODEL_PATH    = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\paper_ir\model_version\ir_model_loso_v2.joblib"
TEST_MANIFEST = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\paper_ir\manifest\test_v1.csv"
OUT_DIR       = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\paper_ir\model_output\eval_external_loso\v2"


def load_model(model_path):
    bundle  = joblib.load(model_path)
    clf     = bundle['model']
    thr     = float(bundle['threshold'])
    feats_z = bundle['features']
    feats   = [f.replace('_z', '') for f in feats_z]
    robust  = bool(bundle.get('settings', {}).get('ROBUST_ZSCORE', True))
    clipv   = bundle.get('settings', {}).get('ZSCORE_CLIP', None)
    return clf, thr, feats_z, feats, robust, clipv


def load_and_normalize_test(manifest_path, feats, feats_z, robust, clipv):
    df = U.build_features_df_from_manifest(manifest_path)
    if df.empty:
        raise RuntimeError("No valid chunks in test manifest.")

    dfz = U.normalize_by_subject_baseline(df, feats=feats, robust=robust)
    if clipv is not None:
        dfz[feats_z] = dfz[feats_z].clip(-abs(clipv), abs(clipv))

    df_used = dfz.dropna(subset=feats_z + ['condition']).copy()
    if df_used.empty:
        raise RuntimeError("Empty after zscore/dropna on test.")
    return df_used


def evaluate(clf, thr, df_used, feats_z):
    X = df_used[feats_z].values
    y = (df_used['condition'] == 'holding').astype(int).values

    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= thr).astype(int)

    acc  = accuracy_score(y, pred)
    roc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    f1   = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)

    print("[OVERALL | chunk-level]")
    print(f"Acc {acc:.3f} | ROC {roc:.3f} | F1 {f1:.3f} | "
          f"Prec {prec:.3f} | Rec {rec:.3f} (thr={thr:.3f})")

    return prob, pred, y


def save_results(df_used, prob, pred, y, out_dir):
    df_used = df_used.copy()
    df_used['y_true']    = y
    df_used['p_holding'] = prob
    df_used['y_pred']    = pred

    subj = (df_used.groupby('subject_id')
                   .apply(lambda g: pd.Series({
                       'N':    len(g),
                       'base': int((g['condition'] == 'baseline').sum()),
                       'hold': int((g['condition'] == 'holding').sum()),
                       'acc':  accuracy_score(g['y_true'], g['y_pred']),
                       'roc':  roc_auc_score(g['y_true'], g['p_holding']) if g['y_true'].nunique() > 1 else np.nan,
                       'f1':   f1_score(g['y_true'], g['y_pred'], zero_division=0),
                       'prec': precision_score(g['y_true'], g['y_pred'], zero_division=0),
                       'rec':  recall_score(g['y_true'], g['y_pred'], zero_division=0),
                   })))

    subj.to_csv(os.path.join(out_dir, "subject_summary.csv"))
    df_used.to_csv(os.path.join(out_dir, "detail.csv"), index=False)
    print(f"\n[saved] subject_summary.csv, detail.csv → {out_dir}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    clf, thr, feats_z, feats, robust, clipv = load_model(MODEL_PATH)
    df_used = load_and_normalize_test(TEST_MANIFEST, feats, feats_z, robust, clipv)
    prob, pred, y = evaluate(clf, thr, df_used, feats_z)
    save_results(df_used, prob, pred, y, OUT_DIR)