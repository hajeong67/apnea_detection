import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
import utils as U

MANIFEST_PATH = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\paper_ir\manifest\train_v1.csv"
MODEL_OUT     = r"C:\Users\user\Desktop\ppg-transformer\ppg-transformer\paper_ir\model_version\ir_model_loso_v2.joblib"

# knobs
RATIO         = 1.0
ROBUST_ZSCORE = True
ZSCORE_CLIP   = 3.0
TUNE_STRATEGY = "fbeta"
F_BETA        = 1.0
TARGET_PREC   = 0.90
USE_POLY2     = True
LOGREG_C      = 0.1
RANDOM_STATE  = 42


def load_and_normalize(manifest_path):
    df = U.build_features_df_from_manifest(manifest_path)
    if df.empty:
        raise RuntimeError("No valid chunks built from manifest.")
    print("[BUILT FEATURES]", df.shape,
          "subjects:", df['subject_id'].nunique(),
          "trials:", df['trial_id'].nunique())

    dfz = U.normalize_by_subject_baseline(df, feats=U.FEATURES, robust=ROBUST_ZSCORE)
    if ZSCORE_CLIP is not None:
        dfz[U.FEATS_Z] = dfz[U.FEATS_Z].clip(-abs(ZSCORE_CLIP), abs(ZSCORE_CLIP))

    df_used = dfz.dropna(subset=U.FEATS_Z + ['condition', 'subject_id']).copy()
    if df_used.empty:
        raise RuntimeError("Empty after normalization/dropna")
    return df_used


def run_fold(clf, tr_df, te_df, groups, te_idx):
    X_tr = tr_df[U.FEATS_Z].values
    y_tr = (tr_df['condition'] == 'holding').astype(int).values
    X_te = te_df[U.FEATS_Z].values
    y_te = (te_df['condition'] == 'holding').astype(int).values

    clf.fit(X_tr, y_tr)

    tr_prob = clf.predict_proba(X_tr)[:, 1]
    thr, _ = U.pick_threshold(
        y_tr, tr_prob,
        subjects=tr_df['subject_id'].astype(str).values,
        conditions=tr_df['condition'].values,
        strategy=TUNE_STRATEGY,
        f_beta=F_BETA,
        target_prec=TARGET_PREC
    )

    te_prob = clf.predict_proba(X_te)[:, 1]
    acc, roc, f1, prec, rec, _ = U.eval_metrics(y_te, te_prob, thr)

    return thr, acc, roc, f1, prec, rec, te_prob, y_te


def print_fold_details(fold, thr, acc, roc, f1, te_prob, te_df, groups, te_idx, clf):
    print(f"[CV fold {fold}] thr={thr:.3f} | Acc={acc:.3f} | ROC={roc:.3f} | F1={f1:.3f} "
          f"| test_subjects={len(np.unique(groups[te_idx]))}")

    te_pred = (te_prob >= thr).astype(int)
    te_df2 = te_df.copy()
    te_df2['pred'] = te_pred
    te_df2['prob'] = te_prob
    print(te_df2.groupby(['subject_id', 'condition'])[['pred']].mean())

    coef = clf.named_steps['logreg'].coef_[0]
    for name, c in sorted(zip(U.FEATS_Z, coef), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:30s} {c:+.4f}")


def run_cv(df_used):
    X      = df_used[U.FEATS_Z].values
    y      = (df_used['condition'] == 'holding').astype(int).values
    groups = df_used['subject_id'].astype(str).values

    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need >=2 subjects for LOSO/GroupKFold.")
    gkf = GroupKFold(n_splits=n_splits)
    clf = U.make_classifier_pipeline(use_poly2=USE_POLY2, logreg_c=LOGREG_C, random_state=RANDOM_STATE)

    accs, rocs, f1s, precs, recs, fold_thrs = [], [], [], [], [], []
    fold_coefs = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        tr_df = U.undersample_baseline_per_subject(
            df_used.iloc[tr_idx].copy(), ratio=RATIO, random_state=RANDOM_STATE)
        te_df = df_used.iloc[te_idx].copy()

        thr, acc, roc, f1, prec, rec, te_prob, _ = run_fold(clf, tr_df, te_df, groups, te_idx)

        accs.append(acc); rocs.append(roc)
        f1s.append(f1);   precs.append(prec)
        recs.append(rec); fold_thrs.append(thr)

        # poly2 계수 추출
        coef_full = clf.named_steps['logreg'].coef_[0]
        if USE_POLY2:
            poly_names = clf.named_steps['poly'].get_feature_names_out(U.FEATS_Z)
            main_idx = [i for i, n in enumerate(poly_names) if n in U.FEATS_Z]
            coef_main = coef_full[main_idx]
        else:
            coef_main = coef_full
        fold_coefs.append(coef_main)

        print_fold_details(fold, thr, acc, roc, f1, te_prob, te_df, groups, te_idx, clf)

    thr_final = float(np.median(fold_thrs))
    print("\n[CV SUMMARY: subject-heldout]")
    print(f"Acc {np.mean(accs):.3f}±{np.std(accs):.3f} | "
          f"ROC {np.mean(rocs):.3f}±{np.std(rocs):.3f} | "
          f"F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f} | "
          f"Prec {np.mean(precs):.3f}±{np.std(precs):.3f} | "
          f"Rec {np.mean(recs):.3f}±{np.std(recs):.3f} | "
          f"thr≈{thr_final:.2f}")

    print_coef_summary(fold_coefs)
    return thr_final

def print_coef_summary(fold_coefs):
    coef_arr  = np.array(fold_coefs)   # (n_folds, n_feats)
    coef_mean = coef_arr.mean(axis=0)
    coef_std  = coef_arr.std(axis=0)
    odds      = np.exp(coef_mean)

    order = np.argsort(-np.abs(coef_mean))
    names_sorted = [U.FEATS_Z[i] for i in order]
    mean_sorted = coef_mean[order]
    std_sorted = coef_std[order]
    odds_sorted = odds[order]

    print("\n[Logistic Regression Coefficients (main effects)]")
    print(f"{'Feature':<20} {'Coef (mean±std)':>20} {'Odds Ratio':>12}")
    print("-" * 55)
    for name, mean, std, odd in zip(names_sorted, mean_sorted, std_sorted, odds_sorted):
        print(f"{name:<20} {mean:+.4f} ± {std:.4f}     {odd:.4f}")

def fit_final_and_save(df_used, thr_final):
    clf = U.make_classifier_pipeline(use_poly2=USE_POLY2, logreg_c=LOGREG_C, random_state=RANDOM_STATE)
    train_all = U.undersample_baseline_per_subject(df_used, ratio=RATIO, random_state=RANDOM_STATE)
    clf.fit(train_all[U.FEATS_Z].values,
            (train_all['condition'] == 'holding').astype(int).values)

    bundle = {
        'model': clf,
        'threshold': thr_final,
        'features': U.FEATS_Z,
        'settings': {
            'SR': U.SR, 'CHUNK': U.CHUNK, 'DC_WIN_SEC': U.DC_WIN_SEC,
            'USE_BANDPASS_FOR_DET': U.USE_BANDPASS_FOR_DET,
            'ROBUST_ZSCORE': ROBUST_ZSCORE, 'ZSCORE_CLIP': ZSCORE_CLIP,
            'RATIO': RATIO,
            'TUNE_STRATEGY': TUNE_STRATEGY, 'F_BETA': F_BETA, 'TARGET_PREC': TARGET_PREC,
            'USE_POLY2': USE_POLY2, 'LOGREG_C': LOGREG_C, 'RANDOM_STATE': RANDOM_STATE
        }
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"\nsaved: {MODEL_OUT}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    df_used   = load_and_normalize(MANIFEST_PATH)
    thr_final = run_cv(df_used)
    fit_final_and_save(df_used, thr_final)