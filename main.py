"""
=============================================================
  Sleep Stage Classification Pipeline (Sleep-EDF OLD FORMAT)
=============================================================

This script:

✔ Loads Sleep-EDF .rec files (even though MNE does not allow .rec)
✔ Converts them temporarily to .edf in RAM
✔ Extracts Pz-Oz or fallback EEG channels
✔ Reads OLD R&K hypnogram format (byte-encoded)
✔ Creates 30s epochs automatically (no timestamps)
✔ Preprocesses (bandpass + zscore)
✔ Computes DWT features (db4, 4 levels)
✔ Performs ANOVA feature selection
✔ Trains RUSBoost-style classifier (AdaBoost + undersampling)
✔ Runs 5-class, 4-class, 3-class, 2-class experiments
✔ Shows confusion matrices & reports
"""

import os
import glob
import tempfile
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import mne
import pywt
from scipy.signal import butter, filtfilt
import scipy.stats as stats

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


# ===========================================================
#                     CONFIG — EDIT THIS
# ===========================================================

DATA_DIR = r"D:\PDPU\Santosh Sir Sleep Reference Paper\sleep_stage_project\sleep-edf-database-1.0.0"

# ===========================================================
#                 OLD FORMAT HYPNOGRAM PARSER
# ===========================================================

def read_hyp_file_old_format(hyp_path):
    """
    OLD Sleep-EDF hypnogram format:
    Stores one byte per 30-second epoch.
    Valid stage byte values:
        0 = Wake
        1 = S1
        2 = S2
        3 = S3
        4 = S4
        5 = REM
        6 = Movement
    """

    with open(hyp_path, "rb") as f:
        data = f.read()

    stages = []

    for b in data:
        if b in [0,1,2,3,4,5,6]:
            stages.append(b)

    stages = np.array(stages, dtype=int)
    return stages

# ===========================================================
#           EDF LOADER (HANDLES .rec by renaming)
# ===========================================================

def load_edf_channel_mne(file_path, target="Pz-Oz"):
    """MNE refuses .rec. Fix = copy to temp .edf first."""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    tmp_path = tmp.name
    tmp.close()

    shutil.copy(file_path, tmp_path)

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    os.remove(tmp_path)

    ch_names = raw.ch_names

    idx = None
    for i, ch in enumerate(ch_names):
        if target.lower() in ch.lower():
            idx = i
            break

    if idx is None:
        fallbacks = ["PZOZ", "PZ", "OZ", "FPZ-CZ", "EEG Fpz-Cz"]
        for f in fallbacks:
            for i, ch in enumerate(ch_names):
                if f.lower() in ch.lower():
                    idx = i
                    break
            if idx is not None:
                break

    if idx is None:
        raise ValueError(f"No EEG channel found in {file_path}")

    fs = int(raw.info["sfreq"])
    data = raw.get_data()[idx]

    return data.astype(np.float32), fs


# ===========================================================
#               BUILD EPOCHS USING OLD FORMAT
# ===========================================================

def epoch_file_pair_old(edf_path, hyp_path, epoch_len_s=30):
    sig, fs = load_edf_channel_mne(edf_path)
    stages = read_hyp_file_old_format(hyp_path)

    samples_per_epoch = epoch_len_s * fs
    total_needed = samples_per_epoch * len(stages)

    if total_needed > len(sig):
        print("⚠ EDF shorter than hypnogram. Trimming hyp.")
        max_epochs = len(sig) // samples_per_epoch
        stages = stages[:max_epochs]

    X, y = [], []

    for i, st in enumerate(stages):
        s = i * samples_per_epoch
        e = s + samples_per_epoch
        X.append(sig[s:e])
        y.append(st)

    return np.array(X), np.array(y)


# ===========================================================
#                 PREPROCESSING
# ===========================================================

def preprocess_epochs(X, fs=100, low=0.5, high=40, amp_thresh=500):
    b, a = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype="band")
    X_out, mask = [], []

    for ep in X:
        ep_filt = filtfilt(b, a, ep)

        if np.max(np.abs(ep_filt)) > amp_thresh:
            mask.append(False)
            continue

        ep_norm = (ep_filt - ep_filt.mean()) / (ep_filt.std() + 1e-9)

        X_out.append(ep_norm)
        mask.append(True)

    return np.array(X_out), np.array(mask)


# ===========================================================
#            DWT FEATURE EXTRACTION
# ===========================================================

def epoch_dwt_features(epoch):
    coeffs = pywt.wavedec(epoch, "db4", level=4)
    feats = []

    for c in coeffs:
        c = np.asarray(c)
        p = np.abs(c) / (np.sum(np.abs(c)) + 1e-12)
        feats += [
            np.sum(c**2),                     # energy
            np.sqrt(np.mean(c**2)),           # RMS
            np.mean(c),                       # mean
            np.std(c),                        # std
            -np.sum(p * np.log2(p + 1e-12)),  # entropy
            stats.skew(c),
            stats.kurtosis(c)
        ]

    return np.array(feats)


# ===========================================================
#               LABEL MAPPERS
# ===========================================================

def map5(y):
    y = y.copy()
    y[y == 4] = 3
    return y

def map4(y):
    out = np.full_like(y, -1)
    out[y==0] = 0
    out[np.isin(y,[1,2])] = 1
    out[np.isin(y,[3,4])] = 2
    out[y==5] = 3
    return out

def map3(y):
    out = np.full_like(y, -1)
    out[y==0] = 0
    out[np.isin(y, [1,2,3,4])] = 1
    out[y==5] = 2
    return out

def map2(y):
    out = np.full_like(y, -1)
    out[y==0] = 0
    out[np.isin(y, [1,2,3,4,5])] = 1
    return out


# ===========================================================
#          RUSBoost-Style Classification Function
# ===========================================================

def run_experiment(X, y, groups, name):
    print("\n====================", name, "====================")

    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []

    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rus", RandomUnderSampler()),
            ("clf", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ))

        ])

        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[te])

        print(f"Fold {i} accuracy = {accuracy_score(y[te], y_pred):.3f}")

        all_true.extend(y[te])
        all_pred.extend(y_pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print("\nFINAL ACCURACY:", accuracy_score(all_true, all_pred))
    print(classification_report(all_true, all_pred, digits=3))

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(5,5))
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.show()


# ===========================================================
#                           MAIN
# ===========================================================

def main():

    rec_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.rec")))
    print("\nFound .rec files:", len(rec_files))

    all_X, all_y, all_subj = [], [], []

    for rec in rec_files:
        base = os.path.splitext(os.path.basename(rec))[0]
        hyp = rec.replace(".rec", ".hyp")

        print("\nProcessing:", base)

        Xs, ys = epoch_file_pair_old(rec, hyp)
        print("Epochs created:", len(Xs))

        all_X.append(Xs)
        all_y.append(ys)
        all_subj += [base]*len(ys)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subj = np.array(all_subj)

    print("\nTotal raw epochs:", len(X))

    # Preprocessing
    Xp, mask = preprocess_epochs(X)
    yp = y[mask]
    sp = subj[mask]

    print("After preprocessing:", len(Xp))

    # Feature extraction
    print("\nExtracting DWT features...")
    F = np.vstack([epoch_dwt_features(ep) for ep in tqdm(Xp)])

    # ================= Experiments =================

    # 5-class
    y5 = map5(yp)
    valid = np.isin(y5, [0,1,2,3,5])
    run_experiment(F[valid], y5[valid], sp[valid], "5-Class")

    # 4-class
    y4 = map4(yp)
    valid = y4>=0
    run_experiment(F[valid], y4[valid], sp[valid], "4-Class")

    # 3-class
    y3 = map3(yp)
    valid = y3>=0
    run_experiment(F[valid], y3[valid], sp[valid], "3-Class")

    # 2-class
    y2 = map2(yp)
    valid = y2>=0
    run_experiment(F[valid], y2[valid], sp[valid], "2-Class")


if __name__ == "__main__":
    main()
