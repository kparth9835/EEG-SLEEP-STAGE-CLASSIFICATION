import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pywt
import mne
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.ensemble import RUSBoostClassifier
import glob, os

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = ""   # path where rec and hyp files are stored

# -----------------------------
# LOAD REC FILE
# -----------------------------
def load_rec_signal(path):
    record = wfdb.rdrecord(path.replace(".rec",""))
    signal = record.p_signal[:, 0]  # Use 1st EEG channel
    fs = int(record.fs)
    return signal, fs

# -----------------------------
# LOAD HYPNOGRAM FILE
# -----------------------------
def load_hypnogram(path):
    stages = []
    with open(path, "r") as f:
        for line in f:
            if "Sleep stage" in line:
                stages.append(line.split()[-1])
    return stages

# Stage mapping
stage_map = {
    "W": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,   # merge deep sleep
    "R": 4
}

# -----------------------------
# EPOCH SIGNAL
# -----------------------------
def create_epochs(sig, fs, epoch_len=30):
    samples = fs * epoch_len
    n_epochs = len(sig) // samples
    sig = sig[:n_epochs * samples]
    return sig.reshape(n_epochs, samples)

# -----------------------------
# DWT FEATURE EXTRACTION
# -----------------------------
def extract_dwt_features(epoch):
    coeffs = pywt.wavedec(epoch, "db4", level=4)
    feat = []
    for c in coeffs:
        feat += [
            np.mean(c),
            np.std(c),
            skew(c),
            kurtosis(c),
            np.sum(c**2)    # Energy
        ]
    return np.array(feat)

# -----------------------------
# PROCESS ALL FILES
# -----------------------------
rec_files = sorted(glob.glob(DATA_PATH + "*.rec"))
hyp_files = sorted(glob.glob(DATA_PATH + "*.hyp"))

X_all, y_all = [], []

for rec_path, hyp_path in zip(rec_files, hyp_files):
    print("Processing:", rec_path)

    signal, fs = load_rec_signal(rec_path)

    # Resample to 100 Hz
    if fs != 100:
        signal = mne.filter.resample(signal, up=100, down=fs)
        fs = 100

    epochs = create_epochs(signal, fs)
    hyp_labels = load_hypnogram(hyp_path)

    labels = [stage_map.get(x, None) for x in hyp_labels]
    labels = [l for l in labels if l is not None]

    epochs = epochs[:len(labels)]
    labels = labels[:len(epochs)]

    feats = [extract_dwt_features(e) for e in epochs]

    X_all.extend(feats)
    y_all.extend(labels)

X = np.array(X_all)
y = np.array(y_all)

print("Final dataset shape:", X.shape, y.shape)

# -----------------------------
# MODEL TRAINING
# -----------------------------
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RUSBoostClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
