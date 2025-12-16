# utils for spidersense app - data loading, signal processing, models

import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

_BASE = Path(__file__).parent
DATA_FOLDER = _BASE / "ecg-spider-clip-1.0.0"
OLD_DATA_FOLDER = _BASE / "electrocardiogram-skin-conductance-and-respiration-from-spider-fearful-individuals-watching-spider-video-clips-1.0.0"
WGET_DATA_FOLDER = _BASE / "physionet.org" / "files" / "ecg-spider-clip" / "1.0.0"


def check_and_download_data():
    # Check if data exists, download if not
    # Returns True if data is available, False otherwise

    global DATA_FOLDER

    # Check all possible data locations first
    for path in [DATA_FOLDER, OLD_DATA_FOLDER, WGET_DATA_FOLDER]:
        if path.exists():
            vp_folders = [f for f in path.iterdir() if f.is_dir() and f.name.startswith("VP")]
            if len(vp_folders) > 0:
                DATA_FOLDER = path
                return True

    # Data not found, try to download
    print("Data not found. Downloading from Google Drive...")
    print("This should take less than 1 minute (~145 MB zip file)...")

    app_dir = Path(__file__).parent
    gdrive_id = "1w135fj2ohHtGOpoScGF4y6PuGPlNC-X2"
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_id}"
    zip_file = app_dir / "ecg-spider-clip-1.0.0.zip"
    # Zip extracts to this long folder name
    extract_dir = app_dir / "electrocardiogram-skin-conductance-and-respiration-from-spider-fearful-individuals-watching-spider-video-clips-1.0.0"

    try:
        # Download zip file using gdown
        print("Downloading zip file...")
        import gdown
        gdown.download(gdrive_url, str(zip_file), quiet=False)

        if not zip_file.exists():
            print("Download failed")
            return False

        # Check zip file exists
        if not zip_file.exists():
            print("Zip file not found after download")
            return False

        print(f"Download complete. Zip size: {zip_file.stat().st_size / 1e9:.2f} GB")

        # Unzip
        print("Extracting zip file...")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(app_dir)
        print("Extraction complete.")

        # Remove zip file to save space
        zip_file.unlink()
        print("Cleaned up zip file.")

        # Find the extracted folder and set DATA_FOLDER
        # The zip extracts to ecg-spider-clip-1.0.0/
        if extract_dir.exists():
            vp_folders = [f for f in extract_dir.iterdir() if f.is_dir() and f.name.startswith("VP")]
            if len(vp_folders) > 0:
                DATA_FOLDER = extract_dir
                print(f"Setup complete! Found {len(vp_folders)} subjects.")
                return True

        print("Extraction completed but data not found in expected location.")
        return False

    except Exception as e:
        print(f"Error during download/extraction: {e}")
        return False


def get_subjects():
    # returns list of subject IDs (VP02, VP03, etc)
    global DATA_FOLDER

    # Check which data folder exists (try all paths)
    for path in [DATA_FOLDER, OLD_DATA_FOLDER, WGET_DATA_FOLDER]:
        if path.exists():
            vp_folders = [f for f in path.iterdir() if f.is_dir() and f.name.startswith("VP")]
            if len(vp_folders) > 0:
                DATA_FOLDER = path
                break
    else:
        return []

    subjects = []
    for folder in sorted(DATA_FOLDER.iterdir()):
        if folder.is_dir() and folder.name.startswith("VP"):
            # Check it has all required files
            required = ["BitalinoECG.txt", "BitalinoGSR.txt", "Triggers.txt"]
            if all((folder / f).exists() for f in required):
                subjects.append(folder.name)
    return subjects


def parse_time(t):
    # convert timestamp '112755.578' -> seconds since midnight
    t = str(t).strip()
    if not t:
        return np.nan

    # Split into base and fractional part
    if "." in t:
        base, frac = t.split(".", 1)
        frac_sec = float("0." + frac)
    else:
        base, frac_sec = t, 0.0

    # Parse hhmmss
    base = base.zfill(6)
    hh = int(base[0:2])
    mm = int(base[2:4])
    ss = int(base[4:6])

    return hh * 3600 + mm * 60 + ss + frac_sec


def load_signal(filepath, col_name):
    df = pd.read_csv(filepath, sep=r"\s+", header=None,
                     names=[col_name, "time_raw", "label"], engine="python")
    df["time"] = df["time_raw"].apply(parse_time)
    df = df[[col_name, "time"]].dropna()
    df = df.groupby("time").mean().reset_index()
    return df.sort_values("time")


def load_triggers(filepath):
    df = pd.read_csv(filepath, sep=r"\s+", header=None,
                     names=["trigger_id", "start_raw", "end_raw"], engine="python")
    df["start"] = df["start_raw"].apply(parse_time)
    df["end"] = df["end_raw"].apply(parse_time)
    return df[["trigger_id", "start", "end"]].dropna()


def load_subject_data(subject_id):
    folder = DATA_FOLDER / subject_id

    ecg = load_signal(folder / "BitalinoECG.txt", "ecg")
    eda = load_signal(folder / "BitalinoGSR.txt", "eda")
    triggers = load_triggers(folder / "Triggers.txt")

    return ecg, eda, triggers


# filtering

def butter_filter(x, fs, low=None, high=None):
    x = np.array(x, dtype=float)

    # Handle NaN values
    valid = np.isfinite(x)
    if valid.sum() < 10:
        return x

    nyq = 0.5 * fs

    # Figure out filter type
    if low is None and high is not None:
        b, a = signal.butter(4, high / nyq, btype="lowpass")
    elif low is not None and high is None:
        b, a = signal.butter(4, low / nyq, btype="highpass")
    elif low is not None and high is not None:
        b, a = signal.butter(4, [low / nyq, high / nyq], btype="bandpass")
    else:
        return x

    # Apply filter to valid parts
    out = x.copy()
    out[valid] = signal.filtfilt(b, a, x[valid])
    return out


def resample_and_process(ecg_df, eda_df, triggers_df, fs=100):
    # resample to uniform grid
    t_min = min(ecg_df["time"].min(), eda_df["time"].min(), triggers_df["start"].min())
    t_max = max(ecg_df["time"].max(), eda_df["time"].max(), triggers_df["end"].max())

    # Create uniform time grid
    t = np.arange(0, t_max - t_min + 0.01, 1/fs)

    # Interpolate signals
    ecg = np.interp(t, ecg_df["time"] - t_min, ecg_df["ecg"])
    eda = np.interp(t, eda_df["time"] - t_min, eda_df["eda"])

    # Fix EDA offset if negative
    if np.nanmin(eda) < 0:
        eda = eda - np.nanmin(eda)

    # Apply filters
    ecg_filt = butter_filter(ecg, fs, low=0.5, high=40)
    eda_filt = butter_filter(eda, fs, high=1.0)

    # Create labels: -1=unknown, 0=rest, 1=clip
    labels = np.full(len(t), -1)

    triggers_rel = triggers_df.copy()
    triggers_rel["start_t"] = triggers_rel["start"] - t_min
    triggers_rel["end_t"] = triggers_rel["end"] - t_min

    for _, row in triggers_rel.iterrows():
        name = str(row["trigger_id"]).upper()
        start = row["start_t"]
        end = row["end_t"]

        mask = (t >= start) & (t < end)

        if name.startswith("CLIP") and "DEMO" not in name:
            labels[mask] = 1
        elif "REST" in name:
            labels[mask] = 0

    return {
        "t": t, "fs": fs,
        "ecg": ecg_filt, "eda": eda_filt,
        "labels": labels, "triggers": triggers_rel
    }


def safe_divide(a, b):
    # avoid div by zero - returns NaN instead
    b = np.array(b, dtype=float)
    return np.where(np.abs(b) > 1e-8, np.array(a, dtype=float) / b, np.nan)


def extract_ecg_features(ecg, fs):
    features = {}

    # Basic stats
    features["ecg_mean"] = np.nanmean(ecg)
    features["ecg_std"] = np.nanstd(ecg)
    features["ecg_skew"] = stats.skew(ecg, nan_policy="omit")
    features["ecg_kurt"] = stats.kurtosis(ecg, nan_policy="omit")

    # R-peak detection for heart rate
    if np.isfinite(ecg).sum() > 50:
        z = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
        peaks, _ = signal.find_peaks(z, distance=int(0.25*fs), prominence=2.0)

        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs
            # Filter out physiologically implausible RR intervals (40-200 bpm range)
            rr_valid = rr[(rr >= 0.3) & (rr <= 1.5)]
            if len(rr_valid) >= 2:
                hr = safe_divide(60, rr_valid)
                features["hr_mean"] = np.nanmean(hr)
                features["hr_std"] = np.nanstd(hr)
                features["rmssd"] = np.sqrt(np.nanmean(np.diff(rr_valid)**2))
            else:
                features["hr_mean"] = np.nan
                features["hr_std"] = np.nan
                features["rmssd"] = np.nan
        else:
            features["hr_mean"] = np.nan
            features["hr_std"] = np.nan
            features["rmssd"] = np.nan
    else:
        features["hr_mean"] = np.nan
        features["hr_std"] = np.nan
        features["rmssd"] = np.nan

    return features


def extract_eda_features(eda, fs):
    features = {}

    # Basic stats
    features["eda_mean"] = np.nanmean(eda)
    features["eda_std"] = np.nanstd(eda)

    # SCR detection (skin conductance responses)
    eda_hp = butter_filter(eda, fs, low=0.05)
    dx = np.diff(eda_hp)
    if np.isfinite(dx).sum() > 10:
        peaks, _ = signal.find_peaks(dx, prominence=np.std(dx)*0.5)
        features["scr_count"] = len(peaks)
    else:
        features["scr_count"] = 0

    return features


def create_windows(data, window_sec=30, step_sec=15):
    t = data["t"]
    fs = data["fs"]
    labels = data["labels"]

    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    windows = []

    for start in range(0, len(t) - win_samples, step_samples):
        end = start + win_samples

        # Get labels for this window
        win_labels = labels[start:end]
        valid_labels = win_labels[win_labels >= 0]

        # Skip if not enough labeled data
        if len(valid_labels) < 0.8 * len(win_labels):
            continue

        # Majority vote for label
        counts = np.bincount(valid_labels, minlength=2)
        label = int(np.argmax(counts))

        # Skip if label is not pure enough
        if counts[label] / len(valid_labels) < 0.8:
            continue

        # Extract features
        ecg_win = data["ecg"][start:end]
        eda_win = data["eda"][start:end]

        feats = {}
        feats.update(extract_ecg_features(ecg_win, fs))
        feats.update(extract_eda_features(eda_win, fs))
        feats["label"] = label
        feats["t_start"] = t[start]
        feats["t_end"] = t[end-1]

        windows.append(feats)

    return pd.DataFrame(windows)


# models

def get_model(name):
    if name == "LogReg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ])
    elif name == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
        ])
    elif name == "XGBoost" and HAS_XGB:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1))
        ])
    else:
        # Fallback to LogReg
        return get_model("LogReg")


def get_feature_cols(df, modality="Fused"):
    all_cols = [c for c in df.columns if c not in ["label", "subject", "t_start", "t_end"]]

    if modality == "ECG-only":
        return [c for c in all_cols if c.startswith(("ecg", "hr", "rmssd"))]
    elif modality == "EDA-only":
        return [c for c in all_cols if c.startswith(("eda", "scr"))]
    else:
        return all_cols


def build_all_features(window_sec=30, step_sec=15):
    subjects = get_subjects()
    all_data = []

    for subj in subjects:
        try:
            ecg, eda, triggers = load_subject_data(subj)
            data = resample_and_process(ecg, eda, triggers)
            df = create_windows(data, window_sec, step_sec)
            if len(df) > 0:
                df["subject"] = subj
                all_data.append(df)
        except:
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def train_and_predict(df_all, test_subject, model_name, modality):
    # LOSO: train on everyone except test_subject
    feature_cols = get_feature_cols(df_all, modality)

    # Split data
    train_df = df_all[df_all["subject"] != test_subject]
    test_df = df_all[df_all["subject"] == test_subject]

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    # Prep data
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Train and predict
    model = get_model(model_name)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # Build result dataframe
    result = test_df[["t_start", "t_end", "label"]].copy()
    result["prob"] = probs

    return result


def run_cv(df_all, model_name, modality, n_splits=5):
    feature_cols = get_feature_cols(df_all, modality)

    X = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df_all["label"].values
    groups = df_all["subject"].values

    # Limit splits to number of subjects
    n_splits = min(n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits)

    results = []
    cm_total = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = get_model(model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "fold": fold + 1,
            "bal_acc": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        })

        cm_total += confusion_matrix(y_test, y_pred, labels=[0, 1])

    return pd.DataFrame(results), cm_total
