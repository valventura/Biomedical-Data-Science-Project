# SpiderSense: Anxiety Prediction from Wearable Biosignals  
**Capstone Project – Biomedical Data Science EN.585.771**  
**Valeria Ventura Subirachs**

**Project Video Link**
https://github.com/valventura/Biomedical-Data-Science-Project/blob/main/Biomedical%20Data%20Science%20Capstone%20VIDEO.mp4

---

## How the App works:

SpiderSense is a Streamlit web app that analyzes physiological signals and predicts whether a participant is watching a fear-inducing spider video clip or in a rest/neutral period.

- Task: **binary classification** (video clip vs rest)  
- Windowing: **30-second windows** with **15-second overlap** (default)  
- Held-out demo: trains on all other subjects and predicts on the selected subject (leave-one-subject-out style)

---

## The Data

The data comes from PhysioNet. Participants watched alternating spider clips and rest periods while biosignals were recorded.

- Signals used: **ECG** (electrocardiogram) and **EDA/GSR** (electrodermal activity / galvanic skin response)
- Nominal sampling: **100 Hz** (resampled to a uniform grid)  
- Download size: **~145 MB** (extracts to **~1.1 GB**)  
- Usable subjects in this release/app: **57**

---

## Auto-download 

If the dataset folder is not found locally, the app will automatically download and extract a zipped copy of the dataset from a personal Google Drive mirror using `gdown` (typically ~1 minute).

This Google Drive download is provided for convenience only. The authoritative source of the dataset is PhysioNet.

**Google Drive mirror:**  
https://drive.google.com/uc?id=1w135fj2ohHtGOpoScGF4y6PuGPlNC-X2

**Original dataset:**  
https://physionet.org/content/ecg-spider-clip/1.0.0/

---

## Setup

1) Install Python 3.8+

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Run the app:
```bash
streamlit run app.py
```

4) Open: http://localhost:8501

---

## How to Use

### Landing page (sidebar controls)
- **Subject**: choose a participant (e.g. VP02)
- **Window / Step**: how the signals are segmented (default 30s / 15s)
- **Threshold**: probability cutoff for predicting clip
- **Model**: Logistic Regression, RandomForest, or XGBoost
- **Modality**: ECG-only, EDA-only, or Fused (ECG+EDA)

### Pages
1. **Explore Subject** – view filtered signals + trigger overlays  
2. **Predict Timeline** – see predicted **P(clip)** over time with interactive threshold  
3. **Model Evaluation** – subject-wise cross-validated performance across the dataset

---

## Threshold 

The model outputs **P(clip)** for each window.

- If **P(clip) ≥ threshold → predict clip**
- If **P(clip) < threshold → predict rest**

Lower thresholds increase clip sensitivity (more detections, more false positives). Higher thresholds reduce false positives but may miss some clips.

---

## Signal Processing

All signal preprocessing, resampling, and feature extraction steps are implemented in **`utils.py`**.

- Raw signals contain irregular timestamps and are first **aligned and resampled to a uniform 100 Hz time grid**
- **ECG** is band-pass filtered (**0.5–40 Hz**) to remove baseline drift and high-frequency noise
- **EDA** is low-pass filtered (**~1 Hz**) and offset-corrected if negative values are present
- Signals are segmented into windows, and labels require **≥ 80% label purity** to avoid transitions

### ECG feature computation (per window)

1. **R-peaks** are detected in the filtered ECG signal using a peak-finding algorithm with a minimum distance constraint to enforce physiologically plausible heart rates  
2. **RR intervals** are computed as time differences between consecutive R-peaks  
3. **Heart rate** is computed as `60 / RR`  
4. **Heart rate variability (HRV)** is computed using **RMSSD**, calculated from successive RR interval differences  
5. Additional waveform statistics (mean, std, skewness, kurtosis) are computed

### EDA feature computation (per window)

1. EDA is low-pass filtered to remove high-frequency noise  
2. A high-pass filtered version of EDA is computed to emphasize rapid changes  
3. The **first derivative** of the high-pass filtered EDA is taken  
4. Prominent peaks in the derivative are detected and counted  
5. The resulting count is stored as **`scr_count`**, a lightweight **SCR proxy** that approximates phasic skin conductance responses

---

## Features Extracted (per window)

**ECG**
- Mean, std, skewness, kurtosis
- Heart rate (mean, std)
- RMSSD (HRV)

**EDA**
- Mean and std of skin conductance
- SCR proxy (`scr_count`)

---

## Model Performance

**5-fold subject-wise GroupKFold cross-validation** is used for the main evaluation.

Typical results (will vary slightly with windowing/modality):

| Model | Balanced Accuracy | ROC-AUC (approx.) |
|------|-------------------|------------------|
| LogReg | ~0.60 | ~0.65 |
| RandomForest | ~0.55–0.62 | ~0.61–0.66 |
| XGBoost | ~0.57–0.65 | ~0.63–0.66 |

*Single-subject ROC curves can be unstable due to limited windows and class imbalance; cross-validated results are the primary metric.*

---

## Files

```text
app.py                    
utils.py                 
pages/1_Explore_Subject.py
pages/2_Predict_Timeline.py
pages/3_Model_Evaluation.py
requirements.txt
README.md
```

---

## Requirements

- streamlit
- pandas
- numpy
- scipy
- scikit-learn
- plotly
- xgboost 
- gdown

---

## Limitations

- Labels represent experimental condition (video clip vs rest), not self-reported anxiety
- Wearable biosignals are noisy; feature extraction and SCR detection are simplified baselines
- This application is intended for educational and research purposes only

---

## Citations

- Ihmig, F. R., Gogeascoechea, A., Schäfer, S., Lass-Hennemann, J., & Michael, T. (2020).  
  *Electrocardiogram, skin conductance and respiration from spider-fearful individuals watching spider video clips* (version 1.0.0). PhysioNet. RRID:SCR_007345.  
  https://doi.org/10.13026/sq6q-zg04

Original publication:

- Ihmig, F. R., Gogeascoechea, A., Neurohr-Parakenings, F., Schäfer, S. K., Lass-Hennemann, J., & Michael, T. (2020).  
  *On-line anxiety level detection from biosignals: machine learning based on a randomized controlled trial with spider-fearful individuals*. PLoS ONE.

Standard PhysioNet citation:

- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., et al. (2000).  
  *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*.  
  Circulation [Online], 101(23), e215–e220. RRID:SCR_007345.
