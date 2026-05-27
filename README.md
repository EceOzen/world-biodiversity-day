# 🐦 Listening to Extinction
### A Machine Learning Analysis of Bird Sound Recordings Across Turkey

> *Built on World Biodiversity Day 2026 — "Acting Locally for Global Impact"*  
> *Code Beyond the Earth × IDB 2026*

---

## What is this?

On May 22, 2026 — International Day for Biological Diversity — I asked a simple but unsettling question:

**Are Turkey's birds moving? And if so, where?**

This repository contains the full three-part analysis pipeline: from raw API calls to geographic shift detection, acoustic fingerprinting, machine learning classification, and acoustic drift over time. The data source is [Xeno-canto](https://xeno-canto.org), a global crowdsourced archive of wildlife sound recordings. Every recording comes with GPS coordinates and a timestamp — which means it's not just an audio library. It's a spatial biodiversity dataset hiding in plain sight.

---

## The Dataset

| | |
|---|---|
| **Source** | Xeno-canto API v3 |
| **Coverage** | Turkey (lat: 36–42.5°N, lon: 26–45°E) |
| **Recordings** | 2,303 |
| **Species** | 256 |
| **Time range** | 1993–2026 |

### 9 Focus Species

| Category | Turkish Name | Scientific Name |
|---|---|---|
| 🔴 Losers | Kınalı Keklik | *Alectoris chukar* |
| 🔴 Losers | Bıldırcın | *Coturnix coturnix* |
| 🔴 Losers | Büyük Toy | *Otis tarda* |
| 🟢 Winners | Saksağan | *Pica pica* |
| 🟢 Winners | Ebabil | *Apus apus* |
| 🟢 Winners | Kızıl Şahin | *Buteo buteo* |
| ⚫ Silent Places | Kara Akbaba | *Aegypius monachus* |
| ⚫ Silent Places | Bozkır Toygarı | *Melanocorypha calandra* |
| ⚫ Silent Places | Alaca Sinekkapan | *Ficedula semitorquata* |

---

## Key Findings

### Part 1 — Where Are Turkey's Birds Going?
- **83 species** drifting southward, **56 species** northward across all 256 species
- Average drift: **−0.037°/year** (≈ 4 km/decade southward)
- **Saksağan (Eurasian Magpie):** fastest northward shift at **+0.031°/year** — thriving in urbanizing landscapes
- **Kınalı Keklik (Chukar Partridge):** southward drift of **−0.067°/year** — toward lower, drier terrain
- **Büyük Toy & Kara Akbaba:** zero recordings — their silence in the data is the data

### Part 2 — Can a Machine Hear the Difference?
- **72 species**, Random Forest classifier, **25.4% accuracy** (baseline: 1.4%)
- Key finding: **variability > tone** — `mfcc_std_2` is the single most important feature
- A machine doesn't identify a bird by the average tone. It identifies it by **how unpredictable the call is**
- **Calandra Lark × Black-headed Bunting:** confused 2×, habitat distance only 32 km — same soundscape
- **Mistle Thrush × Eurasian Wren:** confused 1×, habitat distance 419 km — pure spectral overlap
- **Caspian Snowcock × Eurasian Scops Owl:** a mountain galliform confused with a nocturnal raptor

### Part 3 — Is the Voice Changing?
- **56 species** with ≥5 years of recordings analyzed for acoustic drift
- Cosine similarity between annual MFCC centroids and earliest baseline year
- Isolation Forest anomaly detection per species
- Cross-reference: does geographic drift (Part 1) correlate with acoustic drift (Part 3)?
- Temporal PCA: three periods (pre-2010, 2010–2018, 2019+) visualized in acoustic space

---

## Repository Structure

```
listening-to-extinction/
│
├── notebooks/
│   ├── part1_geographic_shift.ipynb     ← Geographic centroid analysis + MFCC extraction
│   ├── part2_classifier_shap.ipynb      ← Random Forest + SHAP
│   └── part3_acoustic_drift.ipynb       ← Cosine similarity + Isolation Forest + PCA time
│
├── data/
│   ├── turkey_birds_all.csv             ← All 2,303 cleaned recordings (256 species)
│   ├── turkey_birds_focus.csv           ← 9 focus species
│   ├── turkey_birds_shift_all.csv       ← Habitat centroid shift summary (all species)
│   ├── turkey_birds_shift_focus.csv     ← Habitat centroid shift summary (focus species)
│   ├── turkey_birds_centroids_all.csv   ← Annual centroids (all species)
│   ├── turkey_birds_mfcc_full.csv       ← 39-dim MFCC feature matrix (all species)
│   ├── turkey_birds_shap_importance.csv ← SHAP feature importance (Part 2)
│   └── turkey_birds_mfcc_predictions.csv← Model predictions (Part 2)
│
├── models/
│   └── rf_model.pkl                     ← Trained Random Forest + scaler + encoder
│
├── figures/
│   ├── turkey_bird_shift_map.png        ← Habitat shift arrows on Turkey map
│   ├── turkey_birds_overview.png        ← Density map + yearly trend
│   ├── turkey_bird_slope.png            ← N-S drift distribution (all + focus)
│   ├── turkey_bird_density.png          ← Yearly recording density by category
│   ├── turkey_birds_mfcc_heatmap.png    ← MFCC acoustic fingerprints heatmap
│   ├── turkey_birds_pca.png             ← PCA scatter (category + species)
│   ├── turkey_birds_confusion_matrix.png← Confusion matrix (Part 2)
│   ├── turkey_birds_shap_importance.png ← SHAP global feature importance (Part 2)
│   ├── turkey_birds_shap_beeswarm.png   ← SHAP beeswarm focus species (Part 2)
│   ├── turkey_birds_acoustic_drift.png  ← Cosine similarity trajectories (Part 3)
│   ├── turkey_birds_drift_vs_shift.png  ← Geographic × acoustic drift scatter (Part 3)
│   ├── turkey_birds_anomaly.png         ← Isolation Forest anomaly timeline (Part 3)
│   └── turkey_birds_pca_time.png        ← Acoustic space across three time periods (Part 3)
│
└── README.md
```

---

## Methodology

### Part 1 — Geographic Shift Analysis

For each species, we compute the **annual habitat centroid** — the mean latitude and longitude of all recordings in a given year. We then fit a **linear regression** over time:

$$\bar{lat}_{t} = \alpha + \beta \cdot t + \epsilon$$

- **β > 0** → northward shift
- **β < 0** → southward shift
- Minimum 3 years of data required for regression

### Part 1 — MFCC Acoustic Fingerprinting

Using [librosa](https://librosa.org), we extract **39 features** per recording:

| Feature group | Dimensions | What it captures |
|---|---|---|
| MFCC mean | 13 | Spectral shape — the tonal "color" of the sound |
| MFCC std | 13 | Variability — how much the spectrum changes over time |
| MFCC delta mean | 13 | Temporal dynamics — the rate of change in the sound |

### Part 2 — Random Forest Classifier + SHAP

- 72 species with ≥10 Turkey recordings, quality A/B only
- Random Forest (300 trees, `class_weight='balanced'`)
- 5-fold stratified cross-validation before test evaluation
- SHAP TreeExplainer for exact feature attribution
- Key insight: `mfcc_std` group dominates over `mfcc_mean` and `mfcc_delta`

### Part 3 — Acoustic Drift Over Time

- **Cosine similarity:** annual mean MFCC vector vs. earliest year baseline, per species
- **Isolation Forest:** anomaly detection per species (contamination=0.15)
- **Temporal PCA:** acoustic space split into three periods (pre-2010, 2010–2018, 2019+)
- Requires ≥5 distinct recording years per species (56 species qualify)

---

## ⚠️ Methodological Note

Xeno-canto data reflects **voluntary observer contributions**, not systematic biodiversity surveys. Low recording counts should be interpreted as *potential* signals, not confirmed population trends.

Two types of silence exist in this dataset:
1. **Ecological silence** — the species is genuinely rare or absent
2. **Observer silence** — no one has been there to listen

Acoustic drift (Part 3) may additionally reflect changes in the *observer population* rather than the birds themselves. All findings should be interpreted as a framework for hypothesis generation, not confirmed ecological conclusions.

---

## Requirements

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn shap librosa soundfile scipy
```

You will need a **Xeno-canto API key** (free, requires registration at [xeno-canto.org](https://xeno-canto.org)):

```python
XC_API_KEY = "your_key_here"
```

---

## Blog Series

Full write-up at [Code Beyond the Earth](https://codebeyondtheearth.substack.com):

- **Part 1** — Where Are Turkey's Birds Going? *(live — May 22, 2026)*
- **Part 2** — AI Listens: How Does a Machine Hear a Bird? *(live — May 2026)*
- **Part 3** — The Changing Voice: Acoustic Drift Over Time *(live — May 2026)*

---

## Data Attribution

Recordings © respective recordists, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) via [Xeno-canto Foundation](https://xeno-canto.org).

Analysis and code © Ece Özen İldem, 2026. Licensed MIT.

---

*Built on International Day for Biological Diversity 2026.*  
*"Acting Locally for Global Impact" — [IDB 2026](https://www.cbd.int/biodiversity-day/2026)*
