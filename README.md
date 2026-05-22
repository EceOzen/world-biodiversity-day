# world-biodiversity-day
# 🐦 Listening to Extinction
### A Machine Learning Analysis of Bird Sound Recordings Across Turkey

> *Built on World Biodiversity Day 2026 — "Acting Locally for Global Impact"*  
> *Code Beyond the Earth × IDB 2026*

---

## What is this?

On May 22, 2026 — International Day for Biological Diversity — I asked a simple but unsettling question:

**Are Turkey's birds moving? And if so, where?**

This repository contains the full analysis pipeline: from raw API calls to geographic shift detection, acoustic fingerprinting, and machine learning classification. The data source is [Xeno-canto](https://xeno-canto.org), a global crowdsourced archive of wildlife sound recordings. Every recording comes with GPS coordinates and a timestamp — which means it's not just an audio library. It's a spatial biodiversity dataset hiding in plain sight.

---

## The Dataset

| | |
|---|---|
| **Source** | Xeno-canto API v3 |
| **Coverage** | Turkey (lat: 36–42.5°N, lon: 26–45°E) |
| **Recordings** | 2,303 |
| **Species** | 256 |
| **Time range** | 2000–2026 |

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

## Key Findings (Part 1)

- **83 species** drifting southward, **56 species** northward across all 256 species
- Average drift: **−0.037°/year** (≈ 4 km/decade southward)
- **Saksağan (Eurasian Magpie):** fastest northward shift at **+0.031°/year** — thriving in urbanizing landscapes
- **Kınalı Keklik (Chukar Partridge):** southward drift of **−0.067°/year** — toward lower, drier terrain
- **Büyük Toy & Kara Akbaba:** zero recordings — their silence in the data is the data
- PCA on 39 MFCC features: 44.8% variance explained — losers and winners cluster; silent species scatter

---

## Repository Structure

```
listening-to-extinction/
│
├── notebooks/
│   ├── part1_geographic_shift.ipynb     ← Geographic centroid analysis + MFCC extraction
│   ├── part2_classifier_shap.ipynb      ← Random Forest + SHAP (coming soon)
│   └── part3_acoustic_drift.ipynb       ← Embedding similarity over time (coming soon)
│
├── data/
│   ├── turkey_birds_all.csv             ← All 2,303 cleaned recordings (256 species)
│   ├── turkey_birds_focus.csv           ← 9 focus species
│   ├── turkey_birds_shift_all.csv       ← Habitat centroid shift summary (all species)
│   ├── turkey_birds_shift_focus.csv     ← Habitat centroid shift summary (focus species)
│   ├── turkey_birds_centroids_all.csv   ← Annual centroids (all species)
│   └── turkey_birds_mfcc.csv            ← 39-dimensional MFCC feature matrix
│
├── figures/
│   ├── turkey_bird_shift_map.png        ← Habitat shift arrows on Turkey map
│   ├── turkey_birds_overview.png        ← Density map + yearly trend
│   ├── turkey_bird_slope.png            ← N-S drift distribution (all + focus)
│   ├── turkey_bird_density.png          ← Yearly recording density by category
│   ├── turkey_birds_mfcc_heatmap.png    ← MFCC acoustic fingerprints
│   └── turkey_birds_pca.png             ← PCA scatter (category + species)
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
| MFCC mean | 13 | Spectral shape |
| MFCC std | 13 | Variability across time |
| MFCC delta mean | 13 | Temporal change rate |

PCA reduces the 39-dimensional space to 2 components for visualization.

### Part 2 — Classifier + SHAP *(coming soon)*

- Random Forest classifier trained on MFCC features
- SHAP values to identify which acoustic dimensions distinguish species
- Confusion matrix analysis: which species sound alike, and why?
- Question: *what does it mean when a classifier can't tell two species apart?*

### Part 3 — Acoustic Drift *(coming soon)*

- Embedding similarity over time: has a species' sound changed across 20+ years of recordings?
- Anomaly detection in acoustic space
- Question: *can we detect not just where a species is moving, but how its voice is changing?*

---

## Methodological Note

Xeno-canto data reflects **voluntary observer contributions**, not systematic biodiversity surveys. Low recording counts should be interpreted as *potential* signals, not confirmed population trends.

Two types of silence exist in this dataset:
1. **Ecological silence** — the species is genuinely rare or absent
2. **Observer silence** — no one has been there to listen

Distinguishing between them is one of the core challenges of citizen science biodiversity analysis — and one of the most important questions this project raises.

---

## Requirements

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn shap librosa soundfile geopandas
```

You will need a **Xeno-canto API key** (free, requires registration at [xeno-canto.org](https://xeno-canto.org)). Add it to the notebook:

```python
XC_API_KEY = "your_key_here"
```

---

## Blog Series

Full write-up at [Code Beyond the Earth](https://codebeyondtheearth.substack.com):

- **Part 1** — Where Are Turkey's Birds Going? *(live — May 22, 2026)*
- **Part 2** — AI Listens: MFCC Classifier + SHAP *(coming soon)*
- **Part 3** — The Changing Voice: Acoustic Drift Over Time *(coming soon)*

---

## Data Attribution

Recordings © respective recordists, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) via [Xeno-canto Foundation](https://xeno-canto.org).

Analysis and code © Ece Özen, 2026. Licensed MIT.

---

*Built on International Day for Biological Diversity 2026.*  
*"Acting Locally for Global Impact" — [IDB 2026](https://www.cbd.int/biodiversity-day/2026)*
