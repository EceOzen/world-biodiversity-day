# Listening to Extinction
### A Machine Learning Analysis of Bird Sound Recordings Across Turkey

> *Built on World Biodiversity Day 2026 — "Acting Locally for Global Impact"*  
> *Code Beyond the Earth × IDB 2026*

---

## What is this?

On May 22, 2026 — International Day for Biological Diversity — I asked a simple but unsettling question:

**Are Turkey's birds moving? And if so, where?**

This repository contains the full four-part analysis pipeline: from raw API calls to geographic shift detection, acoustic fingerprinting, machine learning classification, acoustic drift over time, and MAE embedding replication. The data source is [Xeno-canto](https://xeno-canto.org), a global crowdsourced archive of wildlife sound recordings.

This project started with a question about biodiversity. It ends with a question about measurement: **what we find depends on how we look.**

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
| Losers | Kınalı Keklik | *Alectoris chukar* |
| Losers | Bıldırcın | *Coturnix coturnix* |
| Losers | Büyük Toy | *Otis tarda* |
| Winners | Saksağan | *Pica pica* |
| Winners | Ebabil | *Apus apus* |
| Winners | Kızıl Şahin | *Buteo buteo* |
| Silent Places | Kara Akbaba | *Aegypius monachus* |
| Silent Places | Bozkır Toygarı | *Melanocorypha calandra* |
| Silent Places | Alaca Sinekkapan | *Ficedula semitorquata* |

---

## Key Findings

### Part 1 — Where Are Turkey's Birds Going?
- **83 species** drifting southward, **56 species** northward across all 256 species
- Average drift: **−0.037°/year** (approx. 4 km/decade southward)
- **Saksağan (Eurasian Magpie):** fastest northward shift at **+0.031°/year** — thriving in urbanizing landscapes
- **Kınalı Keklik (Chukar Partridge):** southward drift of **−0.067°/year** — toward lower, drier terrain
- **Büyük Toy and Kara Akbaba:** zero recordings — their silence in the data is the data

### Part 2 — Can a Machine Hear the Difference?
- **72 species**, Random Forest classifier, **25.4% accuracy** (baseline: 1.4%)
- Key finding: **variability > tone** — `mfcc_std_2` is the single most important feature
- A machine doesn't identify a bird by its average tone. It identifies it by **how unpredictable the call is**
- **Calandra Lark × Black-headed Bunting:** confused 2×, habitat distance only 32 km — same soundscape
- **Mistle Thrush × Eurasian Wren:** confused 1×, habitat distance 419 km — pure spectral overlap
- **Caspian Snowcock × Eurasian Scops Owl:** a mountain galliform confused with a nocturnal raptor

### Part 3 — Is the Voice Changing?
- **56 species** with ≥5 years of recordings analyzed for acoustic drift
- **Cetti's Warbler** most acoustically drifted (similarity: 0.908 to baseline year)
- **European Robin** most stable (0.989) — consistent presence, consistent sound, across three decades
- Anomaly rate trend: **+0.0029/year** — rising acoustic unpredictability
- **Pearson r = −0.063** — geographic drift and acoustic drift are independent signals

### Part 4 — Does a Better Ear Change the Answer?
- Replaced 39-dim MFCC vectors with **768-dim AudioMAE embeddings** (ViT-B/16, AudioSet-2M pretrain)
- **MFCC + RF: 25.4% → AudioMAE + LR: 43.5%** — +18.1 percentage points
- Paper finding confirmed: pretraining scale dominates hand-crafted features
- Classifier ordering on MAE embeddings: **LR > SVM > RF** — the representation does the work
- **Drift analysis changed completely:** MAE embeddings show near-zero drift (0.9996–1.000) vs MFCC's 0.908–0.989 — representation changes what we hear
- Bird-MAE: environment compatibility issue, replication pending

---

## Repository Structure

```
listening-to-extinction/
│
├── notebooks/
│   ├── part1_geographic_shift.ipynb     <- Geographic centroid + MFCC extraction
│   ├── part2_classifier_shap.ipynb      <- Random Forest + SHAP
│   ├── part3_acoustic_drift.ipynb       <- Cosine similarity + Isolation Forest + PCA time
│   └── part4_mae_replication.ipynb      <- AudioMAE embeddings + classifier comparison
│
├── data/
│   ├── turkey_birds_all.csv             <- All 2,303 cleaned recordings (256 species)
│   ├── turkey_birds_focus.csv           <- 9 focus species
│   ├── turkey_birds_shift_all.csv       <- Habitat centroid shift summary (all species)
│   ├── turkey_birds_shift_focus.csv     <- Habitat centroid shift summary (focus species)
│   ├── turkey_birds_centroids_all.csv   <- Annual centroids (all species)
│   ├── turkey_birds_mfcc_full.csv       <- 39-dim MFCC feature matrix
│   ├── turkey_birds_shap_importance.csv <- SHAP feature importance (Part 2)
│   ├── turkey_birds_mfcc_predictions.csv<- Model predictions (Part 2)
│   └── turkey_birds_spectrograms.pt     <- Mel spectrogram tensors (Part 4)
│
├── models/
│   └── rf_model.pkl                     <- Random Forest + scaler + encoder (Part 2)
│
├── figures/
│   <- Part 1
│   ├── turkey_bird_shift_map.png        <- Habitat shift arrows on Turkey map
│   ├── turkey_birds_overview.png        <- Density map + yearly trend
│   ├── turkey_bird_slope.png            <- N-S drift distribution
│   ├── turkey_bird_density.png          <- Yearly recording density by category
│   <- Part 2
│   ├── turkey_birds_mfcc_heatmap.png    <- MFCC acoustic fingerprints
│   ├── turkey_birds_pca.png             <- PCA scatter (category + species)
│   ├── turkey_birds_confusion_matrix.png<- Confusion matrix
│   ├── turkey_birds_shap_importance.png <- SHAP global feature importance
│   ├── turkey_birds_shap_beeswarm.png   <- SHAP beeswarm focus species
│   <- Part 3
│   ├── turkey_birds_acoustic_drift.png  <- Cosine similarity trajectories
│   ├── turkey_birds_drift_vs_shift.png  <- Geographic x acoustic drift scatter
│   ├── turkey_birds_anomaly.png         <- Isolation Forest anomaly timeline
│   ├── turkey_birds_pca_time.png        <- Acoustic space across three time periods
│   <- Part 4
│   ├── turkey_birds_sample_spectrograms.png <- Sample mel spectrograms (6 species)
│   ├── turkey_birds_mae_comparison.png  <- Model accuracy comparison
│   └── turkey_birds_pca_comparison.png  <- MFCC vs AudioMAE embedding space
│
└── README.md
```

---

## Methodology

### Part 1 — Geographic Shift

Annual habitat centroid (mean lat/lon of recordings per year) + linear regression over time.

$$\bar{lat}_{t} = \alpha + \beta \cdot t + \epsilon$$

- **β > 0** → northward shift | **β < 0** → southward shift
- Minimum 3 years of data required

### Part 2 — MFCC + Random Forest + SHAP

39-dim feature vector per recording: 13 MFCC mean + 13 MFCC std + 13 MFCC delta mean.  
Random Forest (300 trees, `class_weight='balanced'`). 5-fold stratified CV.  
SHAP TreeExplainer for exact feature attribution.

### Part 3 — Acoustic Drift

Cosine similarity between annual mean MFCC vectors and earliest year baseline, per species.  
Isolation Forest anomaly detection (contamination=0.15) per species.  
Temporal PCA across three periods: pre-2010, 2010–2018, 2019+.

### Part 4 — AudioMAE Replication

**From MFCC to mel spectrogram:** 39-dim hand-crafted vector → 131,072-dim image (1024×128).  
**AudioMAE:** ViT-B/16, 85.6M parameters, pretrained on AudioSet-2M with MAE objective.  
**Spectrogram format:** `kaldi.fbank`, 128 mel bins, 16kHz, hanning window, 1024 frames.  
**Normalization:** mean=−4.2677393, std=4.5689974×2 (AudioMAE pretraining constants).  
**Classifier:** Logistic Regression (L2, C=1.0) on 768-dim standardized embeddings.

The key insight: the representation does the work. LR on AudioMAE embeddings (+18.1%) outperforms RF on MFCC features because the embedding space is linearly separable to a degree that MFCC space is not.

---

## Reproducibility Note

**Bird-MAE** (DBD-research-group/Bird-MAE-Base) could not be loaded in standard Colab/Kaggle environments due to a compatibility issue between the model's custom initialization code and recent versions of the `transformers` library. Bird-MAE was developed in a specific conda environment (`python=3.10.14`) that differs from standard notebook setups. This is a version dependency issue, not a fundamental problem with the model. The Bird-MAE comparison is pending.

This is worth noting as a general observation: **environment reproducibility is as important as code reproducibility.**

---

## Two Types of Silence

Xeno-canto data reflects voluntary observer contributions, not systematic biodiversity surveys. Two types of silence exist in this dataset:

1. **Ecological silence** — the species is genuinely rare or absent
2. **Observer silence** — no one has been there to listen

Acoustic drift (Part 3) may additionally reflect changes in the observer population rather than the birds themselves. All findings should be interpreted as a framework for hypothesis generation, not confirmed ecological conclusions.

---

## Requirements

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn shap librosa \
            soundfile scipy timm transformers torchaudio
```

Xeno-canto API key (free, register at [xeno-canto.org](https://xeno-canto.org)):

```python
XC_API_KEY = "your_key_here"
```

---

## Blog Series

Full write-up at [Code Beyond the Earth](https://codebeyondtheearth.substack.com):

- **Part 1** — Where Are Turkey's Birds Going? *(live — May 22, 2026)*
- **Part 2** — AI Listens: How Does a Machine Hear a Bird? *(live — May 2026)*
- **Part 3** — The Changing Voice: Acoustic Drift Over Time *(live — May 2026)*
- **Part 4** — Does a Better Ear Change the Answer? *(live — May 2026)*

---

## Citations

```bibtex
@article{liu2026masked,
  title={Masked Autoencoders with Limited Data: Does It Work?
         A Fine-Grained Bioacoustics Case Study},
  author={Liu, Wuao and others},
  journal={FGVC Workshop @ CVPR},
  year={2026}
}

@article{rauch2025birdmae,
  title={Can Masked Autoencoders Also Listen to Birds?},
  author={Rauch, Lukas and Moummad, Ilyass and Heinrich, René
          and Joly, Alexis and Sick, Bernhard and Scholz, Christoph},
  journal={arXiv:2504.12880},
  year={2025}
}
```

---

## Data Attribution

Recordings © respective recordists, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) via [Xeno-canto Foundation](https://xeno-canto.org).

Analysis and code © Ece Özen, 2026. Licensed MIT.

---

*Built on International Day for Biological Diversity 2026.*  
*"Acting Locally for Global Impact" — [IDB 2026](https://www.cbd.int/biodiversity-day/2026)*  
*With thanks to Wuao Liu for the conversation that started Part 4.*
