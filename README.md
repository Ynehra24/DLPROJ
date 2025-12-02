# 🚨 DisasterAI – Multimodal Crisis Understanding

DisasterAI is a fusion pipeline for **marking safe zones and understanding on ground situations** during crises using:

* social media posts (tweets + images)
* satellite / aerial imagery (pre and post disaster)

The repository contains **four classification models (T1–T4)** plus a **Siamese damage model (SEG)** and a **final integrated pipeline** that ties them together.

---

## 🔥 Overview

The long-term goal is to move toward a **disaster foundation model** that can:

* jointly learn from **text, RGB photos, and satellite imagery**
* generalize across **events, locations, and label schemes**
* support **flexible decision-making**: identifying safe zones, prioritizing rescue, and understanding damage patterns.

This repo is a step in that direction: a **5-model system** trained on CrisisMMD and xView2-style data, with a unified pipeline for running them together.

---

## 🧠 Model Suite

> All text encoders use **DistilBERT**, and all image encoders use **ConvNeXt-Tiny** unless otherwise noted.

| Model                                | Input                     | Output                          | Purpose                                     |
| ------------------------------------ | ------------------------- | ------------------------------- | ------------------------------------------- |
| **T1 – Fusion Relevance Classifier** | Tweet text + tweet image  | Informative / Non-Informative   | Filters crisis data that is actually useful |
| **T2 – Humanitarian**                | Tweet text + image        | Humanitarian / Structure / None | High-recall humanitarian signal extraction  |
| **T3 – Light Multimodal Damage**     | Tweet text + image        | Little / Mild / Severe damage   | Damage severity estimation around events    |
| **T4 – Multimodal Subtypes**         | Tweet text + image        | People_Affected / Rescue / None | Distinguishes people vs rescue vs no-human  |
| **SEG – Siamese Damage Classifier**  | Pre + Post satellite crop | No / Minor / Major / Destroyed  | Structure-level change / damage reasoning   |

---

## 🧱 Architecture Details

### **T1 – Fusion Relevance**

* **Inputs:** tweet text + associated image
* **Text branch:** DistilBERT → CLS vector (768-dim) → LayerNorm
* **Image branch:** ConvNeXt-Tiny → pooled feature (e.g., 768-dim)
* **Fusion:** concatenation `[img, text] → Linear → GELU → Linear → logits(2)`
* **Training objective:** weighted cross-entropy with class weights from CrisisMMD label distribution
* **Use:** screen out non-informative posts early in the pipeline.

---

### **T2 – Text-Only Humanitarian**

* **Inputs:** tweet text only
* **Encoder:** DistilBERT (base uncased)
* **Head:**

  * CLS (768) → LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→3)
* **Labels (compressed from 8-way):**

  * `humanitarian` – affected people, injuries, missing, rescue/donation
  * `structure` – infrastructure or vehicle damage
  * `non_informative` – other/irrelevant
* **Role:** provides a **language-only** view of humanitarian intent and type.

---

### **T3 – Light Multimodal Damage**

* **Inputs:** tweet text + associated image
* **Image encoder:** ConvNeXt-Tiny (no classification head, pooled features)
* **Text encoder:** DistilBERT (CLS token)
* **Fusion head:**

  * `[img, text]` concat → Linear(fused_dim→256) → GELU → Dropout
  * Linear(256→64) → GELU → Linear(64→3)
* **Labels:** `little_or_no_damage`, `mild_damage`, `severe_damage`
* **Observation:** strong performance on **severe damage**, confusion between **little vs mild** due to visual ambiguity and class imbalance.

---

### **T4 – Multimodal Humanitarian Subtypes**

* **Inputs:** same tweet text + image pair
* **Encoders:** ConvNeXt-Tiny (image) + DistilBERT (text), identical to T3
* **Fusion:** same late-fusion MLP as T3, but trained for different labels
* **Labels:**

  * `people_affected` – affected, injured, or missing individuals
  * `rescue_needed` – volunteering, donations, search & rescue
  * `no_human` – damage / context information without explicit people
* **Performance:**

  * Strong overall F1 (~0.85), best among the suite
  * Slightly weaker recall for people_affected due to class imbalance.

---

### **SEG – Siamese Damage Segmentation / Classifier**

Implemented in `segmentation.ipynb` with a MiT-B2 backbone.

* **Inputs:** crop around a structure from **pre-disaster** and **post-disaster** satellite imagery
* **Encoder:** `SiameseMiTEncoder`

  * shared MiT-B2 encoder (from `segmentation_models_pytorch`)
  * each image → feature map `(B, 512, 7, 7)` → global average pooling → `(B, 512)`
* **Fusion + classifier (`DamageClassifier`):**

  * Concatenate `[f_pre, f_post]` → `(B, 1024)`
  * Fusion MLP: `1024 → 512 → 512` with ReLU
  * Classifier: `512 → 128 → 4`
* **Outputs:** logits for `["no-damage", "minor-damage", "major-damage", "destroyed"]`
* **Use:** building-level damage reasoning based on **change**, not just appearance.

---

## 🌐 End-to-End Pipeline

The scripts `final_pipeline.ipynb` and `final_segmentandtweet_pipeline.py` connect all pieces:

1. **Satellite Damage Analysis (SEG)**

   * Take pre– and post-disaster imagery.
   * Run the Siamese MiT damage model to classify building damage.
   * Optionally overlay results as a damage map (from `segmentation.ipynb`).

2. **Social Media Filtering (T1)**

   * For each tweet + image, run T1 to keep only **informative** posts.

3. **Humanitarian Interpretation (T2 + T4)**

   * T2 (text-only) provides robust humanitarian vs structure vs non-info view.
   * T4 (multimodal) sharpens focus on **people_affected** and **rescue_needed**.

4. **Damage Context (T3)**

   * T3 links the social media image/text to coarse damage severity around the event.

5. **Safe Zone / Situation View**

   * Combine:

     * SEG’s **spatial damage map**, and
     * T1–T4 outputs (what people say + what images show)
---

## 📁 Repository Structure

```bash
/
├── README.md                           # Project documentation
│
├── T1.py                               # T1 – Informative vs Non-Informative (fusion)
├── T2.py                               # T2 – Human / Structure / Non-Informative (text-only)
├── T3.py                               # T3 – Multimodal damage classifier
├── T4.py                               # T4 – Multimodal humanitarian subtype classifier
│
├── architecture_redefinition.py        # Shared multi-modal architecture definitions
├── final_pipeline.ipynb                # Jupyter notebook: multi-model pipeline experiments
├── final_segmentandtweet_pipeline.py   # Final Python pipeline: segmentation + tweets
│
├── kaggle_textmodel_main.py            # Kaggle execution entrypoint for text models
├── main_DL.ipynb                       # Early development / prototyping notebook
├── main_DL_text.py                     # Standalone text model experimentation
│
├── retraining_text_model.py            # Resume / fine-tune DistilBERT text model
│
└── segmentation.ipynb                  # SEG model + damage overlay visualizations
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Ynehra24/DLPROJ.git
cd DLPROJ
pip install -r requirements.txt
```

(If you use different envs for Kaggle / local, list them in the repo.)

---

## 📊 Datasets

* **CrisisMMD** – multimodal social media crisis dataset
  [https://crisisnlp.qcri.org/crisismmd](https://crisisnlp.qcri.org/crisismmd)
* **xView2 (or similar)** – pre/post satellite building damage dataset
  [https://xview2.org/](https://xview2.org/)

---

## 🚀 Training / Re-Training

> These commands assume dataset paths are configured inside each script or via argparse flags.

```bash
# Tweet relevance and humanitarian tasks
python T1.py         # Train/finetune T1 fusion model
python T2.py         # Train/finetune T2 text-only humanitarian model
python T3.py         # Train/finetune T3 multimodal damage model
python T4.py         # Train/finetune T4 multimodal subtype model

# SEG Siamese damage model (usually run as notebook, but logic can be scripted)
# segmentation.ipynb – training and visualization
```

For **resume training / Kaggle checkpoints**:

```bash
python retraining_text_model.py
python kaggle_textmodel_main.py
```

---

## 🔍 Inference / Testing

Example CLI patterns (adapt to your actual arguments):

```bash
# T1: relevance
python T1.py --mode test --img example.jpg --text "Sample tweet text"

# T2: humanitarian / structure / non-info
python T2.py --mode test --tsv data/task2_test.tsv

# T3: damage severity from tweet + image
python T3.py --mode test --img example.jpg --text "..."

# T4: subtype classification
python T4.py --mode test --img example.jpg --text "..."

# Final combined pipeline: segmentation + tweet models
python final_segmentandtweet_pipeline.py \
    --pre pre_disaster.png \
    --post post_disaster.png \
    --tweets tweets.json
```
---

## 📊 Performance Summary

| Model   | Metric (Test)            | Notes                                        |
| ------- | ------------------------ | -------------------------------------------- |
| **T1**  | F1 = **0.8318**          | Fusion outperforms text-only & image-only    |
| **T2**  | F1 = **0.8297**          | Strong humanitarian detection; text-focused  |
| **T3**  | F1 = **0.6106**          | High for severe damage; mild vs little noisy |
| **T4**  | F1 = **0.8502**          | Best overall; strong on no_human + rescue    |
| **SEG** | Weighted F1 ≈ **0.8854** | Very effective at pre/post change detection  |

---

## Our Output - 
<p align="center">
  <img src="https://github.com/Ynehra24/DLPROJ/blob/main/pngimage.png" width="900">
  <img src="https://github.com/Ynehra24/DLPROJ/blob/main/o1.jpeg" width="900">
  <img src="https://github.com/Ynehra24/DLPROJ/blob/main/o2.jpeg" width="900">
  <img src="https://github.com/Ynehra24/DLPROJ/blob/main/o3.jpeg" width="900">
</p>

## Some sample outputs for the CrisisMMD model - 
📝 Tweet ID 911619664597405697: Hurricane Maria Recovery: OECS Moves Beyond Climate Change to Climate Reality https://t.co/OccF3UHWFM https://t.co/KdYlq0le7X
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 2 T4 Pred: 1 | True: 2

📝 Tweet ID 904925838839169028: Remnants of Harvey Spawn Tornadoes, Floods Across Deep South https://t.co/PqQ24484HC https://t.co/FpaT5Goyvw
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 0 T4 Pred: 1 | True: 2

📝 Tweet ID 910160788849008640: Pearl Fincher Museum continues post-Harvey cleanup efforts https://t.co/XtB919dtLW https://t.co/TBNZNBMPup
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 1 T4 Pred: 1 | True: 2

📝 Tweet ID 929989658351505408: RT @HuffPost: Hundreds dead after powerful #earthquake hits border between Iran and Iraq https://t.co/p8TJjvjOhr https://t.co/l4PJqZxiaz
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 2 T4 Pred: 1 | True: 0

📝 Tweet ID 910225493877813249: Work ongoing to repair damage to #edistobeach post Irma. Lots of sand, barriers out of place @ABCNews4 #chstrfc #chs https://t.co/6hNdGGgQPg
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 0 T4 Pred: 1 | True: 1

📝 Tweet ID 912077714227593216: RT @TreyYingst: New satellite images show widespread destruction in St. Croix after Hurricane Maria https://t.co/qTrQXHNxMQ
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 1 T4 Pred: 1 | True: 2

📝 Tweet ID 908406994461224960: Leftists to the Rescue: Where the State and Big NGOs Fail, Mutual Aid Networks Step In https://t.co/XJa3ScAjx7 https://t.co/oHGm55SwPf
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 1 T3 Pred: 1 | True: 2 T4 Pred: 1 | True: 2

📝 Tweet ID 930010119332679682: #BreakingNews Death toll from 7.3-magnitude #earthquake at #Iran #Iraq border now stands at 330 #Kurdistan https://t.co/vSF8LAEZQo
T1 Pred: 1 | True: 1 T2 Pred: 1 | True: 2 T3 Pred: 1 | True: 2 T4 Pred: 1 | True: 0

==================== FINAL SCORES ====================
T1: 0.95
T2: 0.52
T3: 0.86
T4: 0.99
======================================================

## 📌 Key Learnings

* **Multimodal > Single-Modality**
  Fusion of text + image consistently beats isolated branches, especially for relevance (T1) and subtypes (T4).

* **Imbalance Hurts the Rare Classes**
  Mild damage, structure, and people_affected are under-represented and show reduced F1; balancing or reweighting is critical.

* **Images Encode Damage, Text Encodes Intent**
  Vision models shine on structural damage, while text models excel at describing **who** is affected and **what** is happening.

* **Siamese Pre/Post Modelling is Powerful**
  SEG’s MiT-B1 Siamese encoder learns **change** rather than static appearance and is robust for structure-level damage analysis.

* **Towards a Foundation-Style Disaster Model**
  Combining SEG + T1–T4 into a single pipeline provides a blueprint for a future **multi-task, multi-modal disaster foundation model**.

---

## 📄 License
`Apache 2.0`

---

## 🤝 Credits

* Authors and contributors of this repository
* CrisisMMD & xView2 dataset creators
* DistilBERT, ConvNeXt, MiT, and segmentation_models_pytorch authors
* All upstream open-source libraries used in this project

---
