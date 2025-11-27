# ========================== ENV FIX ==========================
# (You can comment this out if protobuf errors are gone.)
import os
os.system(
    "pip install --upgrade protobuf==3.20.3 transformers accelerate sentencepiece safetensors --quiet"
)

# ========================== IMPORTS ==========================
import glob, warnings
warnings.filterwarnings("ignore")

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    RobertaModel, RobertaTokenizerFast,
    SwinModel, AutoImageProcessor,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import classification_report, accuracy_score

# ============================= PATHS =============================

BASE = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"

DATA = {
    "damage": {
        "train": f"{BASE}/task_damage_text_img_train.tsv",
        "dev":   f"{BASE}/task_damage_text_img_dev.tsv"
    },
    "humanitarian": {
        "train": f"{BASE}/task_humanitarian_text_img_train.tsv",
        "dev":   f"{BASE}/task_humanitarian_text_img_dev.tsv"
    },
    "informative": {
        "train": f"{BASE}/task_informative_text_img_train.tsv",
        "dev":   f"{BASE}/task_informative_text_img_dev.tsv"
    }
}

# ====================== IMAGE INDEXING (DEEP) ======================

print("\n🔍 scanning images...")
PATTERN = "/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES/*/*/*.jpg"
IMAGE_INDEX = {}

for p in tqdm(glob.glob(PATTERN), desc="Indexing images"):
    f = os.path.basename(p)
    if f.startswith("._"):
        continue
    key = f.rsplit(".jpg", 1)[0]
    IMAGE_INDEX[key] = p

print("\n📦 total images found =", len(IMAGE_INDEX))
print(list(IMAGE_INDEX.items())[:5], "\n")

# ========================== LABEL MAPS ============================

DMAP = {
    'little_or_no_damage': 0,
    'mild_damage':         1,
    'severe_damage':       2
}

T2MAP = {
    'not_humanitarian':            0,
    'other_relevant_information':  0,
    'affected_individuals':        1,
    'injured_or_dead_people':      1,
    'missing_or_found_people':     1,
    'rescue_volunteering_or_donation_effort': 1,
    'infrastructure_and_utility_damage':      2,
    'vehicle_damage':                            2
}

# T3TYPE = infra vs vehicle only → 2 classes (0,1)
T3TYPE = {
    'infrastructure_and_utility_damage': 0,
    'vehicle_damage':                    1
}

T4MAP = {
    'affected_individuals':        0,
    'injured_or_dead_people':      0,
    'missing_or_found_people':     0,
    'rescue_volunteering_or_donation_effort': 1,
    'other_relevant_information':           2,
    'not_humanitarian':                     2,
    'infrastructure_and_utility_damage':    2,
    'vehicle_damage':                       2
}

# ======================= TSV LOADING ==============================

def read_tsv(path):
    with open(path, 'r', encoding="utf8") as f:
        hdr = f.readline().strip().split("\t")
        rows = [l.strip().split("\t") for l in f]
    return hdr, rows

def find(h, cands):
    for x in cands:
        if x in h:
            return x
    return h[0]

def load_all():
    TRAIN, DEV = [], []

    for task in DATA:
        for split in ["train", "dev"]:
            hdr, rows = read_tsv(DATA[task][split])
            TXT = find(hdr, ["tweet_text", "text", "tweet"])
            IMG = find(hdr, ["image", "image_id"])
            LAB = find(hdr, ["label", "class", "label_text_image"])

            for r in rows:
                d = {hdr[i]: r[i] if i < len(r) else "" for i in range(len(hdr))}
                img_id = d[IMG].replace(".jpg", "").replace(".jpeg", "")

                item = {
                    "tweet": d[TXT],
                    "img":   IMAGE_INDEX.get(img_id, None),
                    "t1": -1,
                    "t2": -1,
                    "t3t": -1,
                    "t3s": -1,
                    "t4": -1
                }

                if LAB:
                    lab = d[LAB].lower()
                    if task == "informative":
                        item["t1"] = 1 if lab == "informative" else 0
                    if task == "damage":
                        item["t3s"] = DMAP.get(lab, -1)
                    if task == "humanitarian":
                        item["t2"]  = T2MAP.get(lab,  -1)
                        item["t3t"] = T3TYPE.get(lab, -1)
                        item["t4"]  = T4MAP.get(lab,  -1)

                (TRAIN if split == "train" else DEV).append(item)

    print(f"\n📊 rows: {len(TRAIN)} train  |  {len(DEV)} dev\n")
    return TRAIN, DEV

# ======================= CLASS STATS & WEIGHTS ====================

def compute_class_stats(train_rows):
    stats = {
        "t1": Counter(),
        "t2": Counter(),
        "t3t": Counter(),
        "t3s": Counter(),
        "t4": Counter()
    }
    for r in train_rows:
        for k in stats:
            v = r[k]
            if v >= 0:
                stats[k][v] += 1
    return stats

def make_weights(counter, num_classes):
    """
    Milder weighting: 1 / sqrt(freq + 1), normalized to mean=1.
    """
    import math
    freqs = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freqs = freqs + 1.0
    inv = 1.0 / torch.sqrt(freqs)
    inv = inv / inv.mean()
    return inv

# ======================= DATASET ================================

class CRISIS(Dataset):
    def __init__(self, data, tokenizer, processor):
        self.data = data
        self.tok = tokenizer
        self.proc = processor
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]

        try:
            img = Image.open(d["img"]).convert("RGB") if d["img"] else Image.new("RGB", (224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224))

        if torch.rand(1) < 0.4:
            img = self.aug(img)

        T = self.tok(
            d["tweet"],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )
        P = self.proc(images=img, return_tensors="pt")

        return {
            "input_ids":      T.input_ids[0],
            "attention_mask": T.attention_mask[0],
            "pixel_values":   P.pixel_values[0]
        }, {
            "t1":  torch.tensor(d["t1"],  dtype=torch.long),
            "t2":  torch.tensor(d["t2"],  dtype=torch.long),
            "t3t": torch.tensor(d["t3t"], dtype=torch.long),
            "t3s": torch.tensor(d["t3s"], dtype=torch.long),
            "t4":  torch.tensor(d["t4"],  dtype=torch.long)
        }

def collate(b):
    X, Y = zip(*b)
    batch_x = {k: torch.stack([x[k] for x in X]) for k in X[0]}
    batch_y = {k: torch.stack([y[k] for y in Y]) for k in Y[0]}
    return batch_x, batch_y

# ========================= MODEL ===============================

class HEAD(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, o)
        )
    def forward(self, x):
        return self.m(x)

class FUSE(nn.Module):
    """
    Cross-modal Transformer fusion with learnable CLS.
    """
    def __init__(self, d=512, layers=2, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            batch_first=True,
            dim_feedforward=d * 4,
            dropout=0.1,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d))

    def forward(self, t, v):
        B = t.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, t[:, None], v[:, None]], dim=1)
        out = self.enc(seq)
        return out[:, 0]  # CLS token

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.txt = RobertaModel.from_pretrained("roberta-base")
        self.txt.pooler = None  # avoid unused pooler warning
        self.vis = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        # Projections
        self.tp = nn.Linear(self.txt.config.hidden_size, 512)
        self.vp = nn.Linear(self.vis.config.hidden_size, 512)

        # Fusion
        self.fuse = FUSE(512, layers=2, heads=8)

        # Shared representation
        self.shared_norm = nn.LayerNorm(512)

        # T3 expert gate
        self.t3_gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512)
        )

        # Heads
        self.h1  = HEAD(512, 2)  # t1: binary
        self.h2  = HEAD(512, 3)  # t2: 3-way
        self.h3t = HEAD(512, 2)  # t3-type: 2-way (infra vs vehicle)
        self.h3s = HEAD(512, 3)  # t3-severity: 3-way
        self.h4  = HEAD(512, 3)  # t4: 3-way

    def forward(self, B):
        txt_out = self.txt(B["input_ids"], B["attention_mask"])
        t = txt_out.last_hidden_state[:, 0]  # CLS

        vis_out = self.vis(pixel_values=B["pixel_values"])
        v = vis_out.last_hidden_state.mean(1)

        t_proj = self.tp(t)
        v_proj = self.vp(v)

        z = self.fuse(t_proj, v_proj)
        z = self.shared_norm(z)

        z_t3 = self.t3_gate(z)

        return {
            "t1":  self.h1(z),
            "t2":  self.h2(z),
            "t3t": self.h3t(z_t3),
            "t3s": self.h3s(z_t3),
            "t4":  self.h4(z)
        }

# ========================= LOSS HELPERS =======================

def focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Standard focal loss on top of cross-entropy.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

def multi_task_loss(outputs, labels, class_weights, device):
    """
    - t1, t2, t4 → standard CE + weights
    - t3t, t3s   → focal + stronger weight
    """
    total = 0.0
    for k in labels.keys():
        mask = labels[k] >= 0
        if not mask.any():
            continue

        logits = outputs[k][mask]
        targets = labels[k][mask]
        w = None
        if class_weights is not None and k in class_weights and class_weights[k] is not None:
            w = class_weights[k].to(device)

        if k in ["t3t", "t3s"]:
            # heavier focus for minority damage tasks
            loss_k = focal_loss(logits, targets, weight=w, gamma=2.5)
            loss_k = loss_k * 1.7
        else:
            loss_k = F.cross_entropy(logits, targets, weight=w, reduction="mean")

        total = total + loss_k

    return total

# ========================= EVAL ON DEV ===========================

def evaluate(model, loader, device):
    model.eval()
    all_preds = {k: [] for k in ["t1", "t2", "t3t", "t3s", "t4"]}
    all_true  = {k: [] for k in ["t1", "t2", "t3t", "t3s", "t4"]}

    with torch.no_grad():
        for B, Y in loader:
            B = {k: v.to(device) for k, v in B.items()}
            out = model(B)
            for k in all_preds:
                mask = Y[k] >= 0
                if mask.any():
                    preds = out[k][mask].argmax(1).cpu().tolist()
                    labels = Y[k][mask].cpu().tolist()
                    all_preds[k].extend(preds)
                    all_true[k].extend(labels)

    reports = {}
    print("\n================= DEV RESULTS =================\n")
    for k in all_preds:
        if len(all_true[k]) == 0:
            continue
        print(f"\n--- TASK {k.upper()} ---")
        rep_text = classification_report(all_true[k], all_preds[k], digits=4, zero_division=0)
        print(rep_text)
        acc = accuracy_score(all_true[k], all_preds[k])
        print(f"Accuracy: {acc:.4f}")
        rep_dict = classification_report(all_true[k], all_preds[k], output_dict=True, zero_division=0)
        reports[k] = rep_dict

    # Weighted macro-F1 across tasks, with emphasis on humanitarian + damage
    weights = {
        "t1":  1.0,
        "t2":  2.0,
        "t3t": 2.0,
        "t3s": 2.0,
        "t4":  1.5
    }
    total_score = 0.0
    total_w = 0.0
    for k, rep in reports.items():
        if "macro avg" in rep:
            f1 = rep["macro avg"]["f1-score"]
            w = weights.get(k, 1.0)
            total_score += w * f1
            total_w += w

    main_score = total_score / total_w if total_w > 0 else 0.0
    print(f"\n>>> Combined dev score (weighted macro F1) = {main_score:.4f}\n")
    return main_score, reports

# ========================= TRAIN =============================

def train():
    train_rows, dev_rows = load_all()

    # ---- class stats + weights ----
    stats = compute_class_stats(train_rows)
    print("Class stats:", stats)

    class_weights = {
        "t1":  make_weights(stats["t1"],  2) if len(stats["t1"])  > 0 else None,
        "t2":  make_weights(stats["t2"],  3) if len(stats["t2"])  > 0 else None,
        "t3t": make_weights(stats["t3t"], 2) if len(stats["t3t"]) > 0 else None,
        "t3s": make_weights(stats["t3s"], 3) if len(stats["t3s"]) > 0 else None,
        "t4":  make_weights(stats["t4"],  3) if len(stats["t4"])  > 0 else None,
    }
    print("Class weights:")
    for k, v in class_weights.items():
        if v is None:
            print(k, None)
        else:
            print(k, v.tolist())

    # ---- sampler weights: oversample T3 + humanitarian damage ----
    sample_weights = []
    for r in train_rows:
        w = 1.0
        if r["t3t"] >= 0:
            w *= 4.0   # strongly boost t3-type rows
        if r["t3s"] >= 0:
            w *= 3.0   # boost severity rows
        if r["t2"] == 2:
            w *= 2.0   # humanitarian damage (rare-ish)
        sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tok  = RobertaTokenizerFast.from_pretrained("roberta-base")
    proc = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    train_ds = CRISIS(train_rows, tok, proc)
    dev_ds   = CRISIS(dev_rows,   tok, proc)

    TL = DataLoader(train_ds, batch_size=8, sampler=sampler, shuffle=False,
                    collate_fn=collate, num_workers=0)
    DL = DataLoader(dev_ds,   batch_size=8, shuffle=False,
                    collate_fn=collate, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n🟢 Using device:", device, "\n")

    model = MODEL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    epochs = 60
    total_steps = len(TL) * epochs
    warmup_steps = int(0.03 * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs("/kaggle/working/checkpoints_sota", exist_ok=True)
    best_score = -1.0
    best_path  = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for B, Y in tqdm(TL, desc=f"Epoch {ep+1}/{epochs}"):
            B = {k: v.to(device) for k, v in B.items()}
            Y = {k: v.to(device) for k, v in Y.items()}
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(B)
                loss = multi_task_loss(out, Y, class_weights=class_weights, device=device)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item()

        print(f"\n🟣 Epoch {ep+1} Train Loss = {total_loss/len(TL):.4f}\n")

        # ---- dev eval ----
        dev_score, _ = evaluate(model, DL, device)

        # ---- save epoch checkpoint ----
        ep_path = f"/kaggle/working/checkpoints_sota/E{ep+1}.pt"
        torch.save(model.state_dict(), ep_path)
        print("💾 Saved epoch checkpoint:", ep_path)

        # ---- save best by combined macro F1 ----
        if dev_score > best_score:
            best_score = dev_score
            best_path = "/kaggle/working/checkpoints_sota/best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New best model (score={best_score:.4f}) saved to:", best_path)

    print("\nTraining done. Best combined dev score =", best_score, "at", best_path)

# ========================= RUN =============================

if __name__ == "__main__":
    train()
