# ========================== ENV FIX ==========================
import os
os.system("pip install --upgrade protobuf==3.20.3 transformers accelerate sentencepiece safetensors --quiet")

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
    "damage":{
        "train":f"{BASE}/task_damage_text_img_train.tsv",
        "dev":f"{BASE}/task_damage_text_img_dev.tsv"
    },
    "humanitarian":{
        "train":f"{BASE}/task_humanitarian_text_img_train.tsv",
        "dev":f"{BASE}/task_humanitarian_text_img_dev.tsv"
    },
    "informative":{
        "train":f"{BASE}/task_informative_text_img_train.tsv",
        "dev":f"{BASE}/task_informative_text_img_dev.tsv"
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
    IMAGE_INDEX[f.replace(".jpg", "")] = p

print("\n📦 total images found =", len(IMAGE_INDEX))
print(list(IMAGE_INDEX.items())[:5], "\n")

# ========================== LABEL MAPS ============================

DMAP = {'little_or_no_damage':0, 'mild_damage':1, 'severe_damage':2}

T2MAP = {
    'not_humanitarian':0, 'other_relevant_information':0,
    'affected_individuals':1, 'injured_or_dead_people':1,
    'missing_or_found_people':1, 'rescue_volunteering_or_donation_effort':1,
    'infrastructure_and_utility_damage':2, 'vehicle_damage':2
}

# T3T IS *BINARY* (0=infrastructure, 1=vehicle)
T3TYPE = {
    'infrastructure_and_utility_damage':0,
    'vehicle_damage':1
}

T4MAP = {
    'affected_individuals':0, 'injured_or_dead_people':0, 'missing_or_found_people':0,
    'rescue_volunteering_or_donation_effort':1, 'other_relevant_information':2,
    'not_humanitarian':2, 'infrastructure_and_utility_damage':2, 'vehicle_damage':2
}

# ======================= TSV LOADING ==============================

def read_tsv(path):
    with open(path, 'r', encoding="utf8") as f:
        hdr = f.readline().strip().split("\t")
        rows = [l.strip().split("\t") for l in f]
    return hdr, rows

def find(h, c):
    for x in c:
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
                img_id = d[IMG].replace(".jpg", "")

                item = {
                    "tweet": d[TXT],
                    "img": IMAGE_INDEX.get(img_id, None),
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
                        item["t2"]  = T2MAP.get(lab, -1)
                        item["t3t"] = T3TYPE.get(lab, -1)
                        item["t4"]  = T4MAP.get(lab, -1)

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
    # milder weighting: 1 / sqrt(freq + 1)
    import math
    freqs = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freqs = freqs + 1.0
    inv = 1.0 / torch.sqrt(freqs)
    inv = inv / inv.mean()
    return inv

# ---- per-sample weights for WeightedRandomSampler ----
def build_sample_weights(train_rows, stats, max_mult=10.0):
    """
    Each sample gets a weight boosted if it has rare labels in any task.
    Using ~sqrt(majority_freq / freq) per task, multiplicatively.
    """
    weights = []
    for r in train_rows:
        w = 1.0
        for task in ["t1","t2","t3t","t3s","t4"]:
            lbl = r[task]
            if lbl < 0 or len(stats[task]) == 0:
                continue
            freq = stats[task][lbl]
            maj_freq = max(stats[task].values())
            # sqrt imbalance factor
            if freq > 0:
                factor = (maj_freq / float(freq)) ** 0.5
                w *= factor
        # clip insanely large weights
        w = min(w, max_mult)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

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
        except:
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
            "input_ids": T.input_ids[0],
            "attention_mask": T.attention_mask[0],
            "pixel_values": P.pixel_values[0]
        }, {
            "t1": torch.tensor(d["t1"]),
            "t2": torch.tensor(d["t2"]),
            "t3t": torch.tensor(d["t3t"]),
            "t3s": torch.tensor(d["t3s"]),
            "t4": torch.tensor(d["t4"])
        }

def collate(b):
    X, Y = zip(*b)
    return {
        k: torch.stack([x[k] for x in X]) for k in X[0]
    }, {
        k: torch.stack([y[k] for y in Y]) for k in Y[0]
    }

# ========================= MODEL ===============================

class HEAD(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(.2),
            nn.Linear(256, o)
        )
    def forward(self, x):
        return self.m(x)

class FUSE(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        L = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=8,
            batch_first=True,
            dim_feedforward=d*4
        )
        self.enc = nn.TransformerEncoder(L, 2)
        self.cls = nn.Parameter(torch.randn(1, 1, d))

    def forward(self, a, b):
        B = a.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, a[:, None], b[:, None]], 1)
        return self.enc(seq)[:, 0]

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = RobertaModel.from_pretrained("roberta-base")
        self.txt.pooler = None  # avoid unused pooler warning
        self.vis = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.tp = nn.Linear(self.txt.config.hidden_size, 512)
        self.vp = nn.Linear(self.vis.config.hidden_size, 512)
        self.f = FUSE(512)
        self.h1 = HEAD(512, 2)  # T1
        self.h2 = HEAD(512, 3)  # T2
        self.h3t = HEAD(512, 2) # T3T (binary)  << FIXED
        self.h3s = HEAD(512, 3) # T3S
        self.h4 = HEAD(512, 3)  # T4

    def forward(self, B):
        t = self.txt(B["input_ids"], B["attention_mask"]).last_hidden_state[:, 0]
        v = self.vis(pixel_values=B["pixel_values"]).last_hidden_state.mean(1)
        z = self.f(self.tp(t), self.vp(v))
        return {
            "t1": self.h1(z),
            "t2": self.h2(z),
            "t3t": self.h3t(z),
            "t3s": self.h3s(z),
            "t4": self.h4(z)
        }

# ==================== LOSS (minority-aware) =======================

def focal_like_ce(logits, targets, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

def multi_task_loss(o, y, class_weights, device):
    """
    T1, T2 -> weighted CE
    T3T, T3S, T4 -> weighted focal-CE with higher task scaling
    """
    total = 0.0

    task_scales = {
        "t1": 1.0,
        "t2": 1.0,
        "t3t": 2.0,   # push hard on minority humanitarian types
        "t3s": 1.7,
        "t4": 1.3
    }

    for k in y:
        m = y[k] >= 0
        if not m.any():
            continue
        logits = o[k][m]
        targets = y[k][m]
        alpha = class_weights.get(k, None)
        alpha = alpha.to(device) if alpha is not None else None

        if k in ["t1", "t2"]:
            l = F.cross_entropy(logits, targets, weight=alpha, reduction="mean")
        else:
            l = focal_like_ce(logits, targets, alpha=alpha, gamma=2.0)

        total = total + task_scales[k] * l

    return total

# ========================= EVAL ON DEV ===========================

def evaluate(model, loader, device):
    model.eval()
    all_preds = {k:[] for k in ["t1","t2","t3t","t3s","t4"]}
    all_true  = {k:[] for k in ["t1","t2","t3t","t3s","t4"]}

    with torch.no_grad():
        for B, Y in loader:
            B = {k:v.to(device) for k,v in B.items()}
            out = model(B)
            for k in all_preds:
                mask = Y[k] >= 0
                if mask.any():
                    preds = out[k][mask].argmax(1).cpu().tolist()
                    labels = Y[k][mask].cpu().tolist()
                    all_preds[k] += preds
                    all_true[k]  += labels

    reports = {}
    print("\n================= DEV RESULTS =================\n")
    for k in all_preds:
        if len(all_true[k]) == 0:
            continue
        print(f"\n--- TASK {k.upper()} ---")
        rep_str = classification_report(all_true[k], all_preds[k], digits=4, zero_division=0)
        print(rep_str)
        acc = accuracy_score(all_true[k], all_preds[k])
        print(f"Accuracy: {acc:.4f}")
        reports[k] = classification_report(all_true[k], all_preds[k], output_dict=True, zero_division=0)

    main_score = 0.0
    if "t2" in reports:
        main_score = reports["t2"]["macro avg"]["f1-score"]
    elif "t1" in reports:
        main_score = reports["t1"]["macro avg"]["f1-score"]
    print(f"\n>>> Selected dev score (T2 macro F1) = {main_score:.4f}\n")
    return main_score

# ========================= TRAIN =============================

def train():
    train_rows, dev_rows = load_all()

    # ---- class stats + weights ----
    stats = compute_class_stats(train_rows)
    print("Class stats:", stats)

    class_weights = {
        "t1": make_weights(stats["t1"], 2),
        "t2": make_weights(stats["t2"], 3),
        "t3t": make_weights(stats["t3t"], 2),  # 2 classes for T3T
        "t3s": make_weights(stats["t3s"], 3),
        "t4": make_weights(stats["t4"], 3),
    }
    print("Class weights:")
    for k, v in class_weights.items():
        print(k, v.tolist())

    # ---- sampler weights ----
    sample_weights = build_sample_weights(train_rows, stats, max_mult=10.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tok = RobertaTokenizerFast.from_pretrained("roberta-base")
    proc = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    TL = DataLoader(
        CRISIS(train_rows, tok, proc),
        batch_size=8,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate
    )
    DL = DataLoader(
        CRISIS(dev_rows, tok, proc),
        batch_size=8,
        shuffle=False,
        collate_fn=collate
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n🟢 Using device:", device, "\n")

    model = MODEL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    epochs = 30
    total_steps = len(TL) * epochs
    warmup_steps = int(0.03 * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs("/kaggle/working/checkpoints_sota", exist_ok=True)
    best_score = -1.0
    best_path  = None

    for ep in range(epochs):
        model.train()
        total = 0.0

        for B, Y in tqdm(TL, desc=f"Epoch {ep+1}/{epochs}"):
            B = {k: v.to(device) for k, v in B.items()}
            Y = {k: v.to(device) for k, v in Y.items()}
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(B)
                loss = multi_task_loss(out, Y, class_weights=class_weights, device=device)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total += loss.item()

        print(f"\n🟣 Epoch {ep+1} Train Loss = {total/len(TL):.4f}\n")

        # ---- dev eval ----
        dev_score = evaluate(model, DL, device)

        # ---- save epoch checkpoint ----
        ep_path = f"/kaggle/working/checkpoints_sota/E{ep+1}.pt"
        torch.save(model.state_dict(), ep_path)
        print("💾 Saved epoch checkpoint:", ep_path)

        # ---- save best by T2 macro F1 ----
        if dev_score > best_score:
            best_score = dev_score
            best_path = "/kaggle/working/checkpoints_sota/best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New best model (score={best_score:.4f}) saved to:", best_path)

    print("\nTraining done. Best dev score =", best_score, "at", best_path)

# ========================= RUN =============================

train()
