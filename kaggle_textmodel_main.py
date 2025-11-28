##############################################################
#   5 × SPECIALIZED MODELS FOR ITSACRISIS (KAGGLE, CUDA)     #
#   - Separate models for: t1, t2, t3t, t3s, t4              #
#   - CrisisMMD text+image                                  #
#   - Checkpoints: E{epoch}_NEW_FINAL.pt                    #
#   - Uploads to: yathnehr/itsacrisis-checkpoints           #
##############################################################

import os, glob, warnings
from collections import Counter

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    RobertaModel,
    RobertaTokenizerFast,
    SwinModel,
    get_cosine_schedule_with_warmup,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# Kaggle dataset upload
try:
    import kagglehub
except ImportError:
    kagglehub = None

##############################################################
# PATHS / GLOBALS
##############################################################

BASE = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"
IMG_BASE = "/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES"
KAGGLE_DATASET_HANDLE = "yatharthnehra/crisis-epochs"

DATA = {
    "damage": {
        "train": f"{BASE}/task_damage_text_img_train.tsv",
        "dev":   f"{BASE}/task_damage_text_img_dev.tsv",
    },
    "humanitarian": {
        "train": f"{BASE}/task_humanitarian_text_img_train.tsv",
        "dev":   f"{BASE}/task_humanitarian_text_img_dev.tsv",
    },
    "informative": {
        "train": f"{BASE}/task_informative_text_img_train.tsv",
        "dev":   f"{BASE}/task_informative_text_img_dev.tsv",
    },
}

NUM_CLASSES = {
    "t1":  2,
    "t2":  3,
    "t3t": 2,
    "t3s": 3,
    "t4":  3,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🟢 Using device: {device.upper()}\n")

##############################################################
# IMAGE INDEX (GLOBAL)
##############################################################

print("🔍 Indexing images...")
IMAGE_INDEX = {}
for p in tqdm(glob.glob(f"{IMG_BASE}/**/*.jpg", recursive=True)):
    f = os.path.basename(p)
    if f.startswith("._"):
        continue
    key = f.rsplit(".jpg", 1)[0]
    IMAGE_INDEX[key] = p

print(f"\n📦 Total images indexed: {len(IMAGE_INDEX)}\n")

##############################################################
# TSV LOADING & MERGED ROWS
##############################################################

def read_tsv(path):
    with open(path, "r", encoding="utf8") as f:
        hdr = f.readline().strip().split("\t")
        rows = [l.strip().split("\t") for l in f]
    return hdr, rows

def find(h, cands):
    for x in cands:
        if x in h:
            return x
    return h[0]

# label maps (same as your previous script)
DMAP = {
    "little_or_no_damage": 0,
    "mild_damage":         1,
    "severe_damage":       2,
}

T2MAP = {
    "not_humanitarian":           0,
    "other_relevant_information": 0,
    "affected_individuals":       1,
    "injured_or_dead_people":     1,
    "missing_or_found_people":    1,
    "rescue_volunteering_or_donation_effort": 1,
    "infrastructure_and_utility_damage":      2,
    "vehicle_damage":             2,
}

T3TYPE = {
    "infrastructure_and_utility_damage": 0,
    "vehicle_damage":                    1,
}

T4MAP = {
    "affected_individuals":             0,
    "injured_or_dead_people":           0,
    "missing_or_found_people":          0,
    "rescue_volunteering_or_donation_effort": 1,
    "other_relevant_information":       2,
    "not_humanitarian":                 2,
    "infrastructure_and_utility_damage":2,
    "vehicle_damage":                   2,
}

def load_all():
    TRAIN, DEV = [], []

    for task, paths in DATA.items():
        for split in ["train", "dev"]:
            hdr, rows = read_tsv(paths[split])

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
                    "t4": -1,
                }

                lab = d[LAB].lower() if LAB in d and d[LAB] else ""

                if task == "informative":
                    item["t1"] = 1 if lab == "informative" else 0

                if task == "damage":
                    item["t3s"] = DMAP.get(lab, -1)

                if task == "humanitarian":
                    item["t2"]  = T2MAP.get(lab, -1)
                    item["t3t"] = T3TYPE.get(lab, -1)
                    item["t4"]  = T4MAP.get(lab, -1)

                if split == "train":
                    TRAIN.append(item)
                else:
                    DEV.append(item)

    print(f"\n📊 Rows: {len(TRAIN)} train  |  {len(DEV)} dev\n")
    return TRAIN, DEV

##############################################################
# CLASS STATS & WEIGHTS
##############################################################

def compute_class_stats(train_rows):
    stats = {k: Counter() for k in ["t1", "t2", "t3t", "t3s", "t4"]}
    for r in train_rows:
        for k in stats:
            v = r[k]
            if v >= 0:
                stats[k][v] += 1
    return stats

def make_weights(counter, num_classes):
    freqs = torch.tensor([counter.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    freqs = freqs + 1.0
    inv = 1.0 / torch.sqrt(freqs)
    inv = inv / inv.mean()
    return inv

##############################################################
# DATASET & COLLATE
##############################################################

BASE_IMG_T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

AUG_IMG_T = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class CRISIS(Dataset):
    def __init__(self, data, tokenizer, train=True):
        self.data = data
        self.tok = tokenizer
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]

        try:
            img = Image.open(d["img"]).convert("RGB") if d["img"] else Image.new("RGB", (224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224))

        if self.train and torch.rand(1).item() < 0.4:
            img = AUG_IMG_T(img)
        else:
            img = BASE_IMG_T(img)

        T = self.tok(
            d["tweet"],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        X = {
            "input_ids":      T.input_ids[0],
            "attention_mask": T.attention_mask[0],
            "pixel_values":   img,
        }
        Y = {
            "t1":  torch.tensor(d["t1"],  dtype=torch.long),
            "t2":  torch.tensor(d["t2"],  dtype=torch.long),
            "t3t": torch.tensor(d["t3t"], dtype=torch.long),
            "t3s": torch.tensor(d["t3s"], dtype=torch.long),
            "t4":  torch.tensor(d["t4"],  dtype=torch.long),
        }
        return X, Y

def collate(b):
    X, Y = zip(*b)
    batch_x = {k: torch.stack([x[k] for x in X]) for k in X[0]}
    batch_y = {k: torch.stack([y[k] for y in Y]) for k in Y[0]}
    return batch_x, batch_y

##############################################################
# FUSION BACKBONE
##############################################################

class FUSE(nn.Module):
    def __init__(self, d=512, layers=2, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            batch_first=True,
            dim_feedforward=d * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d))

    def forward(self, t, v):
        B = t.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, t[:, None], v[:, None]], dim=1)
        out = self.enc(seq)
        return out[:, 0]

class TextImageBackbone(nn.Module):
    def __init__(self, d_model=512, fuse_layers=2, heads=8, txt_drop=0.1, vis_drop=0.1):
        super().__init__()
        self.txt = RobertaModel.from_pretrained("roberta-base")
        self.txt.pooler = None
        self.vis = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        self.txt_drop = nn.Dropout(txt_drop)
        self.vis_drop = nn.Dropout(vis_drop)

        self.tp = nn.Linear(self.txt.config.hidden_size, d_model)
        self.vp = nn.Linear(self.vis.config.hidden_size, d_model)

        self.fuse = FUSE(d=d_model, layers=fuse_layers, heads=heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, B):
        txt_out = self.txt(B["input_ids"], B["attention_mask"])
        t_cls = txt_out.last_hidden_state[:, 0]
        t_cls = self.txt_drop(t_cls)

        vis_out = self.vis(pixel_values=B["pixel_values"])
        v = vis_out.last_hidden_state.mean(1)
        v = self.vis_drop(v)

        t_proj = self.tp(t_cls)
        v_proj = self.vp(v)

        z = self.fuse(t_proj, v_proj)
        z = self.norm(z)
        return z

##############################################################
# TASK-SPECIFIC MODELS
##############################################################

class HEAD(nn.Module):
    def __init__(self, d, o, hidden=256, drop=0.2):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, o),
        )

    def forward(self, x):
        return self.m(x)

# T1: informative (binary) – simple, strong regularization
class ModelT1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextImageBackbone(d_model=384, fuse_layers=1, heads=6)
        self.head = HEAD(384, 2, hidden=256, drop=0.3)

    def forward(self, B):
        z = self.backbone(B)
        return self.head(z)

# T2: humanitarian multi-class
class ModelT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextImageBackbone(d_model=512, fuse_layers=2, heads=8)
        self.head = HEAD(512, 3, hidden=384, drop=0.3)

    def forward(self, B):
        z = self.backbone(B)
        return self.head(z)

# T3T: infra vs vehicle (binary) – harder, use deeper fusion
class ModelT3T(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextImageBackbone(d_model=512, fuse_layers=3, heads=8)
        self.gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
        )
        self.head = HEAD(512, 2, hidden=256, drop=0.3)

    def forward(self, B):
        z = self.backbone(B)
        z = self.gate(z)
        return self.head(z)

# T3S: damage severity (3-way) – hardest, deeper head + focal loss later
class ModelT3S(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextImageBackbone(d_model=512, fuse_layers=3, heads=8)
        self.gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.head = HEAD(512, 3, hidden=384, drop=0.35)

    def forward(self, B):
        z = self.backbone(B)
        z = self.gate(z)
        return self.head(z)

# T4: aggregated humanitarian (3-way)
class ModelT4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextImageBackbone(d_model=448, fuse_layers=2, heads=7)
        self.head = HEAD(448, 3, hidden=320, drop=0.3)

    def forward(self, B):
        z = self.backbone(B)
        return self.head(z)

##############################################################
# LOSSES
##############################################################

def focal_loss(logits, targets, weight=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

##############################################################
# GENERIC TRAIN / EVAL FOR ONE TASK
##############################################################

def train_single_task(
    task_name,
    model_class,
    train_loader,
    dev_loader,
    num_classes,
    class_weights=None,
    epochs=10,
    use_focal=False,
    lr=3e-5,
    weight_decay=1e-2,
):
    print(f"\n====================================================")
    print(f"🚀 TRAINING MODEL FOR TASK: {task_name.upper()}")
    print(f"====================================================\n")

    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_macro_f1 = -1.0

    for ep in range(1, epochs + 1):
        ##################################################
        # TRAIN
        ##################################################
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{task_name.upper()} Epoch {ep}/{epochs}")
        for B, Y in pbar:
            B = {k: v.to(device) for k, v in B.items()}
            y = Y[task_name]  # (batch,)
            mask = (y >= 0) & (y < num_classes)
            if not mask.any():
                continue

            y = y[mask].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(B)
                logits = logits[mask]

                w = None
                if class_weights is not None:
                    w = class_weights.to(device)

                if use_focal:
                    loss = focal_loss(logits, y, weight=w, gamma=2.5)
                else:
                    loss = F.cross_entropy(logits, y, weight=w)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"\n🟣 {task_name.upper()} Epoch {ep} Train Loss = {avg_loss:.4f}")

        ##################################################
        # EVAL
        ##################################################
        model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for B, Y in dev_loader:
                B = {k: v.to(device) for k, v in B.items()}
                y = Y[task_name]
                mask = (y >= 0) & (y < num_classes)
                if not mask.any():
                    continue
                y = y[mask]

                logits = model(B)
                logits = logits[mask]
                preds = logits.argmax(dim=1).cpu()

                all_true.extend(y.tolist())
                all_pred.extend(preds.tolist())

        print("\n===================== EVAL =====================\n")
        if len(all_true) == 0:
            print(f"No labels for task {task_name}, skipping eval.\n")
            macro_f1 = 0.0
        else:
            report = classification_report(all_true, all_pred, digits=4, zero_division=0)
            cm = confusion_matrix(all_true, all_pred)
            macro_f1 = f1_score(all_true, all_pred, average="macro")
            acc = accuracy_score(all_true, all_pred)

            print(report)
            print("Confusion Matrix (rows=true, cols=pred):")
            print(cm)
            print(f"\nMacro-F1: {macro_f1:.4f} | Accuracy: {acc:.4f}")

        ##################################################
        # SAVE & UPLOAD
        ##################################################
        filename = f"E{ep}_NEW_FINAL.pt"
        local_path = f"/kaggle/working/{filename}"
        torch.save(model.state_dict(), local_path)
        print(f"\n💾 Saved checkpoint → {local_path}")

        if kagglehub is not None:
            try:
                kagglehub.dataset_upload(KAGGLE_DATASET_HANDLE, "/kaggle/working/")
                print(f"📤 Uploaded {filename} to Kaggle dataset: {KAGGLE_DATASET_HANDLE}\n")
            except Exception as e:
                print("⚠ Kaggle upload failed:", e, "\n")
        else:
            print("⚠ kagglehub not available, skipping upload\n")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            print(f"🏆 New best {task_name.upper()} macro-F1: {best_macro_f1:.4f}\n")

    print(f"\n✅ Finished training task {task_name.upper()} | Best macro-F1: {best_macro_f1:.4f}\n")


##############################################################
# MAIN: LOAD DATA, BUILD LOADERS, TRAIN ALL 5 MODELS
##############################################################

def main():
    train_rows, dev_rows = load_all()

    stats = compute_class_stats(train_rows)
    print("Class stats per task:")
    for k, v in stats.items():
        print(f"  {k}: {dict(v)}")

    class_weights = {
        "t1":  make_weights(stats["t1"],  NUM_CLASSES["t1"])  if stats["t1"]  else None,
        "t2":  make_weights(stats["t2"],  NUM_CLASSES["t2"])  if stats["t2"]  else None,
        "t3t": make_weights(stats["t3t"], NUM_CLASSES["t3t"]) if stats["t3t"] else None,
        "t3s": make_weights(stats["t3s"], NUM_CLASSES["t3s"]) if stats["t3s"] else None,
        "t4":  make_weights(stats["t4"],  NUM_CLASSES["t4"])  if stats["t4"]  else None,
    }

    print("\nClass weights:")
    for k, w in class_weights.items():
        print(f"  {k}: {None if w is None else w.tolist()}")

    # sampling weights: give more weight to rare damage / humanitarian
    sample_weights = []
    for r in train_rows:
        w = 1.0
        if r["t3t"] >= 0:
            w *= 3.0
        if r["t3s"] >= 0:
            w *= 4.0
        if r["t2"] == 2:
            w *= 2.0
        sample_weights.append(w)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tok = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_ds = CRISIS(train_rows, tok, train=True)
    dev_ds   = CRISIS(dev_rows,   tok, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # epochs per task (more for harder ones)
    epochs_cfg = {
        "t1": 10,
        "t2": 12,
        "t3t": 18,
        "t3s": 22,
        "t4": 12,
    }

    # T1
    train_single_task(
        task_name="t1",
        model_class=ModelT1,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_classes=NUM_CLASSES["t1"],
        class_weights=class_weights["t1"],
        epochs=epochs_cfg["t1"],
        use_focal=False,
    )

    # T2
    train_single_task(
        task_name="t2",
        model_class=ModelT2,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_classes=NUM_CLASSES["t2"],
        class_weights=class_weights["t2"],
        epochs=epochs_cfg["t2"],
        use_focal=False,
    )

    # T3T (hard) – use focal loss
    train_single_task(
        task_name="t3t",
        model_class=ModelT3T,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_classes=NUM_CLASSES["t3t"],
        class_weights=class_weights["t3t"],
        epochs=epochs_cfg["t3t"],
        use_focal=True,
    )

    # T3S (hardest) – use focal loss
    train_single_task(
        task_name="t3s",
        model_class=ModelT3S,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_classes=NUM_CLASSES["t3s"],
        class_weights=class_weights["t3s"],
        epochs=epochs_cfg["t3s"],
        use_focal=True,
    )

    # T4
    train_single_task(
        task_name="t4",
        model_class=ModelT4,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_classes=NUM_CLASSES["t4"],
        class_weights=class_weights["t4"],
        epochs=epochs_cfg["t4"],
        use_focal=False,
    )

    print("\n🔥 ALL TASKS TRAINED SEQUENTIALLY. DONE.\n")

if __name__ == "__main__":
    main()
