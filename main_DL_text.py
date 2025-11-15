# model code and training loop 
# multimodal_damage_fixed.py
# Fixed + improved multimodal training from scratch:
#  - CLIP-style text + image encoders (from-scratch)
#  - Focal Loss for classification
#  - Cheap translation/paraphrase augmentation (on-the-fly) with optional googletrans fallback
#  - Strong image augmentations
#  - Notebook-safe (num_workers=0) and MPS/CUDA/CPU compatible
#  - Checkpoint saving

import os, re, random, json, math, time
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------- CONFIG -----------------------------
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-4
IMG_SIZE = 224
EMB_DIM = 512       # CLIP embedding dim
TEXT_DIM = 512
IMG_DIM = 512
CLS_HIDDEN = 256
MODEL_DIR = "./checkpoints_multimodal_fixed"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILES = [
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_dev.tsv",
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv",
]
TEST_FILES = [
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_test.tsv",
]

NUM_WORKERS = 0  # keep 0 for notebooks / macOS/MPS
PIN_MEMORY = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# -------------------------- REPRO --------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

# ------------------------- UTILITIES ------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# cheap synonym map (very small fallback) — you can expand
SYNONYM_MAP = {
    "car": ["vehicle", "auto"],
    "house": ["home", "residence"],
    "flood": ["inundation"],
    "fire": ["blaze"],
    "collapsed": ["fell", "caved"],
    "help": ["assist"],
}

def cheap_paraphrase(tokens, p_del=0.15, p_swap=0.1, p_syn=0.08, p_char_noise=0.03):
    # token-level deletion
    toks = [t for t in tokens if (random.random() > p_del or t in ["<cls>"])]
    # random swap occasionally
    if len(toks) >= 2 and random.random() < p_swap:
        i,j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    # synonym replacement
    for i,t in enumerate(toks):
        if random.random() < p_syn and t in SYNONYM_MAP:
            toks[i] = random.choice(SYNONYM_MAP[t])
    # char noise
    if random.random() < p_char_noise and len(toks) > 0:
        i = random.randrange(len(toks))
        w = toks[i]
        pos = random.randrange(max(1, len(w)))
        toks[i] = w[:pos] + random.choice("aeiou") + w[pos:]
    return toks

# optional googletrans fallback if installed (used only if available)
TRANSLATOR = None
TRANSLATOR_NAME = None
try:
    from googletrans import Translator as GoogleTranslator  # type: ignore
    TRANSLATOR = GoogleTranslator()
    TRANSLATOR_NAME = "googletrans"
    print("googletrans available for optional BT (will be used if ENABLE_BT=True).")
except Exception:
    TRANSLATOR = None
    TRANSLATOR_NAME = None

def cheap_backtranslate(text, langs_chain=("fr","de")):
    # If googletrans available, do quick roundtrip through small chain; else fallback to cheap paraphrase
    if TRANSLATOR is not None:
        try:
            cur = text
            for lang in langs_chain:
                cur = TRANSLATOR.translate(cur, dest=lang).text
            cur = TRANSLATOR.translate(cur, dest="en").text
            return clean_text(cur)
        except Exception:
            pass
    # fallback: token-level paraphrase
    toks = re.findall(r"\b[\w']+\b", text.lower())
    toks2 = cheap_paraphrase(toks, p_del=0.12, p_swap=0.08, p_syn=0.1, p_char_noise=0.02)
    return " ".join(toks2)

# ------------------------- DATA ------------------------------
def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df["tweet_text"] = df["tweet_text"].astype(str).apply(clean_text)
    return df

train_df = pd.concat([load_tsv(p) for p in TRAIN_FILES], ignore_index=True)
test_df = pd.concat([load_tsv(p) for p in TEST_FILES], ignore_index=True)

label_le = LabelEncoder()
train_df["label_id"] = label_le.fit_transform(train_df["label"])
test_df["label_id"] = label_le.transform(test_df["label"])
n_classes = len(label_le.classes_)
print("Classes:", label_le.classes_.tolist())

# ---------------------- VOCAB & TOKENIZER ---------------------
def basic_tokenizer(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenizer(t))
    vocab = {"<unk>":0, "<pad>":1, "<cls>":2}
    for w,cnt in counter.items():
        if cnt >= min_freq:
            vocab[w] = len(vocab)
    return vocab

MIN_FREQ = 2
vocab = build_vocab(train_df["tweet_text"].tolist(), min_freq=MIN_FREQ)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)
with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

def encode_text_ids(text, max_len=MAX_LEN):
    toks = basic_tokenizer(text)[: max_len - 1]
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
    ids = [vocab["<cls>"]] + ids
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

# --------------------- IMAGE AUGMENTATION ---------------------
# Stronger image augmentations: RandomResizedCrop, ColorJitter, RandAug-ish, RandomErasing
img_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

img_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# --------------------- DATASET -------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, df, do_aug=False, enable_bt=False, bt_prob=0.05):
        self.df = df.reset_index(drop=True)
        self.do_aug = do_aug
        self.enable_bt = enable_bt
        self.bt_prob = bt_prob
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["tweet_text"]
        # cheap on-the-fly augmentations (paraphrase) for text (cheaper than full BT)
        if self.do_aug:
            # small chance to run expensive-ish BT if enabled and translator exists
            if self.enable_bt and TRANSLATOR is not None and random.random() < self.bt_prob:
                text = cheap_backtranslate(text)
            else:
                # cheap paraphrase
                toks = basic_tokenizer(text)
                toks = cheap_paraphrase(toks, p_del=0.12, p_swap=0.08, p_syn=0.06, p_char_noise=0.02)
                text = " ".join(toks)
        text_ids = torch.tensor(encode_text_ids(text), dtype=torch.long)

        # image load + transformation
        img_path = row.get("image", None)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))

        img = img_train_transform(img) if self.do_aug else img_test_transform(img)

        label = torch.tensor(row["label_id"], dtype=torch.long)
        return {"text_ids": text_ids, "image": img, "label": label}

# Create datasets and loaders
train_dataset = MultimodalDataset(train_df, do_aug=True, enable_bt=False, bt_prob=0.05)
test_dataset = MultimodalDataset(test_df, do_aug=False, enable_bt=False)

# Weighted sampler to mitigate class imbalance (still reasonable)
counts = train_df["label_id"].value_counts().sort_index()
weights_per_class = 1.0 / torch.sqrt(torch.tensor(counts.values, dtype=torch.float) + 1e-6)
sample_weights = [weights_per_class[int(l)] for l in train_df["label_id"].tolist()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ----------------------- MODEL: CLIP-style encoders -----------------------
# Text encoder: small Transformer stack
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, key_padding_mask=mask)[0]
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=TEXT_DIM, n_head=8, n_layers=4, d_ff=1024, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, EMB_DIM)
    def forward(self, ids):
        mask = ids.eq(1)  # padding index
        x = self.emb(ids) + self.pos[:, :ids.size(1), :]
        for l in self.layers:
            x = l(x, mask)
        # pool via first token (cls-like) then project
        pooled = self.norm(x[:,0])
        out = self.project(pooled)   # EMB_DIM
        out = F.normalize(out, dim=-1)
        return out

# Image encoder: small convnet -> projection (acts like CLIP visual encoder)
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=EMB_DIM):
        super().__init__()
        # small efficient conv backbone
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(256, out_dim)
    def forward(self, x):
        f = self.net(x).view(x.size(0), -1)
        out = self.proj(f)
        out = F.normalize(out, dim=-1)
        return out

# Fusion classifier: uses concatenated CLIP embeddings (text+img) and classification head
class MultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.text_enc = TextEncoder(vocab_size)
        self.img_enc = ImageEncoder()
        # classification head
        self.cls = nn.Sequential(
            nn.Linear(EMB_DIM*2, CLS_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(CLS_HIDDEN, n_classes)
        )
        # optional temperature for contrastive (not used for classification)
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, ids, images):
        t = self.text_enc(ids)
        v = self.img_enc(images)
        # both t and v are normalized
        fused = torch.cat([t, v], dim=1)
        logits = self.cls(fused)
        return logits, t, v

model = MultimodalClassifier(vocab_size=vocab_size, n_classes=n_classes).to(DEVICE)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

# ------------------------- LOSSES -------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            # alpha can be passed as CPU tensor; move to logits device
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal = focal * at
        return focal.mean()

# compute class alpha from training counts (inverse sqrt)
class_counts = train_df["label_id"].value_counts().sort_index().values
alpha = (1.0 / np.sqrt(class_counts + 1e-6))
alpha = alpha / alpha.sum() * len(alpha)  # normalize to mean 1
focal_criterion = FocalLoss(gamma=2.0, alpha=alpha)

# optional contrastive loss between text and image (not required, but good to align)
class NTXent(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.t = temp
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # 2B x D
        sim = torch.matmul(z, z.t()) / self.t
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        positives = torch.exp((z1 * z2).sum(dim=1)/self.t)
        positives = torch.cat([positives, positives], dim=0)
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log(positives / denom)
        return loss.mean()

ntxent = NTXent(temp=0.07)

# ----------------------- OPTIMIZER / SCHEDULER -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------------ EVALUATION FUNC ----------------------------
def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for b in loader:
            ids = b["text_ids"].to(DEVICE)
            imgs = b["image"].to(DEVICE)
            y = b["label"].to(DEVICE)
            logits, _, _ = model(ids, imgs)
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    return acc, trues, preds

# ------------------------- TRAIN LOOP -------------------------------
best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    tot_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        ids = batch["text_ids"].to(DEVICE)
        imgs = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits, t_emb, v_emb = model(ids, imgs)

        # classification loss with FocalLoss
        cls_loss = focal_criterion(logits, y)

        # optional auxiliary contrastive loss to align modalities (small weight)
        with torch.no_grad():
            # no grad: compute normalized embeddings only for stable training
            pass
        contra_loss = ntxent(t_emb, v_emb) * 0.1

        loss = cls_loss + contra_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += float(loss.item())
        pbar.set_postfix(loss=tot_loss / (pbar.n + 1e-12))

    scheduler.step()

    val_acc, val_trues, val_preds = evaluate(model, test_loader)
    print(f"\nEpoch {epoch} | Val Acc: {val_acc:.4f}")
    print(classification_report(val_trues, val_preds, target_names=label_le.classes_, zero_division=0))

    # save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocab,
            "epoch": epoch
        }, os.path.join(MODEL_DIR, "best_multimodal_fixed.pt"))
        print("Saved best model.")

print("Training complete. Best val acc:", best_acc)


#testing
# ============================================================
#  test_damagenet_text_final.py
#  Standalone testing script for DamageNetTextFromScratch
# ============================================================

import os, re, json, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# CONFIG (EDIT THESE)
# ------------------------------------------------------------
VOCAB_PATH = "/Volumes/Extreme SSD/DL_Proj/damagenet_text_minority_ckpt/vocab.json"
CHECKPOINT_PATH = "/Volumes/Extreme SSD/DL_Proj/damagenet_text_minority_ckpt/best_minority.pt"

TEST_FILE = "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_test.tsv"

MAX_LEN = 128
EMB_DIM = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1
BATCH_SIZE = 32

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("DEVICE:", DEVICE)


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def basic_tokenizer(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def encode_text_ids_from_tokens(tokens, vocab, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:max_len - 1]]
    ids = [vocab["<cls>"]] + ids
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

# ------------------------------------------------------------
# LOAD VOCAB
# ------------------------------------------------------------

print("Loading vocab:", VOCAB_PATH)
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------------------------------------------------------
# LOAD TEST TSV
# ------------------------------------------------------------

print("Loading test file:", TEST_FILE)
df = pd.read_csv(TEST_FILE, sep="\t")
df["tweet_text"] = df["tweet_text"].astype(str).apply(clean_text)

label_le = LabelEncoder()
df["label_id"] = label_le.fit_transform(df["label"].astype(str))
n_classes = len(label_le.classes_)
print("CLASSES:", label_le.classes_.tolist())


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        toks = basic_tokenizer(row["tweet_text"])
        ids = encode_text_ids_from_tokens(toks, self.vocab)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long)
        }

test_dataset = TestDataset(df, vocab)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ------------------------------------------------------------
# MODEL ARCHITECTURE (MATCHES TRAINING)
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        out = self.attn(x, x, x, key_padding_mask=mask)[0]
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, EMB_DIM))
        self.layers = nn.ModuleList([
            TransformerBlock(EMB_DIM, N_HEADS, FF_DIM, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.norm = nn.LayerNorm(EMB_DIM)

    def forward(self, ids):
        mask = ids.eq(0)
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x[:, 0])

class DamageNetTextFromScratch(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.encoder = TextEncoder(vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(EMB_DIM // 2, n_classes)
        )

    def forward(self, ids, logit_adjust=None):
        z = self.encoder(ids)
        logits = self.classifier(z)
        return logits + logit_adjust if logit_adjust is not None else logits


# ------------------------------------------------------------
# LOAD CHECKPOINT (HANDLES ALL FORMATS)
# ------------------------------------------------------------

print("Loading checkpoint:", CHECKPOINT_PATH)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = DamageNetTextFromScratch(vocab_size, n_classes).to(DEVICE)

if isinstance(ckpt, dict) and "model_state" in ckpt:
    print("Checkpoint format: {model_state: ...}")
    model.load_state_dict(ckpt["model_state"])
    logit_adjust = ckpt.get("logit_adjust", None)

elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("classifier") for k in ckpt.keys()):
    print("Checkpoint format: raw state_dict dict")
    model.load_state_dict(ckpt)
    logit_adjust = None

else:
    print("Checkpoint format: raw tensor-only state_dict")
    model.load_state_dict(ckpt)
    logit_adjust = None

if logit_adjust is not None:
    logit_adjust = torch.tensor(logit_adjust, dtype=torch.float32, device=DEVICE)
    print("Loaded logit adjustment.")
else:
    print("No logit adjustment found.")


# ------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------

print("\nRunning inference...")
model.eval()
preds = []
trues = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        ids = batch["input_ids"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        logits = model(ids, logit_adjust)
        pred = logits.argmax(1)

        preds.extend(pred.cpu().tolist())
        trues.extend(y.cpu().tolist())


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

acc = accuracy_score(trues, preds)

print("\n================= TEST RESULTS =================")
print(f"Accuracy: {acc:.4f}\n")

print(classification_report(
    trues,
    preds,
    target_names=label_le.classes_,
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(trues, preds))

print("\n================================================\n")



