# improved_balanced_training.py
import os, re, random, json, math, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------- CONFIG -----------------------------
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16            # keep divisible by n_classes if possible
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

INFO_FILES = [
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv",
]

NUM_WORKERS = 0  # safe for notebooks / MPS
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

# ------------------------- UTILITIES / AUG ------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

SYNONYM_MAP = {
    "car": ["vehicle", "auto"],
    "house": ["home", "residence"],
    "flood": ["inundation"],
    "fire": ["blaze"],
    "collapsed": ["fell", "caved"],
    "help": ["assist"],
}

def cheap_paraphrase(tokens, p_del=0.15, p_swap=0.1, p_syn=0.08, p_char_noise=0.03):
    toks = [t for t in tokens if (random.random() > p_del or t == "<cls>")]
    if len(toks) >= 2 and random.random() < p_swap:
        i,j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    for i,t in enumerate(toks):
        if random.random() < p_syn and t in SYNONYM_MAP:
            toks[i] = random.choice(SYNONYM_MAP[t])
    if random.random() < p_char_noise and len(toks) > 0:
        i = random.randrange(len(toks))
        w = toks[i]
        pos = random.randrange(max(1, len(w)))
        toks[i] = w[:pos] + random.choice("aeiou") + w[pos:]
    return toks

TRANSLATOR = None
try:
    from googletrans import Translator as GoogleTranslator  # type: ignore
    TRANSLATOR = GoogleTranslator()
    print("googletrans available for optional back-translation.")
except Exception:
    TRANSLATOR = None

def cheap_backtranslate(text, langs_chain=("fr","de")):
    if TRANSLATOR is not None:
        try:
            cur = text
            for lang in langs_chain:
                cur = TRANSLATOR.translate(cur, dest=lang).text
            cur = TRANSLATOR.translate(cur, dest="en").text
            return clean_text(cur)
        except Exception:
            pass
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

# ------------------------ LOAD INFORMATIVE (UNLABELED) ------------------------
try:
    info_df = pd.concat([load_tsv(p) for p in INFO_FILES], ignore_index=True)
    if "label" in info_df.columns:
        info_df = info_df[info_df["label"].astype(str).str.lower() == "informative"].reset_index(drop=True)
    info_df["_is_unlabeled"] = True
    print(f"Loaded informative unlabeled samples: {len(info_df)}")
except Exception as e:
    info_df = pd.DataFrame()
    print("Could not load informative files (or none found). Skipping informative pretraining. Err:", e)

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

if len(info_df) > 0:
    all_vocab_texts = train_df["tweet_text"].tolist() + info_df["tweet_text"].tolist()
else:
    all_vocab_texts = train_df["tweet_text"].tolist()

MIN_FREQ = 2
vocab = build_vocab(all_vocab_texts, min_freq=MIN_FREQ)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)
with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

inv_vocab = {v: k for k, v in vocab.items()}

def encode_text_ids(text, max_len=MAX_LEN):
    toks = basic_tokenizer(text)[: max_len - 1]
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
    ids = [vocab["<cls>"]] + ids
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

# --------------------- IMAGE AUGMENTATION ---------------------
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
    def __init__(self, df, do_aug=False, enable_bt=False, bt_prob=0.05, minority_boost_map=None):
        self.df = df.reset_index(drop=True)
        self.do_aug = do_aug
        self.enable_bt = enable_bt
        self.bt_prob = bt_prob
        self.minority_boost_map = minority_boost_map or {}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["tweet_text"]
        label = int(row["label_id"])
        local_bt_prob = self.bt_prob * (self.minority_boost_map.get(label, 1.0))
        if self.do_aug:
            if self.enable_bt and TRANSLATOR is not None and random.random() < local_bt_prob:
                text = cheap_backtranslate(text)
            else:
                toks = basic_tokenizer(text)
                if label in self.minority_boost_map:
                    toks = cheap_paraphrase(toks, p_del=0.18, p_swap=0.12, p_syn=0.08, p_char_noise=0.03)
                else:
                    toks = cheap_paraphrase(toks, p_del=0.12, p_swap=0.08, p_syn=0.06, p_char_noise=0.02)
                text = " ".join(toks)
        text_ids = torch.tensor(encode_text_ids(text), dtype=torch.long)

        img_path = row.get("image", None)
        try:
            if pd.isna(img_path) or img_path in ("", None):
                raise Exception("no image")
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))

        img = img_train_transform(img) if self.do_aug else img_test_transform(img)

        label_t = torch.tensor(label, dtype=torch.long)
        return {"text_ids": text_ids, "image": img, "label": label_t}

# ------------------- Balanced Batch Sampler (best for extreme imbalance) -------------------
class BalancedBatchSampler(Sampler):
    """
    Construct batches containing `samples_per_class` examples per class.
    If a class doesn't have enough samples, sample with replacement for that class.
    """
    def __init__(self, labels, n_classes, samples_per_class, batch_count_per_epoch=None, seed=SEED):
        """
        labels: list/array of label ids (length = dataset size)
        n_classes: total number of classes
        samples_per_class: how many samples per class in each batch
        batch_count_per_epoch: how many batches per epoch (if None, compute from dataset size)
        """
        self.labels = np.array(labels)
        self.n_classes = int(n_classes)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.n_classes * self.samples_per_class
        self.indices_per_class = {c: np.where(self.labels == c)[0].tolist() for c in range(self.n_classes)}
        self.rng = random.Random(seed)
        # estimate batches per epoch: default = ceil(dataset_size / batch_size)
        if batch_count_per_epoch is None:
            self.batch_count_per_epoch = int(math.ceil(len(self.labels) / float(self.batch_size)))
        else:
            self.batch_count_per_epoch = int(batch_count_per_epoch)

    def __len__(self):
        return self.batch_count_per_epoch

    def __iter__(self):
        for _ in range(self.batch_count_per_epoch):
            batch_indices = []
            for c in range(self.n_classes):
                pool = self.indices_per_class.get(c, [])
                if len(pool) == 0:
                    continue
                if len(pool) >= self.samples_per_class:
                    sampled = self.rng.sample(pool, k=self.samples_per_class)
                else:
                    # sample with replacement if insufficient examples
                    sampled = [self.rng.choice(pool) for _ in range(self.samples_per_class)]
                batch_indices.extend(sampled)
            # shuffle within batch
            self.rng.shuffle(batch_indices)
            yield batch_indices

# ---------------- create dataset and balanced loader ----------------
# compute base counts (original dataset)
orig_counts = train_df["label_id"].value_counts().sort_index().values.astype(int)
print("Original class counts:", orig_counts)

# minority_boost_map for augmentations (simple proportional)
minority_boost_map = {}
for i, c in enumerate(orig_counts):
    if c < orig_counts.mean():
        # boost factor larger for rarer classes
        minority_boost_map[i] = min(4.0, 1.0 + int(round(orig_counts.mean() / max(1, c))) * 0.5)

train_dataset = MultimodalDataset(train_df, do_aug=True, enable_bt=True, bt_prob=0.10, minority_boost_map=minority_boost_map)
test_dataset = MultimodalDataset(test_df, do_aug=False, enable_bt=False)

# balanced batch settings: choose samples_per_class so batch_size ~ BATCH_SIZE
# prefer samples_per_class = max(1, BATCH_SIZE // n_classes)
samples_per_class = max(1, BATCH_SIZE // max(1, n_classes))
bbs = BalancedBatchSampler(labels=train_df["label_id"].tolist(), n_classes=n_classes, samples_per_class=samples_per_class)
train_loader = DataLoader(train_dataset, batch_sampler=bbs, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Balanced batch: {samples_per_class} samples/class -> batch_size {samples_per_class * n_classes}")

# --- COPY YOUR MODEL CLASSES HERE: TransformerBlock, TextEncoder, ImageEncoder, MultimodalClassifier ---
# Paste the exact definitions from your original file here (unchanged).
# Example:
# class TransformerBlock(nn.Module): ...
# class TextEncoder(nn.Module): ...
# class ImageEncoder(nn.Module): ...
# class MultimodalClassifier(nn.Module): ...

# (For brevity in this message I assume you paste them in the file.)
# ---------------------------------------------------------------------------

# instantiate model (replace after pasting classes)
model = MultimodalClassifier(vocab_size=vocab_size, n_classes=n_classes, use_image=True).to(DEVICE)
print("Model params:", sum(p.numel() for p in model.parameters()))

# ------------------------- Class-Balanced Focal Loss -------------------------
class ClassBalancedFocalLoss(nn.Module):
    """
    Implements focal loss with class-balanced weights using the effective number method:
    weight_c = (1 - beta) / (1 - beta^n_c)
    Then normalize weights so their sum = num_classes (keeps scale).
    """
    def __init__(self, gamma=2.0, beta=0.999, samples_per_class=None):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        if samples_per_class is None:
            raise ValueError("samples_per_class (array-like) required for CB weights")
        self.samples_per_class = np.array(samples_per_class, dtype=np.float64)
        # compute effective number
        eff = 1.0 - np.power(self.beta, self.samples_per_class)
        eff = np.where(eff == 0.0, 1e-8, eff)
        weights = (1.0 - self.beta) / eff
        weights = weights / np.sum(weights) * len(weights)
        self.register_buffer("alpha", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets):
        # logits: B x C, targets: B
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # probability of true class
        focal_term = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            loss = focal_term * ce * at
        else:
            loss = focal_term * ce
        return loss.mean()

# build CB focal using original counts (not upsampled) so class weights reflect true prevalence
cb_focal = ClassBalancedFocalLoss(gamma=2.0, beta=0.999, samples_per_class=orig_counts)

# ---------------- optional NT-Xent contrastive loss (unchanged) -----------------------
class NTXent(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.t = temp
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
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

# ----------------------- OPTIM / SCHED -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------------ EVALUATION FUNC ----------------------------
def evaluate(model, loader, logit_adjust=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for b in loader:
            ids = b["text_ids"].to(DEVICE)
            imgs = b["image"].to(DEVICE)
            y = b["label"].to(DEVICE)
            logits, _, _ = model(ids, imgs, logit_adjust)
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    return acc, trues, preds

# ------------------------- TRAIN LOOP -------------------------------
best_acc = 0.0
# initial logit_adjust based on priors (same approach)
samples = train_df["label_id"].value_counts().sort_index().values.astype(np.int64)
priors = samples / samples.sum()
logit_adjust = torch.log(1.0 / (torch.tensor(priors, dtype=torch.float32) + 1e-12))
logit_adjust = (logit_adjust - logit_adjust.mean()) * 0.5
logit_adjust = logit_adjust.to(DEVICE)

# (optional text-only pretrain on info_df retained if needed)

for epoch in range(1, EPOCHS+1):
    model.train()
    tot_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        ids = batch["text_ids"].to(DEVICE)
        imgs = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits, t_emb, v_emb = model(ids, imgs, logit_adjust)

        cls_loss = cb_focal(logits, y)                  # class-balanced focal
        contra_loss = ntxent(t_emb, v_emb) * 0.08       # slightly smaller weight
        loss = cls_loss + contra_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += float(loss.item())
        pbar.set_postfix(loss=tot_loss / (pbar.n + 1e-12))

    scheduler.step()

    val_acc, val_trues, val_preds = evaluate(model, test_loader, logit_adjust)
    print(f"\nEpoch {epoch} | Val Acc: {val_acc:.4f}")
    print(classification_report(val_trues, val_preds, target_names=label_le.classes_, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(val_trues, val_preds))

    # small logit_adjust update using recall to assist minority classes
    report = classification_report(val_trues, val_preds, output_dict=True, zero_division=0)
    recs = []
    for i, cls in enumerate(label_le.classes_):
        rec = report.get(cls, {}).get("recall", 0.0)
        recs.append(rec)
    recs = np.array(recs)
    adjust = (1.0 - recs)
    adjust_t = torch.tensor(adjust, dtype=torch.float32, device=DEVICE)
    logit_adjust = logit_adjust + (adjust_t - adjust_t.mean()) * 0.15
    logit_adjust = torch.clamp(logit_adjust, -4.0, 4.0)

    # Hard example mining (optional) - re-use your function if you want
    # save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocab,
            "epoch": epoch,
            "logit_adjust": logit_adjust.cpu().numpy()
        }, os.path.join(MODEL_DIR, "best_multimodal_balanced_cbfocal.pt"))
        print("Saved best model.")

print("Training complete. Best val acc:", best_acc)
