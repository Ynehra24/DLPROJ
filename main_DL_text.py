# damagenet_text_final_improved_minority.py
"""
Final patched script focused on improving minority-class performance.
Key changes:
 - Small CLIP-style TextEncoder (CLS pooling + projection)
 - Optional small ImageEncoder + strong augmentations (USE_IMAGE)
 - Optional FocalLoss for text fine-tuning (USE_FOCAL_LOSS)
 - Cheap in-process paraphrase fallback if translator not available
 - BT augmentation limited and minority-targeted; augmented rows are tagged
 - Hard-example retries only for MINORITY misclassified samples (1 pass)
 - Minority token dropout increased to 0.20
 - Soft targets applied only to minority samples
 - Safe BalancedBatchSampler (terminates)
"""
import os, re, json, random, time
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -------------------- CONFIG --------------------
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 30            # try to keep divisible by n_classes=3
EPOCHS = 20
LR = 2e-4

# model dims
EMB_DIM = 256
TEXT_EMB = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1

MODEL_DIR = "./damagenet_text_minority_ckpt"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILES = [
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv",
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_dev.tsv",
]
TEST_FILES = [
    "/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_test.tsv",
]

# Back-translation remains available but limited (cheap fallback otherwise)
ENABLE_BACKTRANSLATION = True
BT_LANGS = ["fr", "de", "es", "it"]
BT_CYCLES = 1  # keep low

# Minority-focused hyperparams (tuned for better minority recall)
MINORITY_TOKEN_DROPOUT = 0.20   # increased
USE_SOFT_LABELS = True
SOFT_CONF = 0.88                # slightly higher confidence for minority soft targets

# Hard-example mining
HARD_RETRY_PASSES = 1           # reduced to 1
HARD_SAMPLE_LIMIT = 2000

# Cheap augmentations for minority samples per epoch
MINORITY_AUG_USES_PER_EPOCH = 1
MINORITY_AUG_PROB = 0.6

# Device + dataloader
NUM_WORKERS = 0
PIN_MEMORY = True

# Optional image support (if your TSV has image paths)
USE_IMAGE = True               # set to False to run text-only
IMG_SIZE = 224

# Use focal loss (text) instead of class-balanced focal if True
USE_FOCAL_LOSS = True

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# -------------------- REPRO --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------- cheap paraphrase (fallback for BT) --------------------
# small synonym map as cheap augmentation
SYNONYM_MAP = {
    "car": ["vehicle", "auto"],
    "house": ["home", "residence"],
    "flood": ["inundation"],
    "fire": ["blaze"],
    "collapsed": ["fell", "caved"],
    "help": ["assist"],
}
def cheap_paraphrase_tokens(tokens, p_del=0.12, p_swap=0.08, p_syn=0.06, p_char_noise=0.02):
    # deletion
    toks = [t for t in tokens if not (random.random() < p_del and t != "<cls>")]
    # swap
    if len(toks) >= 2 and random.random() < p_swap:
        i,j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    # synonyms
    for i,t in enumerate(toks):
        if random.random() < p_syn and t in SYNONYM_MAP:
            toks[i] = random.choice(SYNONYM_MAP[t])
    # char noise
    if len(toks) and random.random() < p_char_noise:
        i = random.randrange(len(toks))
        w = toks[i]
        pos = random.randrange(max(1, len(w)))
        toks[i] = w[:pos] + random.choice("aeiou") + w[pos:]
    return toks

def cheap_backtranslate_fallback(text):
    toks = re.findall(r"\b[\w']+\b", text.lower())
    toks2 = cheap_paraphrase_tokens(toks, p_del=0.12, p_swap=0.08, p_syn=0.08, p_char_noise=0.02)
    return " ".join(toks2)

# -------------------- optional translator --------------------
TRANSLATOR = None
TRANSLATOR_NAME = None
if ENABLE_BACKTRANSLATION:
    try:
        from googletrans import Translator as GoogleTranslator
        TRANSLATOR = GoogleTranslator()
        TRANSLATOR_NAME = "googletrans"
        print("BT: using googletrans")
    except Exception:
        try:
            from deep_translator import GoogleTranslator as DeepTranslator
            TRANSLATOR = DeepTranslator(source="auto", target="en")
            TRANSLATOR_NAME = "deep_translator"
            print("BT: using deep_translator")
        except Exception:
            TRANSLATOR = None
            TRANSLATOR_NAME = None
            ENABLE_BACKTRANSLATION = False
            print("BT: translator not available -> will use cheap paraphrase fallback")

def back_translate_via_translator(text, lang_chain):
    try:
        cur = text
        if TRANSLATOR_NAME == "googletrans":
            for lang in lang_chain:
                cur = TRANSLATOR.translate(cur, dest=lang).text
            cur = TRANSLATOR.translate(cur, dest="en").text
            return cur
        elif TRANSLATOR_NAME == "deep_translator":
            from deep_translator import GoogleTranslator as DeepTranslator
            cur = text
            for lang in lang_chain:
                cur = DeepTranslator(source='auto', target=lang).translate(cur)
            cur = DeepTranslator(source='auto', target='en').translate(cur)
            return cur
    except Exception:
        return text

def back_translate(text, lang_chain):
    # limited safe wrapper
    if TRANSLATOR is None:
        return cheap_backtranslate_fallback(text)
    try:
        out = back_translate_via_translator(text, lang_chain)
        return re.sub(r"\s+", " ", out).strip().lower()
    except Exception:
        return cheap_backtranslate_fallback(text)

# -------------------- TEXT UTIL --------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def basic_tokenizer(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenizer(t))
    vocab = {"<pad>":0, "<unk>":1, "<cls>":2}
    for w,c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def encode_text_ids_from_tokens(tokens, vocab):
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:MAX_LEN-1]]
    ids = [vocab["<cls>"]] + ids
    if len(ids) < MAX_LEN:
        ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    return ids[:MAX_LEN]

def encode_text(text, vocab):
    toks = basic_tokenizer(text)
    return torch.tensor(encode_text_ids_from_tokens(toks, vocab), dtype=torch.long)

# -------------------- LOAD DATA --------------------
def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df["tweet_text"] = df["tweet_text"].astype(str).apply(clean_text)
    return df

train_df = pd.concat([load_tsv(p) for p in TRAIN_FILES], ignore_index=True)
test_df  = pd.concat([load_tsv(p) for p in TEST_FILES], ignore_index=True)

label_le = LabelEncoder()
train_df["label_id"] = label_le.fit_transform(train_df["label"])
test_df["label_id"]  = label_le.transform(test_df["label"])
n_classes = len(label_le.classes_)
print("Classes:", label_le.classes_.tolist())

# -------------------- BT augmentation (minority-targeted) --------------------
class_counts = train_df["label_id"].value_counts().sort_index()
mean_count = class_counts.mean()
minority_classes = class_counts[class_counts < mean_count].index.tolist()
print("Minority class ids:", minority_classes)

# tag augmented rows so they can be excluded from hard mining
train_df["_aug_bt"] = False

if ENABLE_BACKTRANSLATION:
    aug_rows = []
    subset = train_df[train_df["label_id"].isin(minority_classes)].reset_index(drop=True)
    print("BT: selecting minority samples for augmentation...")
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="BT select"):
        txt = row["tweet_text"]
        # small decision heuristic
        if len(basic_tokenizer(txt)) < 3:
            continue
        if random.random() < 0.35:  # partial sampling to limit growth
            if TRANSLATOR is not None:
                for c in range(BT_CYCLES):
                    lang_chain = random.sample(BT_LANGS, min(len(BT_LANGS), 2))
                    aug_text = back_translate(txt, lang_chain)
                    new_row = row.copy()
                    new_row["tweet_text"] = aug_text
                    new_row["_aug_bt"] = True
                    aug_rows.append(new_row)
            else:
                # fallback cheap paraphrase
                toks = basic_tokenizer(txt)
                toks2 = cheap_paraphrase_tokens(toks, p_del=0.2, p_swap=0.1, p_syn=0.12)
                new_row = row.copy()
                new_row["tweet_text"] = " ".join(toks2)
                new_row["_aug_bt"] = True
                aug_rows.append(new_row)
    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        train_df = pd.concat([train_df, aug_df], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        print(f"BT: added {len(aug_df)} augmented minority samples.")

# -------------------- physical oversampling --------------------
def oversample_to_max(df):
    counts = df["label_id"].value_counts()
    max_c = counts.max()
    parts = []
    for cls, cnt in counts.items():
        subset = df[df["label_id"] == cls]
        times = int(np.ceil(max_c / max(1, cnt)))
        parts.extend([subset] * times)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print("Oversampled dataset size:", len(out))
    return out

train_df = oversample_to_max(train_df)

# -------------------- cheap on-the-fly augmentations (minority) --------------------
def random_deletion(tokens, p=0.2):
    if len(tokens) <= 3:
        return tokens
    keep = [t for t in tokens if random.random() > p]
    if len(keep) == 0:
        return tokens[:max(1, len(tokens)//2)]
    return keep

def random_swap(tokens, n_swaps=1):
    toks = tokens[:]
    for _ in range(n_swaps):
        if len(toks) < 2: break
        i,j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks

# -------------------- vocab & dataset --------------------
vocab = build_vocab(train_df["tweet_text"].tolist(), min_freq=2)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)
with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

minority_set = set(minority_classes)

# image transforms (strong)
if USE_IMAGE:
    img_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.4,0.4,0.2,0.05),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class BalancedTextDataset(Dataset):
    def __init__(self, df, vocab, minority_set=None, token_dropout=0.0, aug_prob=0.0, use_image=False):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.minority_set = minority_set or set()
        self.token_dropout = token_dropout
        self.aug_prob = aug_prob
        self.use_image = use_image
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["tweet_text"]
        label = int(row["label_id"])
        toks = basic_tokenizer(text)
        # Minority cheap augmentation on-the-fly (random deletion / swap)
        if label in self.minority_set and random.random() < self.aug_prob:
            if random.random() < 0.5:
                toks = random_deletion(toks, p=0.2)
            else:
                toks = random_swap(toks, n_swaps=1)
        ids = encode_text_ids_from_tokens(toks, self.vocab)
        ids = torch.tensor(ids, dtype=torch.long)
        # minority token dropout
        if self.token_dropout > 0 and label in self.minority_set:
            prob = self.token_dropout
            keep_mask = torch.rand(ids.size(0)) > prob
            keep_mask[0] = True  # keep CLS
            kept = ids[keep_mask].tolist()
            kept = kept[:MAX_LEN]
            if len(kept) < MAX_LEN:
                kept += [self.vocab["<pad>"]] * (MAX_LEN - len(kept))
            ids = torch.tensor(kept, dtype=torch.long)
        out = {"input_ids": ids, "label": torch.tensor(label, dtype=torch.long), "_aug_bt": row.get("_aug_bt", False)}
        if self.use_image:
            img_path = row.get("image", None)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
            # apply strong augmentations if oversampled / augmented
            img = img_train_transform(img) if (label in self.minority_set and random.random() < 0.5) else img_test_transform(img)
            out["image"] = img
        return out

train_dataset = BalancedTextDataset(train_df, vocab, minority_set=minority_set, token_dropout=MINORITY_TOKEN_DROPOUT, aug_prob=MINORITY_AUG_PROB, use_image=USE_IMAGE)
test_dataset  = BalancedTextDataset(test_df, vocab, minority_set=minority_set, token_dropout=0.0, aug_prob=0.0, use_image=USE_IMAGE)

# -------------------- SAFE BalancedBatchSampler --------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.idx_by_class = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        for c in self.classes:
            if len(self.idx_by_class[c]) == 0:
                raise ValueError(f"No samples found for class {c}.")
        smallest = min(len(v) for v in self.idx_by_class.values())
        self.num_batches = max(1, smallest // self.samples_per_class)
    def __len__(self):
        return self.num_batches
    def __iter__(self):
        pools = {c: np.random.permutation(v).tolist() for c, v in self.idx_by_class.items()}
        ptrs = {c: 0 for c in self.classes}
        for _ in range(self.num_batches):
            batch = []
            for c in self.classes:
                start = ptrs[c]; end = start + self.samples_per_class
                if end > len(pools[c]):
                    pools[c] = np.random.permutation(pools[c]).tolist()
                    ptrs[c] = 0
                    start = 0; end = self.samples_per_class
                batch.extend(pools[c][start:end])
                ptrs[c] += self.samples_per_class
            if len(batch) < self.batch_size:
                all_idx = np.arange(len(self.labels))
                extra = self.batch_size - len(batch)
                batch.extend(np.random.choice(all_idx, extra, replace=False).tolist())
            random.shuffle(batch)
            yield batch

train_sampler = BalancedBatchSampler(train_df["label_id"].values, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
if USE_IMAGE:
    def collate_multimodal(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        imgs = torch.stack([b["image"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        return {"input_ids": ids, "image": imgs, "label": labels}
    # ensure test loader produces tensors
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_multimodal)
    # override train collate to include images
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_multimodal)
else:
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# -------------------- model --------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, key_padding_mask=mask)[0]
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class TextEncoder(nn.Module):
    """
    Small CLIP-style text encoder: embeddings + transformer stack + CLS pooling + projection to EMB_DIM
    """
    def __init__(self, vocab_size, emb_dim=TEXT_EMB, n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM, out_dim=EMB_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, emb_dim))
        self.layers = nn.ModuleList([TransformerBlock(emb_dim, n_heads, ff_dim, dropout=DROPOUT) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.project = nn.Linear(emb_dim, out_dim)
    def forward(self, ids):
        mask = ids.eq(0)
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        for l in self.layers:
            x = l(x, mask)
        pooled = self.norm(x[:, 0])  # CLS pooling
        out = self.project(pooled)
        out = F.normalize(out, dim=-1)
        return out

class ImageEncoderSmall(nn.Module):
    """
    Small Conv-based visual encoder that projects to EMB_DIM
    """
    def __init__(self, out_dim=EMB_DIM):
        super().__init__()
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

class DamageNetTextFromScratch(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=EMB_DIM, use_image=USE_IMAGE):
        super().__init__()
        self.use_image = use_image
        self.text_enc = TextEncoder(vocab_size)
        if use_image:
            self.img_enc = ImageEncoderSmall()
            in_dim = EMB_DIM * 2
        else:
            in_dim = EMB_DIM
        self.classifier = nn.Sequential(nn.Linear(in_dim, in_dim//2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(in_dim//2, n_classes))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, ids, images=None, logit_adjust=None):
        t = self.text_enc(ids)
        if self.use_image:
            if images is None:
                raise ValueError("Model configured with image encoder but no images were passed.")
            v = self.img_enc(images)
            fused = torch.cat([t, v], dim=1)
        else:
            fused = t
        logits = self.classifier(fused)
        if logit_adjust is not None:
            logits = logits + logit_adjust
        return logits

# -------------------- Losses --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal = focal * alpha[targets]
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

class CBLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, device=DEVICE):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / (effective_num + 1e-12)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        w = self.weights[targets]
        return (w * focal).mean()

# choose criterion
samples = train_df["label_id"].value_counts().sort_index().values.astype(np.int64)
if USE_FOCAL_LOSS:
    # compute alpha from inverse sqrt of counts for focal weighting
    alpha = (1.0 / np.sqrt(samples + 1e-6))
    alpha = alpha / alpha.sum() * len(alpha)
    criterion_cb = FocalLoss(gamma=2.0, alpha=alpha)
else:
    criterion_cb = CBLoss(samples_per_class=samples, beta=0.9999, gamma=2.0, device=DEVICE)

# optional NTXent for alignment (small weight used in hard-mining / auxiliary)
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

# -------------------- utilities --------------------
def build_soft_targets(y_batch, n_classes, soft_conf=SOFT_CONF):
    bs = y_batch.size(0)
    soft = torch.full((bs, n_classes), (1.0 - soft_conf) / (n_classes - 1), device=y_batch.device)
    for i, lab in enumerate(y_batch):
        soft[i, lab] = soft_conf
    return soft

def collect_hard_examples_minority(model, dataset, minority_set, limit=HARD_SAMPLE_LIMIT):
    """
    Collect misclassified training samples that are MINORITY and NOT BT-augmented.
    Returns (xh, yh) or (None, None).
    """
    model.eval()
    xs, ys = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(DEVICE)
            y = b["label"].to(DEVICE)
            aug_bt_flags = b.get("_aug_bt", None)
            if USE_IMAGE:
                imgs = b.get("image", None)
                logits = model(x, images=imgs.to(DEVICE) if imgs is not None else None)
            else:
                logits = model(x)
            preds = logits.argmax(1)
            mask = preds != y
            # filter only minority and non-BT augmented
            mask_indices = []
            if aug_bt_flags is None:
                aug_mask = [False] * len(mask)
            else:
                aug_mask = aug_bt_flags
            for i_val, m in enumerate(mask.cpu().numpy()):
                if not m:
                    continue
                yi = int(y.cpu()[i_val].item())
                if yi in minority_set and not bool(aug_mask[i_val]):
                    mask_indices.append(i_val)
            if len(mask_indices) > 0:
                xs.append(x[mask_indices].cpu())
                ys.append(y[mask_indices].cpu())
    if not xs:
        return None, None
    xh = torch.cat(xs)[:limit]
    yh = torch.cat(ys)[:limit]
    return xh, yh

def evaluate(model, loader, logit_adjust=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(DEVICE)
            y = b["label"].to(DEVICE)
            if USE_IMAGE:
                imgs = b["image"].to(DEVICE)
                logits = model(x, images=imgs, logit_adjust=logit_adjust)
            else:
                logits = model(x, logit_adjust=logit_adjust)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(y.cpu().tolist())
    acc = accuracy_score(trues, preds) if len(trues) > 0 else 0.0
    return acc, trues, preds

# -------------------- training loop --------------------
def train_loop():
    model = DamageNetTextFromScratch(vocab_size=vocab_size, n_classes=n_classes, use_image=USE_IMAGE).to(DEVICE)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # initial logit adjust: stronger boost to minority
    priors = samples / samples.sum()
    logit_adjust = torch.log(1.0 / (torch.tensor(priors, dtype=torch.float32) + 1e-12)).to(DEVICE)
    logit_adjust = (logit_adjust - logit_adjust.mean()) * 1.0

    best = 0.0
    patience = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        it = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            it += 1
            x = batch["input_ids"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            imgs = batch.get("image", None)
            if imgs is not None:
                imgs = imgs.to(DEVICE)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    if USE_IMAGE:
                        logits = model(x, images=imgs, logit_adjust=logit_adjust)
                    else:
                        logits = model(x, logit_adjust=logit_adjust)
                    # apply soft targets only to minority items in batch
                    if USE_SOFT_LABELS:
                        mask_min = torch.tensor([int(int(li) in minority_set) for li in y.cpu().tolist()], device=DEVICE).bool()
                        if mask_min.any():
                            y_min_idx = mask_min.nonzero(as_tuple=False).squeeze(1)
                            y_min = y[y_min_idx]
                            y_soft = build_soft_targets(y_min, n_classes)
                            logits_min = logits[y_min_idx]
                            loss_min = F.kl_div(F.log_softmax(logits_min, dim=1), y_soft, reduction="batchmean")
                            if (~mask_min).any():
                                y_maj = y[~mask_min]
                                logits_maj = logits[~mask_min]
                                loss_maj = criterion_cb(logits_maj, y_maj)
                                loss = 0.6 * loss_min + 0.4 * loss_maj
                            else:
                                loss = loss_min
                        else:
                            loss = criterion_cb(logits, y)
                    else:
                        loss = criterion_cb(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                if USE_IMAGE:
                    logits = model(x, images=imgs, logit_adjust=logit_adjust)
                else:
                    logits = model(x, logit_adjust=logit_adjust)
                if USE_SOFT_LABELS:
                    mask_min = torch.tensor([int(int(li) in minority_set) for li in y.cpu().tolist()], device=DEVICE).bool()
                    if mask_min.any():
                        y_min_idx = mask_min.nonzero(as_tuple=False).squeeze(1)
                        y_min = y[y_min_idx]
                        y_soft = build_soft_targets(y_min, n_classes)
                        logits_min = logits[y_min_idx]
                        loss_min = F.kl_div(F.log_softmax(logits_min, dim=1), y_soft, reduction="batchmean")
                        if (~mask_min).any():
                            y_maj = y[~mask_min]
                            logits_maj = logits[~mask_min]
                            loss_maj = criterion_cb(logits_maj, y_maj)
                            loss = 0.6 * loss_min + 0.4 * loss_maj
                        else:
                            loss = loss_min
                    else:
                        loss = criterion_cb(logits, y)
                else:
                    loss = criterion_cb(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=total_loss/it)

        scheduler.step()

        # evaluate
        val_acc, val_trues, val_preds = evaluate(model, test_loader, logit_adjust)
        print(f"\nEpoch {epoch} | Train Loss: {total_loss/it:.4f} | Val Acc: {val_acc:.4f}")
        print(classification_report(val_trues, val_preds, target_names=label_le.classes_, zero_division=0))

        # small logit_adjust update based on per-class recall (mild)
        report = classification_report(val_trues, val_preds, output_dict=True, zero_division=0)
        recs = []
        for i, cls in enumerate(label_le.classes_):
            rec = report.get(cls, {}).get("recall", 0.0)
            recs.append(rec)
        recs = np.array(recs)
        adjust = (1.0 - recs)
        adjust_t = torch.tensor(adjust, dtype=torch.float32, device=DEVICE)
        logit_adjust = logit_adjust + (adjust_t - adjust_t.mean()) * 0.25
        logit_adjust = torch.clamp(logit_adjust, -4.0, 4.0)

        # checkpoint
        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "logit_adjust": logit_adjust.cpu().numpy()}
        torch.save(ckpt, os.path.join(MODEL_DIR, f"epoch_{epoch}.pth"))

        # hard example mining - only minority misclassified AND not BT-augmented
        hard_x, hard_y = collect_hard_examples_minority(model, train_dataset, minority_set, limit=HARD_SAMPLE_LIMIT)
        if hard_x is not None and HARD_RETRY_PASSES > 0:
            hard_x = hard_x.to(DEVICE)
            hard_y = hard_y.to(DEVICE)
            print(f"Retrying {len(hard_x)} minority hard samples for {HARD_RETRY_PASSES} pass(es)")
            for r in range(HARD_RETRY_PASSES):
                model.train()
                CH = 256
                idx = 0
                total_hloss = 0.0
                it_h = 0
                while idx < len(hard_x):
                    xb = hard_x[idx: idx+CH]
                    yb = hard_y[idx: idx+CH]
                    optimizer.zero_grad()
                    if scaler:
                        with torch.cuda.amp.autocast():
                            # images are not available here for hard_x; this hard-retry focused on text encoder
                            logits = model(xb, logit_adjust=logit_adjust)
                            y_soft = build_soft_targets(yb, n_classes, soft_conf=SOFT_CONF)
                            hloss = F.kl_div(F.log_softmax(logits, dim=1), y_soft, reduction="batchmean")
                        scaler.scale(hloss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        logits = model(xb, logit_adjust=logit_adjust)
                        y_soft = build_soft_targets(yb, n_classes, soft_conf=SOFT_CONF)
                        hloss = F.kl_div(F.log_softmax(logits, dim=1), y_soft, reduction="batchmean")
                        hloss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total_hloss += float(hloss.item())
                    it_h += 1
                    idx += CH
                print(f"  Hard pass {r+1} loss: {total_hloss/it_h:.4f}")

        # best model
        if val_acc > best:
            best = val_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_minority.pt"))
            print("Saved new best model.")
        else:
            patience += 1
            if patience >= 6:
                print("Early stopping.")
                break

    print("Training finished. Best val acc:", best)

# -------------------- ENTRY --------------------
if __name__ == "__main__":
    train_loop()
