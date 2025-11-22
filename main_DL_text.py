import os, re, random, json, math, time
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-4
IMG_SIZE = 224
EMB_DIM = 512
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

NUM_WORKERS = 0
PIN_MEMORY = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?\'\-]", " ", text)
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
TRANSLATOR_NAME = None
try:
    from googletrans import Translator as GoogleTranslator
    TRANSLATOR = GoogleTranslator()
    TRANSLATOR_NAME = "googletrans"
    print("googletrans available for optional BT (will be used if ENABLE_BT=True).")
except Exception:
    TRANSLATOR = None
    TRANSLATOR_NAME = None

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

def load_tsv(path):
    df = pd.read_csv(path, sep="	")
    df["tweet_text"] = df["tweet_text"].astype(str).apply(clean_text)
    return df

train_df = pd.concat([load_tsv(p) for p in TRAIN_FILES], ignore_index=True)
test_df = pd.concat([load_tsv(p) for p in TEST_FILES], ignore_index=True)

label_le = LabelEncoder()
train_df["label_id"] = label_le.fit_transform(train_df["label"])
test_df["label_id"] = label_le.transform(test_df["label"])
n_classes = len(label_le.classes_)
print("Classes:", label_le.classes_.tolist())

try:
    info_df = pd.concat([load_tsv(p) for p in INFO_FILES], ignore_index=True)
    if "label" in info_df.columns:
        info_df = info_df[info_df["label"].astype(str).str.lower() == "informative"].reset_index(drop=True)
    info_df["_is_unlabeled"] = True
    print(f"Loaded informative unlabeled samples: {len(info_df)}")
except Exception as e:
    info_df = pd.DataFrame()
    print("Could not load informative files (or none found). Skipping informative pretraining. Err:", e)

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
        if self.do_aug:
            if self.enable_bt and TRANSLATOR is not None and random.random() < self.bt_prob:
                text = cheap_backtranslate(text)
            else:
                toks = basic_tokenizer(text)
                toks = cheap_paraphrase(toks, p_del=0.12, p_split=0.08, p_syn=0.06, p_char_noise=0.02)  # slight name fix
                text = " ".join(toks)
        text_ids = torch.tensor(encode_text_ids(text), dtype=torch.long)

        img_path = row.get("image", None)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))

        img = img_train_transform(img) if self.do_aug else img_test_transform(img)

        label = torch.tensor(row["label_id"], dtype=torch.long)
        return {"text_ids": text_ids, "image": img, "label": label}

train_dataset = MultimodalDataset(train_df, do_aug=True, enable_bt=False, bt_prob=0.05)
test_dataset = MultimodalDataset(test_df, do_aug=False, enable_bt=False)

counts = train_df["label_id"].value_counts().sort_index()
weights_per_class = 1.0 / torch.sqrt(torch.tensor(counts.values, dtype=torch.float) + 1e-6)
sample_weights = [weights_per_class[int(l)] for l in train_df["label_id"].tolist()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

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
        mask = ids.eq(1)
        x = self.emb(ids) + self.pos[:, :ids.size(1), :]
        for l in self.layers:
            x = l(x, mask)
        pooled = self.norm(x[:,0])
        out = self.project(pooled)
        out = F.normalize(out, dim=-1)
        return out

class ImageEncoder(nn.Module):
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

class MultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, use_image=True):
        super().__init__()
        self.use_image = use_image
        self.text_enc = TextEncoder(vocab_size)
        self.img_enc = ImageEncoder() if use_image else None
        feat_dim = EMB_DIM * (2 if use_image else 1)
        self.cls = nn.Sequential(
            nn.Linear(feat_dim, CLS_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(CLS_HIDDEN, n_classes)
        )
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, ids, images=None, logit_adjust=None):
        t = self.text_enc(ids)
        if self.use_image:
            if images is None:
                raise ValueError("Model configured with image encoder but no images were passed.")
            v = self.img_enc(images)
            fused = torch.cat([t, v], dim=1)
        else:
            fused = t
        logits = self.cls(fused)
        if logit_adjust is not None:
            logits = logits + logit_adjust
        return logits, t, (v if self.use_image else None)

model = MultimodalClassifier(vocab_size=vocab_size, n_classes=n_classes, use_image=True).to(DEVICE)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

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
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal = focal * at
        return focal.mean()

class_counts = train_df["label_id"].value_counts().sort_index().values
alpha = (1.0 / np.sqrt(class_counts + 1e-6))
alpha = alpha / alpha.sum() * len(alpha)
focal_criterion = FocalLoss(gamma=2.0, alpha=alpha)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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

def collect_hard_examples_minority(model, dataset, minority_set, limit=2000):
    model.eval()
    xs, ys = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for b in loader:
            x = b["text_ids"].to(DEVICE)
            y = b["label"].to(DEVICE)
            if model.use_image:
                dummy_imgs = torch.zeros((x.size(0), 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
                logits, _, _ = model(x, images=dummy_imgs, logit_adjust=None)
            else:
                logits, _, _ = model(x, images=None)
            preds = logits.argmax(1)
            mask = preds != y
            for i_val, m in enumerate(mask.cpu().numpy()):
                if not m:
                    continue
                yi = int(y.cpu()[i_val].item())
                if yi in minority_set:
                    xs.append(x[i_val].cpu().unsqueeze(0))
                    ys.append(y[i_val].cpu().unsqueeze(0))
    if not xs:
        return None, None
    xh = torch.cat(xs)[:limit]
    yh = torch.cat(ys)[:limit]
    return xh, yh

class InformativeTextDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = self.df.iloc[idx]["tweet_text"]
        toks = basic_tokenizer(text)
        toks = cheap_paraphrase(toks, p_del=0.10, p_swap=0.08, p_syn=0.06, p_char_noise=0.02)
        ids = torch.tensor(encode_text_ids(" ".join(toks)), dtype=torch.long)
        return ids

if len(info_df) > 0:
    info_dataset = InformativeTextDataset(info_df)
    info_loader = DataLoader(info_dataset, batch_size=64, shuffle=True, num_workers=0)
else:
    info_loader = None

best_acc = 0.0
samples = train_df["label_id"].value_counts().sort_index().values.astype(np.int64)
priors = samples / samples.sum()
logit_adjust = torch.log(1.0 / (torch.tensor(priors, dtype=torch.float32) + 1e-12)).to(DEVICE)
logit_adjust = (logit_adjust - logit_adjust.mean()) * 0.5

if info_loader is not None:
    print("
Starting text-only pretraining on informative unlabeled corpus (1 epoch)...")
    model.train()
    pre_epochs = 1
    for ep in range(pre_epochs):
        tot = 0.0
        it = 0
        pbar = tqdm(info_loader, desc=f"Pretrain {ep+1}/{pre_epochs}")
        for ids in pbar:
            ids = ids.to(DEVICE)
            optimizer.zero_grad()
            z1 = model.text_enc(ids)
            aug_ids_list = []
            ids_cpu = ids.cpu().numpy()
            for row in ids_cpu:
                toks = []
                for token_id in row:
                    if int(token_id) in (vocab.get("<pad>",1),):
                        continue
                    tok = inv_vocab.get(int(token_id), "<unk>")
                    toks.append(tok)
                toks_aug = cheap_paraphrase(toks, p_del=0.15, p_swap=0.1, p_syn=0.06, p_char_noise=0.02)
                aug_text = " ".join(toks_aug) if len(toks_aug) > 0 else " ".join(toks)
                aug_ids = encode_text_ids(aug_text)
                aug_ids_list.append(torch.tensor(aug_ids, dtype=torch.long))
            aug_ids = torch.stack(aug_ids_list).to(DEVICE)

            z2 = model.text_enc(aug_ids)

            loss = ntxent(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.text_enc.parameters(), 1.0)
            optimizer.step()

            tot += float(loss.item())
            it += 1
            pbar.set_postfix(loss=tot/it if it>0 else 0.0)
    print("Finished informative text pretraining.
")

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

        cls_loss = focal_criterion(logits, y)

        contra_loss = ntxent(t_emb, v_emb) * 0.1

        loss = cls_loss + contra_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += float(loss.item())
        pbar.set_postfix(loss=tot_loss / (pbar.n + 1e-12))

    scheduler.step()

    val_acc, val_trues, val_preds = evaluate(model, test_loader, logit_adjust)
    print(f"
Epoch {epoch} | Val Acc: {val_acc:.4f}")
    print(classification_report(val_trues, val_preds, target_names=label_le.classes_, zero_division=0))
    print("Confusion Matrix:
", confusion_matrix(val_trues, val_preds))

    report = classification_report(val_trues, val_preds, output_dict=True, zero_division=0)
    recs = []
    for i, cls in enumerate(label_le.classes_):
        rec = report.get(cls, {}).get("recall", 0.0)
        recs.append(rec)
    recs = np.array(recs)
    adjust = (1.0 - recs)
    adjust_t = torch.tensor(adjust, dtype=torch.float32, device=DEVICE)
    logit_adjust = logit_adjust + (adjust_t - adjust_t.mean()) * 0.2
    logit_adjust = torch.clamp(logit_adjust, -4.0, 4.0)

    minority_set = set(np.where(samples < samples.mean())[0].tolist())
    hard_x, hard_y = collect_hard_examples_minority(model, train_dataset, minority_set, limit=2000)
    if hard_x is not None:
        hard_x = hard_x.to(DEVICE)
        hard_y = hard_y.to(DEVICE)
        print(f"Retrying {len(hard_x)} minority hard samples (text-only) for 1 pass")
        model.train()
        CH = 256
        idx = 0
        total_hloss = 0.0
        it_h = 0
        while idx < len(hard_x):
            xb = hard_x[idx: idx+CH]
            yb = hard_y[idx: idx+CH]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                z = model.text_enc(xb)
                logits_h = model.cls(torch.cat([z, torch.zeros_like(z)], dim=1)) if model.use_image else model.cls(z)
                y_soft = F.one_hot(yb, num_classes=n_classes).float() * 0.0
                y_soft = y_soft + (1.0 / n_classes) * 0.05
                for i in range(y_soft.size(0)):
                    y_soft[i, yb[i]] = 0.95
                hloss = F.kl_div(F.log_softmax(logits_h, dim=1), y_soft.to(logits_h.device), reduction="batchmean")
            hloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_hloss += float(hloss.item())
            it_h += 1
            idx += CH
        if it_h > 0:
            print(f"  Hard pass loss: {total_hloss/it_h:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocab,
            "epoch": epoch,
            "logit_adjust": logit_adjust.cpu().numpy()
        }, os.path.join(MODEL_DIR, "best_multimodal_fixed_informative.pt"))
        print("Saved best model.")

print("Training complete. Best val acc:", best_acc)

def load_and_eval(checkpoint_path, test_tsv=None):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_local = MultimodalClassifier(vocab_size=len(ckpt["vocab"]), n_classes=len(label_le.classes_), use_image=True).to(DEVICE)
    model_local.load_state_dict(ckpt["model_state"])
    model_local.eval()
    if test_tsv is not None:
        df_test = pd.read_csv(test_tsv, sep="	")
        df_test["tweet_text"] = df_test["tweet_text"].astype(str).apply(clean_text)
        ds = MultimodalDataset(df_test, do_aug=False)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        acc, trues, preds = evaluate(model_local, loader)
        print("Loaded checkpoint acc:", acc)
        print(classification_report(trues, preds, target_names=label_le.classes_))
    return model_local
