import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

IMG_SIZE = 224
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-4

def clean_text(text):
    import re
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def basic_tokenizer(text):
    import re
    return re.findall(r"\b[\w']+\b", text.lower())

def build_vocab(texts, min_freq=2):
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenizer(t))
    vocab = {"<unk>":0, "<pad>":1, "<cls>":2}
    for w,cnt in counter.items():
        if cnt >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def encode_text_ids(text, vocab, max_len=MAX_LEN):
    toks = basic_tokenizer(text)[: max_len - 1]
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
    ids = [vocab["<cls>"]] + ids
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

class TweetDataset(Dataset):
    def __init__(self, df, vocab, label_encoder, img_transform, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.img_transform = img_transform
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_ids = torch.tensor(encode_text_ids(row["tweet_text"], self.vocab, self.max_len), dtype=torch.long)
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
        img = self.img_transform(img)
        label = torch.tensor(self.label_encoder.transform([row["label"]])[0], dtype=torch.long)
        return {"text_ids": text_ids, "image": img, "label": label}

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layers=2, d_ff=512, max_len=MAX_LEN, out_dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, d_ff, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, out_dim)
    def forward(self, ids):
        x = self.emb(ids) + self.pos[:, :ids.size(1), :]
        for l in self.layers:
            x = l(x)
        pooled = self.norm(x[:,0])
        out = self.project(pooled)
        out = F.normalize(out, dim=-1)
        return out

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)
    def forward(self, x):
        x = self.net(x).view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x

class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.text_enc = TextEncoder(vocab_size)
        self.img_enc = ImageEncoder()
        self.cls = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, text_ids, images):
        t = self.text_enc(text_ids)
        v = self.img_enc(images)
        x = torch.cat([t, v], dim=1)
        logits = self.cls(x)
        return logits

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        text_ids = batch["text_ids"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(text_ids, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1e-12))
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in pbar:
            text_ids = batch["text_ids"].to(DEVICE)
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(text_ids, images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            pred = logits.argmax(1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix(loss=total_loss / (pbar.n + 1e-12))
    acc = np.mean(np.array(preds) == np.array(trues))
    return acc, preds, trues

if __name__ == "__main__":
    train_val_files = [
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv", "damage"),
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_dev.tsv", "damage"),
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv", "informative"),
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv", "informative"),
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv", "humanitarian"),
        ("/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv", "humanitarian")
    ]
    train_dfs, val_dfs = [], []
    for path, label in train_val_files:
        df = pd.read_csv(path, sep="\t")
        df["tweet_text"] = df["tweet_text"].astype(str).apply(clean_text)
        df["label"] = label
        if "train" in path:
            train_dfs.append(df)
        else:
            val_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    label_encoder = LabelEncoder()
    all_labels = pd.concat([train_df["label"], val_df["label"]], ignore_index=True)
    label_encoder.fit(all_labels)
    n_classes = len(label_encoder.classes_)
    vocab = build_vocab(train_df["tweet_text"].tolist(), min_freq=2)
    vocab_size = len(vocab)
    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = TweetDataset(train_df, vocab, label_encoder, img_transform)
    val_ds = TweetDataset(val_df, vocab, label_encoder, img_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = MultiModalClassifier(vocab_size, n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Loss={loss:.4f} | Val Acc={acc:.4f}")
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "label_encoder": label_encoder.classes_
    }, "multimodal_tweet_model.pt")
    print("Training complete.")

def predict_tweet(model, tweet_text, image_path, vocab, label_encoder):
    model.eval()
    text = clean_text(tweet_text)
    text_ids = torch.tensor([encode_text_ids(text, vocab)], dtype=torch.long).to(DEVICE)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(text_ids, img)
        pred = logits.argmax(1).item()
    return label_encoder[pred]
