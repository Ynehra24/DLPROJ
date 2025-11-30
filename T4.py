import os, random, numpy as np, pandas as pd
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import timm
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_ROOT    = "/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES"
TSV_ROOT    = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"

TRAIN_TSV   = os.path.join(TSV_ROOT, "task_humanitarian_text_img_train.tsv")
DEV_TSV     = os.path.join(TSV_ROOT, "task_humanitarian_text_img_dev.tsv")
TEST_TSV    = os.path.join(TSV_ROOT, "task_humanitarian_text_img_test.tsv")

TEXT_COL    = "tweet_text"
IMG_COL     = "image"
LABEL_COL   = "label_text"   # fine-grained humanitarian labels

NUM_CLASSES = 3   # people_affected, rescue_needed, no_human
IMG_SIZE    = 256
MAX_LEN     = 96
BATCH_SIZE  = 8

EPOCHS      = 10
PATIENCE    = 3

LR_TXT      = 2e-5
LR_IMG      = 3e-5
LR_HEAD     = 8e-4

BACKBONE_IMG = "convnext_tiny"

MODEL_PATH  = "T4_MULTIMODAL.pt"

print("🟢 DEVICE =", DEVICE)
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

print("\n📂 Loading Task 4 TSVs...")
df_train = pd.read_csv(TRAIN_TSV, sep="\t")
df_dev   = pd.read_csv(DEV_TSV,   sep="\t")
df_test  = pd.read_csv(TEST_TSV,  sep="\t")

print("Train columns:", df_train.columns.tolist())

# Paper: people affected (affected/ injured/ missing), rescue needed (rescue_volunteering_or_donation_effort), else no_human. [attached_file:67]
def map_to_t4(lbl: str) -> str:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
    ]:
        return "people_affected"
    elif lbl in [
        "rescue_volunteering_or_donation_effort",
    ]:
        return "rescue_needed"
    else:
        return "no_human"

for df in (df_train, df_dev, df_test):
    df["t4_label"] = df[LABEL_COL].apply(map_to_t4)

labels = ["people_affected", "rescue_needed", "no_human"]
label2id = {lab:i for i, lab in enumerate(labels)}
id2label = {i:lab for lab, i in label2id.items()}
print("Label2id:", label2id)

df_train["label_id"] = df_train["t4_label"].map(label2id)
df_dev["label_id"]   = df_dev["t4_label"].map(label2id)
df_test["label_id"]  = df_test["t4_label"].map(label2id)

print("Train counts:", Counter(df_train.label_id))

train_tfms = T.Compose([
    T.Resize((IMG_SIZE+16, IMG_SIZE+16)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize([.485,.456,.406],[.229,.224,.225])
])

eval_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([.485,.456,.406],[.229,.224,.225])
])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class T4Dataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[TEXT_COL])
        y    = int(row["label_id"])
        img_rel = row[IMG_COL]

        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        path = os.path.join(IMG_ROOT, img_rel)
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
        img = train_tfms(img) if self.train else eval_tfms(img)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img,
        }
        return batch, torch.tensor(y, dtype=torch.long)

train_ds = T4Dataset(df_train, train=True)
dev_ds   = T4Dataset(df_dev,   train=False)
test_ds  = T4Dataset(df_test,  train=False)

labels_train = df_train.label_id.to_numpy()
class_counts = np.bincount(labels_train)
inv = 1.0 / np.maximum(class_counts, 1)
w_classes = inv / inv.mean()
sample_w = w_classes[labels_train]
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

print("\n✅ Task 4 multimodal dataloaders ready")

class T4Multimodal(nn.Module):
    def __init__(self):
        super().__init__()
        # Image encoder (ConvNeXt-Tiny instead of Inception, but same role). [attached_file:67]
        self.img = timm.create_model(
            BACKBONE_IMG,
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1,
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,IMG_SIZE,IMG_SIZE)
            d_img = self.img(dummy).shape[-1]
        print("🔍 T4 image feat dim:", d_img)

        # Text encoder (DistilBERT; BERT in paper). [attached_file:67]
        self.txt = AutoModel.from_pretrained("distilbert-base-uncased")
        d_txt = self.txt.config.hidden_size
        print("🔍 T4 text feat dim:", d_txt)

        self.txt_ln = nn.LayerNorm(d_txt)

        fused_dim = d_img + d_txt
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def encode_image(self, B):
        return self.img(B["image"])

    def encode_text(self, B):
        out = self.txt(
            input_ids=B["input_ids"],
            attention_mask=B["attention_mask"]
        ).last_hidden_state
        cls = out[:, 0, :]
        return self.txt_ln(cls)

    def forward(self, B):
        f_img = self.encode_image(B)
        f_txt = self.encode_text(B)
        fused = torch.cat([f_img, f_txt], dim=-1)
        logits = self.head(fused)
        return logits

def make_optimizer(model, lr_txt, lr_img, lr_head):
    return torch.optim.AdamW(
        [
            {"params": model.txt.parameters(),  "lr": lr_txt},
            {"params": model.img.parameters(),  "lr": lr_img},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=0.01
    )

def run_epoch(loader, model, optimizer, criterion, train=True, label="Epoch", amp=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    P, T = [], []

    for B, y in tqdm(loader, desc=label):
        y = y.to(DEVICE)
        B = {k: v.to(DEVICE, non_blocking=True) for k, v in B.items()}

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp and DEVICE.type=="cuda"):
            logits = model(B)
            loss   = criterion(logits, y)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        P.extend(logits.argmax(1).cpu().tolist())
        T.extend(y.cpu().tolist())

    f1 = f1_score(T, P, average="macro", zero_division=0)
    cm = confusion_matrix(T, P)
    return total_loss / len(loader), f1, cm

def train_t4():
    model = T4Multimodal().to(DEVICE)

    inv = 1.0 / np.maximum(class_counts, 1)
    w = inv / inv.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = make_optimizer(model, LR_TXT, LR_IMG, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    wait    = 0

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_f1, _ = run_epoch(
            train_loader, model, optimizer, criterion,
            train=True, label=f"T4 Train E{epoch}", amp=True
        )
        dv_loss, dv_f1, dv_cm = run_epoch(
            dev_loader, model, optimizer, criterion,
            train=False, label="T4 Dev", amp=True
        )
        print(f"T4 Train E{epoch} Loss={tr_loss:.4f} F1={tr_f1:.4f}")
        print(f"T4 Dev Macro-F1={dv_f1:.4f}")
        print(dv_cm, "\n")

        scheduler.step()

        if dv_f1 > best_f1:
            best_f1, wait = dv_f1, 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("💾 NEW BEST SAVED (T4) ->", MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⛔ Early stopping (T4).")
                break

    print("\n🔥 BEST T4 Dev Macro-F1:", best_f1)
    return best_f1

best_f1_t4 = train_t4()

print("\n📥 Loading best checkpoint for T4 TEST...")
model = T4Multimodal().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

P_all, T_all = [], []
with torch.no_grad():
    for B, y in tqdm(test_loader, desc="T4 Test"):
        B = {k: v.to(DEVICE, non_blocking=True) for k, v in B.items()}
        logits = model(B)
        preds  = logits.argmax(1).cpu().tolist()
        P_all.extend(preds)
        T_all.extend(y.tolist())

print("\n============== FINAL TEST RESULTS (T4 MULTIMODAL) ==============")
print(classification_report(
    T_all,
    P_all,
    digits=4,
    target_names=[id2label[i] for i in range(NUM_CLASSES)],
    zero_division=0,
))
print("T4 Confusion Matrix:\n", confusion_matrix(T_all, P_all))
print("T4 Final Test Macro-F1:", f1_score(T_all, P_all, average="macro", zero_division=0))
