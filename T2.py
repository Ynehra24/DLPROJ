############################################################
#   TASK 2 — HUMAN / STRUCTURE / NON-INFO FUSION (T2T+T2S)
############################################################

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

# ================== CONFIG ==================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_ROOT    = "/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES"
TSV_ROOT    = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"

TRAIN_TSV   = os.path.join(TSV_ROOT, "task_humanitarian_text_img_train.tsv")
DEV_TSV     = os.path.join(TSV_ROOT, "task_humanitarian_text_img_dev.tsv")
TEST_TSV    = os.path.join(TSV_ROOT, "task_humanitarian_text_img_test.tsv")

TEXT_COL    = "tweet_text"
IMG_COL     = "image"
LABEL_COL   = "label_text"   # humanitarian label from text side

NUM_CLASSES = 3
IMG_SIZE    = 256
MAX_LEN     = 96
BATCH_SIZE  = 8

EPOCHS_T2T  = 6
EPOCHS_T2S  = 10
PATIENCE    = 3

LR_TXT      = 2e-5
LR_HEAD_T2T = 4e-4

LR_IMG      = 3e-5
LR_HEAD_T2S = 8e-4

BACKBONE_IMG = "convnext_tiny"

print("🟢 DEVICE =", DEVICE)
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# ================== LOAD TSVs ==================
print("\n📂 Loading Task 2 TSVs...")
df_train = pd.read_csv(TRAIN_TSV, sep="\t")
df_dev   = pd.read_csv(DEV_TSV,   sep="\t")
df_test  = pd.read_csv(TEST_TSV,  sep="\t")

print("Train columns:", df_train.columns.tolist())

# map 8-way humanitarian labels -> 3-way Task 2 labels
def map_to_task2(lbl: str) -> str:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
        "rescue_volunteering_or_donation_effort",
    ]:
        return "humanitarian"
    elif lbl in [
        "infrastructure_and_utility_damage",
        "vehicle_damage",
    ]:
        return "structure"
    else:
        # "not_humanitarian" and "other_relevant_information"
        return "non_informative"

for df in (df_train, df_dev, df_test):
    df["t2_label"] = df[LABEL_COL].apply(map_to_task2)

labels = ["humanitarian", "non_informative", "structure"]
label2id = {lab:i for i, lab in enumerate(labels)}
id2label = {i:lab for lab, i in label2id.items()}
print("Label2id:", label2id)

df_train["label_id"] = df_train["t2_label"].map(label2id)
df_dev["label_id"]   = df_dev["t2_label"].map(label2id)
df_test["label_id"]  = df_test["t2_label"].map(label2id)

print("Train counts:", Counter(df_train.label_id))

# ================== TFMS & TOKENIZER ==========
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

# ================== DATASETS ===================
class T2TextDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[TEXT_COL])
        y    = int(row["label_id"])

        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}, torch.tensor(y, dtype=torch.long)


class T2ImageDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_rel = row[IMG_COL]
        y       = int(row["label_id"])

        path = os.path.join(IMG_ROOT, img_rel)
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))

        img = train_tfms(img) if self.train else eval_tfms(img)
        return img, torch.tensor(y, dtype=torch.long)

# text
train_ds_txt = T2TextDataset(df_train)
dev_ds_txt   = T2TextDataset(df_dev)
test_ds_txt  = T2TextDataset(df_test)

train_loader_txt = DataLoader(train_ds_txt, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
dev_loader_txt   = DataLoader(dev_ds_txt,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
test_loader_txt  = DataLoader(test_ds_txt,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

# image
train_ds_img = T2ImageDataset(df_train, train=True)
dev_ds_img   = T2ImageDataset(df_dev,   train=False)
test_ds_img  = T2ImageDataset(df_test,  train=False)

labels_train = df_train.label_id.to_numpy()
class_counts = np.bincount(labels_train)
inv = 1.0 / np.maximum(class_counts, 1)
w_classes = inv / inv.mean()
sample_w = w_classes[labels_train]
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader_img = DataLoader(train_ds_img, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
dev_loader_img   = DataLoader(dev_ds_img,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
test_loader_img  = DataLoader(test_ds_img,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

print("\n✅ Task 2 text & image dataloaders ready")

# ================== MODELS =====================
class T2TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = AutoModel.from_pretrained("distilbert-base-uncased")
        H = self.txt.config.hidden_size
        self.ln = nn.LayerNorm(H)
        self.head = nn.Sequential(
            nn.Linear(H, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, B):
        out = self.txt(
            input_ids=B["input_ids"],
            attention_mask=B["attention_mask"]
        ).last_hidden_state
        cls = out[:, 0, :]
        cls = self.ln(cls)
        return self.head(cls)


class T2ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = timm.create_model(
            BACKBONE_IMG,
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1,
        )
        with torch.no_grad():
            d = self.img(torch.zeros(1,3,IMG_SIZE,IMG_SIZE)).shape[-1]
        print("🔍 Task2 image feat dim:", d)

        self.head = nn.Sequential(
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        feat = self.img(x)
        return self.head(feat)

# ================== TRAIN HELPERS =============
def train_text_model():
    model = T2TextModel().to(DEVICE)
    inv = 1.0 / np.maximum(class_counts, 1)
    w = inv / inv.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.txt.parameters(),  "lr": LR_TXT},
            {"params": model.head.parameters(), "lr": LR_HEAD_T2T},
        ],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_T2T)

    best_f1 = 0.0
    wait    = 0

    for epoch in range(1, EPOCHS_T2T+1):
        print(f"\n===== T2T E{epoch}/{EPOCHS_T2T} =====")
        model.train()
        total_loss = 0.0
        for B, y in tqdm(train_loader_txt, desc=f"T2T Train E{epoch}"):
            y = y.to(DEVICE)
            B = {k:v.to(DEVICE) for k,v in B.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(B)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print("Train loss:", total_loss / len(train_loader_txt))

        # dev
        model.eval()
        P, T_ = [], []
        with torch.no_grad():
            for B, y in tqdm(dev_loader_txt, desc="T2T Dev"):
                B = {k:v.to(DEVICE) for k,v in B.items()}
                logits = model(B)
                preds  = logits.argmax(1).cpu().tolist()
                P.extend(preds)
                T_.extend(y.tolist())
        f1 = f1_score(T_, P, average="macro", zero_division=0)
        print("Dev F1:", f1)
        print(confusion_matrix(T_, P))

        scheduler.step()
        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), "T2T_TEXT.pt")
            print("💾 Saved T2T_TEXT.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⛔ Early stop T2T")
                break
    print("Best T2T F1:", best_f1)
    return best_f1

def train_image_model():
    model = T2ImageModel().to(DEVICE)
    inv = 1.0 / np.maximum(class_counts, 1)
    w = inv / inv.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.img.parameters(),  "lr": LR_IMG},
            {"params": model.head.parameters(), "lr": LR_HEAD_T2S},
        ],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_T2S)

    best_f1 = 0.0
    wait    = 0

    for epoch in range(1, EPOCHS_T2S+1):
        print(f"\n===== T2S E{epoch}/{EPOCHS_T2S} =====")
        model.train()
        total_loss = 0.0
        for imgs, y in tqdm(train_loader_img, desc=f"T2S Train E{epoch}"):
            imgs = imgs.to(DEVICE)
            y    = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print("Train loss:", total_loss / len(train_loader_img))

        model.eval()
        P, T_ = [], []
        with torch.no_grad():
            for imgs, y in tqdm(dev_loader_img, desc="T2S Dev"):
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                preds  = logits.argmax(1).cpu().tolist()
                P.extend(preds)
                T_.extend(y.tolist())
        f1 = f1_score(T_, P, average="macro", zero_division=0)
        print("Dev F1:", f1)
        print(confusion_matrix(T_, P))

        scheduler.step()
        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), "T2S_IMAGE.pt")
            print("💾 Saved T2S_IMAGE.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⛔ Early stop T2S")
                break
    print("Best T2S F1:", best_f1)
    return best_f1

# ================== TRAIN T2T & T2S =============
best_t2t = train_text_model()
best_t2s = train_image_model()

# ================== FUSION (TASK 2) ============
text_model  = T2TextModel().to(DEVICE)
image_model = T2ImageModel().to(DEVICE)
text_model.load_state_dict(torch.load("T2T_TEXT.pt",  map_location=DEVICE))
image_model.load_state_dict(torch.load("T2S_IMAGE.pt", map_location=DEVICE))
text_model.eval()
image_model.eval()

def fuse_task2(text_logits, img_logits):
    # F_humanstruct(t,i):
    #   2 (structure)       if t==2 or i==2
    #   0 (humanitarian)    elif t==0 or i==0
    #   1 (non_informative) else
    t_pred = text_logits.argmax(dim=1)
    i_pred = img_logits.argmax(dim=1)
    fused  = torch.empty_like(t_pred)

    mask_struct = (t_pred == 2) | (i_pred == 2)
    fused[mask_struct] = 2

    mask_human = ~mask_struct & ((t_pred == 0) | (i_pred == 0))
    fused[mask_human] = 0

    mask_rest = ~(mask_struct | mask_human)
    fused[mask_rest] = 1

    return fused

all_preds, all_true = [], []

with torch.no_grad():
    for (B_txt, y_txt), (imgs, y_img) in tqdm(
        zip(test_loader_txt, test_loader_img),
        desc="Task2 Fusion Test",
        total=len(test_loader_txt)
    ):
        # assume loaders have same ordering
        B_txt = {k:v.to(DEVICE) for k,v in B_txt.items()}
        imgs  = imgs.to(DEVICE)

        logits_txt = text_model(B_txt)
        logits_img = image_model(imgs)

        fused = fuse_task2(logits_txt, logits_img)
        all_preds.extend(fused.cpu().tolist())
        all_true.extend(y_txt.tolist())

print("\n============== TASK 2 FUSION TEST RESULTS ==============")
print(classification_report(
    all_true,
    all_preds,
    digits=4,
    target_names=[id2label[i] for i in range(NUM_CLASSES)],
    zero_division=0,
))
print("Confusion matrix:\n", confusion_matrix(all_true, all_preds))
print("Macro-F1:", f1_score(all_true, all_preds, average="macro", zero_division=0))
