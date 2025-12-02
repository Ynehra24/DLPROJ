import os, random, numpy as np, pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModel

# ================== CONFIG ==================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TSV_ROOT    = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"
TRAIN_TSV   = os.path.join(TSV_ROOT, "task_humanitarian_text_img_train.tsv")
DEV_TSV     = os.path.join(TSV_ROOT, "task_humanitarian_text_img_dev.tsv")
TEST_TSV    = os.path.join(TSV_ROOT, "task_humanitarian_text_img_test.tsv")
TEXT_COL    = "tweet_text"
LABEL_COL   = "label_text"
NUM_CLASSES = 3
MAX_LEN     = 96
BATCH_SIZE  = 8
EPOCHS      = 6
PATIENCE    = 3
LR_TXT      = 2e-5
LR_HEAD     = 4e-4

print("🟢 DEVICE =", DEVICE)
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# ================== LOAD TSVs ==================
print("\n📂 Loading Task 2 TSVs...")
df_train = pd.read_csv(TRAIN_TSV, sep="\t")
df_dev   = pd.read_csv(DEV_TSV,   sep="\t")
df_test  = pd.read_csv(TEST_TSV,  sep="\t")

# Map 8-way humanitarian labels -> 3-way Task 2 labels
def map_to_task2(lbl: str) -> str:
    if lbl in ["affected_individuals", "injured_or_dead_people", "missing_or_found_people", "rescue_volunteering_or_donation_effort"]:
        return "humanitarian"
    elif lbl in ["infrastructure_and_utility_damage", "vehicle_damage"]:
        return "structure"
    else:  # "not_humanitarian" and "other_relevant_information"
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

# ================== TOKENIZER ===============
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ================== DATASET ==================
class T2TextDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[TEXT_COL])
        y = int(row["label_id"])
        enc = tokenizer(text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}, torch.tensor(y, dtype=torch.long)

train_ds = T2TextDataset(df_train)
dev_ds   = T2TextDataset(df_dev)
test_ds  = T2TextDataset(df_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Class weights for imbalance
labels_train = df_train.label_id.to_numpy()
class_counts = np.bincount(labels_train)
class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32, device=DEVICE)

print("\n✅ Task 2 text dataloader ready")

# ================== MODEL =====================
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
        out = self.txt(input_ids=B["input_ids"], attention_mask=B["attention_mask"]).last_hidden_state
        cls = out[:, 0, :]
        cls = self.ln(cls)
        return self.head(cls)

# ================== TRAINING ==================
def train_model():
    model = T2TextModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {"params": model.txt.parameters(),  "lr": LR_TXT},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_f1 = 0.0
    wait = 0
    
    for epoch in range(1, EPOCHS+1):
        print(f"\n===== T2T E{epoch}/{EPOCHS} =====")
        model.train()
        total_loss = 0.0
        for B, y in tqdm(train_loader, desc=f"Train E{epoch}"):
            y = y.to(DEVICE)
            B = {k:v.to(DEVICE) for k,v in B.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(B)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        print("Train loss:", total_loss / len(train_loader))
        
        # Dev eval
        model.eval()
        P, T_ = [], []
        with torch.no_grad():
            for B, y in tqdm(dev_loader, desc="Dev"):
                B = {k:v.to(DEVICE) for k,v in B.items()}
                logits = model(B)
                preds = logits.argmax(1).cpu().tolist()
                P.extend(preds)
                T_.extend(y.tolist())
        
        f1 = f1_score(T_, P, average="macro", zero_division=0)
        print("Dev F1:", f1)
        print("Confusion matrix:\n", confusion_matrix(T_, P))
        
        scheduler.step()
        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), "T2T_TEXT.pt")
            print("💾 Saved T2T_TEXT.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⛔ Early stopping")
                break
    
    print("Best Dev F1:", best_f1)
    return best_f1

# ================== TRAIN & TEST ==============
best_f1 = train_model()

# Test evaluation
model = T2TextModel().to(DEVICE)
model.load_state_dict(torch.load("T2T_TEXT.pt", map_location=DEVICE))
model.eval()

P, T_ = [], []
with torch.no_grad():
    for B, y in tqdm(test_loader, desc="Test"):
        B = {k:v.to(DEVICE) for k,v in B.items()}
        logits = model(B)
        preds = logits.argmax(1).cpu().tolist()
        P.extend(preds)
        T_.extend(y.tolist())

print("\n============== TASK 2 TEXT-ONLY TEST RESULTS ==============")
print(classification_report(T_, P, digits=4, target_names=[id2label[i] for i in range(NUM_CLASSES)], zero_division=0))
print("Confusion matrix:\n", confusion_matrix(T_, P))
print("Macro-F1:", f1_score(T_, P, average="macro", zero_division=0))
