###############################################################
#   🚀 T2 FUSION (RoBERTa-base + CLIP ViT-B/32)
#   ✅ Better training (less freezing, stronger head)
#   ✅ Class weights + sampler
#   ✅ AMP (no deprecation warnings if torch>=2.0)
###############################################################

from google.colab import drive
drive.mount('/content/drive')
print("🔗 Drive Mounted")

import os, random, torch, numpy as np, pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
from PIL import Image

# use torch.amp if available (Torch 2.x), else fallback to cuda.amp
try:
    from torch.amp import autocast, GradScaler
    AMP_KW = dict(device_type="cuda", dtype=torch.float16)
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    AMP_KW = dict()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Device:", device)

############################
#  ⚙ CONFIG
############################
IMG_SIZE     = 168
EPOCHS       = 10
ACCUM_STEPS  = 2
LR           = 3e-5       # 🔥 higher LR than before (was 8e-6)
BATCH_SIZE   = 8
VAL_BS       = 16
SEED         = 42

SAVE = "/content/drive/MyDrive/crisis_models/T2_FUSION_BETTER.pt"
os.makedirs(os.path.dirname(SAVE), exist_ok=True)

############################
#  SEEDING
############################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

############################
#  LOAD DATA
############################
BASE="/content/drive/MyDrive/crisismmd_datasplit_all"
IMG="/content/drive/MyDrive/data_image"

hum=pd.read_csv(f"{BASE}/task_humanitarian_text_img_train.tsv",sep="\t")
dam=pd.read_csv(f"{BASE}/task_damage_text_img_train.tsv",sep="\t")
inf=pd.read_csv(f"{BASE}/task_informative_text_img_train.tsv",sep="\t")

df=pd.concat([hum,dam,inf],ignore_index=True)

def map_t2(x):
    x=x.lower()
    if "not_humanitarian" in x or "not_informative" in x: 
        return 0
    if any(z in x for z in ["infra","damage","vehicle","severe","mild"]):
        return 2
    return 1

df["t2"]=df["label"].apply(map_t2)
print("\n📊 LABEL COUNTS:\n",df["t2"].value_counts(),"\n")

############################
#  INDEX IMAGES
############################
IMG_INDEX={}
for r,_,files in os.walk(IMG):
    for f in files:
        if f.lower().endswith(("jpg","png","jpeg")):
            IMG_INDEX[f.split('.')[0]] = os.path.join(r,f)

print("📦 Images indexed:",len(IMG_INDEX))

############################
#  TOKENIZER + CLIP
############################
print("🔤 Loading tokenizer...")
tok  = AutoTokenizer.from_pretrained("roberta-base")

print("🖼 Loading CLIP processor...")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

############################
#  DATASET
############################
class FusionDS(Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self,i):
        r = self.df.iloc[i]

        # text
        enc = tok(
            r["tweet_text"],
            padding="max_length",
            truncation=True,
            max_length=192,
            return_tensors="pt"
        )
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]

        # image
        imgid = str(r["image"]).split('.')[0]
        p = IMG_INDEX.get(imgid,None)
        if p and os.path.exists(p):
            img = Image.open(p).convert("RGB")
        else:
            img = Image.new("RGB",(IMG_SIZE,IMG_SIZE))

        img = img.resize((IMG_SIZE,IMG_SIZE))
        pix = proc(images=img,return_tensors="pt")["pixel_values"][0]

        label = torch.tensor(r["t2"],dtype=torch.long)
        return ids,mask,pix,label

############################
#  SPLIT + CLASS WEIGHTS
############################
train_df,val_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["t2"],
    random_state=SEED
)
y = train_df["t2"].values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
cw = torch.tensor(weights,dtype=torch.float32).to(device)
print("⚖ Class weights:", cw.tolist())

sampler = WeightedRandomSampler(
    weights=[cw[c].item() for c in y],
    num_samples=len(y),
    replacement=True
)

train_loader = DataLoader(
    FusionDS(train_df),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    pin_memory=True,
    num_workers=2
)
val_loader   = DataLoader(
    FusionDS(val_df),
    batch_size=VAL_BS,
    shuffle=False,
    pin_memory=True,
    num_workers=2
)

############################
#  FUSION MODEL
############################
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        print("🧠 Loading RoBERTa-base...")
        self.txt = AutoModel.from_pretrained("roberta-base")

        print("👁  Loading CLIP (ViT-B/32)...")
        self.vis = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # 🔒 Freeze fewer layers than before – still faster, but more capacity
        for name,p in self.txt.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(6)):  # 0–5 frozen
                p.requires_grad=False

        # freeze only very early CLIP blocks
        for name,p in self.vis.vision_model.named_parameters():
            if "layers.0" in name:        # only block0
                p.requires_grad=False

        # 🔁 Projection to smaller fusion space
        self.txt_proj = nn.Sequential(
            nn.Linear(768,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(768,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head = nn.Sequential(
            nn.Linear(512+512,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,3)
        )

    def forward(self,ids,mask,pix):
        t_out = self.txt(ids,mask).last_hidden_state[:,0]  # [CLS]
        v_out = self.vis.vision_model(pix).pooler_output

        t = self.txt_proj(t_out)
        v = self.img_proj(v_out)

        x = torch.cat([t,v],dim=1)
        logits = self.head(x)
        return logits

############################
#  TRAIN
############################
model = Fusion().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.03)
scaler = GradScaler()

best_f1 = 0.0

for ep in range(1,EPOCHS+1):
    model.train()
    epoch_loss = 0.0
    steps = 0

    loop = tqdm(train_loader, desc=f"🔥 FUSION E{ep}/{EPOCHS}")
    opt.zero_grad(set_to_none=True)

    for i,(ids,mask,pix,labels) in enumerate(loop):
        ids    = ids.to(device)
        mask   = mask.to(device)
        pix    = pix.to(device)
        labels = labels.to(device)

        with autocast(**AMP_KW):
            logits = model(ids,mask,pix)
            loss = criterion(logits,labels)
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        steps += 1

        if (i+1) % ACCUM_STEPS == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * ACCUM_STEPS
        loop.set_postfix(loss=epoch_loss / steps)

    # ================= VAL =================
    model.eval()
    P,T = [],[]
    with torch.no_grad():
        for ids,mask,pix,labels in val_loader:
            ids    = ids.to(device)
            mask   = mask.to(device)
            pix    = pix.to(device)
            labels = labels.to(device)

            with autocast(**AMP_KW):
                logits = model(ids,mask,pix)
            preds = logits.argmax(1)

            P += preds.cpu().tolist()
            T += labels.cpu().tolist()

    macro_f1 = f1_score(T,P,average="macro")
    print(f"\n🔥 VAL Macro F1 = {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), SAVE)
        print(f"⭐ NEW BEST ({best_f1:.4f}) → {SAVE}")

print("\n✅ TRAINING DONE")
print("🏆 BEST VAL MACRO F1:", best_f1)

############################
#  OPTIONAL: DETAILED VAL REPORT
############################
print("\n📄 Validation classification report:")
print(classification_report(T,P,digits=4))
