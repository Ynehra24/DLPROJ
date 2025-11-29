########################################################################################################
# T1 — INFORMATIVE vs NOT INFORMATIVE  (FINAL — FIXED IMAGE MAPPING)
#
#  Text Model   : distilroberta-base  (RAM friendly)
#  Image Model  : CLIP ViT-B/32       (vision frozen)
#  Fusion Model : Late fusion — works now because image linking is fixed
#
#  Corrects TSV paths like:
#     data_image/california_wildfires/10_10_2017/9177.jpg
#
#  to actual dataset paths like:
#     /kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES/california_wildfires/10_10_2017/9177.jpg
#
########################################################################################################

import os, warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from collections import Counter
from sklearn.metrics import f1_score

from torchvision import transforms
from transformers import RobertaTokenizerFast, AutoModel, CLIPModel


# ==============================================================
# PATHS
# ==============================================================
BASE = "/kaggle/input/crisisman/ITSACRISIS/ITSACRISIS"
IMG  = "/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES"
OUT  = "/kaggle/working"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n🟢 DEVICE =",device)


# ==============================================================
# 🔥 CORRECT IMAGE INDEXING — TSV RELATIVE PATHS ARE USED
# ==============================================================
IMAGE_INDEX = {}

for root,_,files in os.walk(IMG):
    for file in files:
        full_path = os.path.join(root,file).replace("\\","/")

        # Convert absolute path -> TSV-style key:
        # IMG=/kaggle/input/crisisman/CRISISIMAGES/CRISISIMAGES
        # TSV wants: data_image/<...>
        rel = full_path.replace(IMG, "data_image")

        # allow match with & without .jpg
        IMAGE_INDEX[rel] = full_path
        IMAGE_INDEX[rel.replace(".jpg","")] = full_path
        IMAGE_INDEX[file] = full_path

print("📦 Images indexed =",len(IMAGE_INDEX))


# ==============================================================
# LOAD TSV
# ==============================================================
def load_T1():
    TRAIN,DEV=[],[]

    def read(p):
        with open(p,"r",encoding="utf8") as f:
            head=f.readline().strip().split("\t")
            return head,[l.strip().split("\t") for l in f]

    for split in ["train","dev"]:
        h,rows = read(f"{BASE}/task_informative_text_img_{split}.tsv")
        TXT = next(x for x in ["text","tweet","tweet_text"] if x in h)
        IMGID=next(x for x in ["image","image_id"] if x in h)
        LAB = next(x for x in ["label","label_text_image"] if x in h)

        for r in rows:
            d={h[i]:r[i] for i in range(len(r))}
            y=1 if d[LAB].lower()=="informative" else 0

            key_raw = d[IMGID].replace("\\","/")     # keep full path
            img = IMAGE_INDEX.get(key_raw,None)

            (TRAIN if split=="train" else DEV).append({
                "tweet":d[TXT],
                "img":img,
                "label":y
            })

    c=Counter(x["label"] for x in TRAIN)
    W=torch.tensor([len(TRAIN)/c[0],len(TRAIN)/c[1]],dtype=torch.float32).to(device)

    # report
    print("\n📊 DATA LOADED")
    print("TRAIN:",Counter(x["label"] for x in TRAIN))
    print("DEV  :",Counter(x["label"] for x in DEV))

    print(f"🖼 TRAIN with images = {sum(1 for x in TRAIN if x['img'])} / {len(TRAIN)}")
    print(f"🖼 DEV   with images = {sum(1 for x in DEV   if x['img'])} / {len(DEV)}")

    return TRAIN,DEV,W

TRAIN,DEV,CLASS_W = load_T1()


# ==============================================================
# TOKENIZER + DATALOADERS
# ==============================================================
tok = RobertaTokenizerFast.from_pretrained("distilroberta-base")

class TXT_DS(Dataset):
    def __init__(self,x): self.x=x
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        d=self.x[i]
        T=tok(d["tweet"],padding="max_length",truncation=True,max_length=72,return_tensors="pt")
        return {"ids":T.input_ids[0],"mask":T.attention_mask[0]},torch.tensor(d["label"])

TXT_TL=DataLoader(TXT_DS(TRAIN),batch_size=8,shuffle=True)
TXT_DL=DataLoader(TXT_DS(DEV),  batch_size=16,shuffle=False)


# ==============================================================
# IMAGE DATALOADER
# ==============================================================
T_img = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.481,0.457,0.408],[0.269,0.261,0.276])
])

class IMG_DS(Dataset):
    def __init__(self,x): self.x=[d for d in x if d["img"]]
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        d=self.x[i]
        img=Image.open(d["img"]).convert("RGB")
        return T_img(img),torch.tensor(d["label"])

IMG_TL=DataLoader(IMG_DS(TRAIN),batch_size=32,shuffle=True)
IMG_DL=DataLoader(IMG_DS(DEV),  batch_size=32,shuffle=False)


# ==============================================================
# FUSION COMBINED
# ==============================================================
class FUSION_DS(Dataset):
    def __init__(self,x): self.x=[d for d in x if d["img"]]
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        d=self.x[i]
        T=tok(d["tweet"],padding="max_length",truncation=True,max_length=72,return_tensors="pt")
        img=Image.open(d["img"]).convert("RGB")
        return {"ids":T.input_ids[0],"mask":T.attention_mask[0],"img":T_img(img)},torch.tensor(d["label"])

FUS_TRAIN=FUSION_DS(TRAIN)
FUS_DEV  =FUSION_DS(DEV)

print(f"\n🔗 FUSION SAMPLES TRAIN = {len(FUS_TRAIN)}")
print(f"🔗 FUSION SAMPLES DEV   = {len(FUS_DEV)}\n")

FUS_TL=DataLoader(FUS_TRAIN,batch_size=8,shuffle=True)
FUS_DL=DataLoader(FUS_DEV,  batch_size=8,shuffle=False)


# ==============================================================
# MODELS
# ==============================================================
class TEXT(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=AutoModel.from_pretrained("distilroberta-base")
        self.fc=nn.Sequential(nn.Linear(768,384),nn.GELU(),nn.Linear(384,2))
    def forward(self,B):
        x=self.m(input_ids=B["ids"],attention_mask=B["mask"]).last_hidden_state[:,0]
        return self.fc(x)

class IMG(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.m.vision_model.parameters(): p.requires_grad=False
        self.fc=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,2))
    def forward(self,X):
        with torch.no_grad(): feat=self.m.vision_model(X).pooler_output
        return self.fc(feat)

class FUSION(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt=AutoModel.from_pretrained("distilroberta-base")
        self.img=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.img.vision_model.parameters(): p.requires_grad=False
        self.txtp=nn.Linear(768,256)
        self.imgp=nn.Linear(768,256)
        self.fc =nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,2))
    def forward(self,B):
        t=self.txt(input_ids=B["ids"],attention_mask=B["mask"]).last_hidden_state[:,0]
        with torch.no_grad(): v=self.img.vision_model(B["img"]).pooler_output
        return self.fc(torch.cat([self.txtp(t),self.imgp(v)],dim=1))


# ==============================================================
# TRAIN TEXT / IMAGE / FUSION
# ==============================================================
def train_text(E=3):
    M=TEXT().to(device);opt=torch.optim.AdamW(M.parameters(),lr=3e-5);best=-1
    for ep in range(E):
        M.train()
        for B,y in tqdm(TXT_TL):
            B={k:v.to(device)for k,v in B.items()};y=y.to(device)
            loss=F.cross_entropy(M(B),y,weight=CLASS_W);opt.zero_grad();loss.backward();opt.step()
        M.eval();Y=[];P=[]
        with torch.no_grad():
            for B,y in TXT_DL:
                B={k:v.to(device)for k,v in B.items()};P+=M(B).argmax(1).cpu().tolist();Y+=y.tolist()
        f=f1_score(Y,P,average="macro");print("TEXT F1 =",f)
        if f>best: best=f;torch.save(M.state_dict(),OUT+"/T1_TEXT.pt")
    return best

def train_img(E=3):
    M=IMG().to(device);opt=torch.optim.AdamW(M.fc.parameters(),lr=3e-4);best=-1
    for ep in range(E):
        M.train()
        for X,y in tqdm(IMG_TL):
            X=X.to(device);y=y.to(device)
            loss=F.cross_entropy(M(X),y,weight=CLASS_W);opt.zero_grad();loss.backward();opt.step()
        M.eval();Y=[];P=[]
        with torch.no_grad():
            for X,y in IMG_DL:
                X=X.to(device);P+=M(X).argmax(1).cpu().tolist();Y+=y.tolist()
        f=f1_score(Y,P,average="macro");print("IMG F1 =",f)
        if f>best: best=f;torch.save(M.state_dict(),OUT+"/T1_IMAGE.pt")
    return best

def train_fusion(E=4):
    M=FUSION().to(device);best=-1
    opt=torch.optim.AdamW([
        {"params":M.txt.parameters(),"lr":3e-5},
        {"params":M.txtp.parameters(),"lr":5e-5},
        {"params":M.imgp.parameters(),"lr":5e-5},
        {"params":M.fc.parameters(),"lr":5e-5}],weight_decay=1e-2)
    for ep in range(E):
        M.train()
        for B,y in tqdm(FUS_TL):
            B={k:v.to(device)for k,v in B.items()};y=y.to(device)
            loss=F.cross_entropy(M(B),y,weight=CLASS_W)
            opt.zero_grad();loss.backward();opt.step()
        M.eval();Y=[];P=[]
        with torch.no_grad():
            for B,y in FUS_DL:
                B={k:v.to(device)for k,v in B.items()};P+=M(B).argmax(1).cpu().tolist();Y+=y.tolist()
        f=f1_score(Y,P,average="macro");print("FUSION F1 =",f)
        if f>best: best=f;torch.save(M.state_dict(),OUT+"/T1_FUSION.pt")
    return best


# ==============================================================
# RUN
# ==============================================================
print("\n🔥 TRAINING...\n")
print("TEXT F1:",train_text())
print("IMAGE F1:",train_img())
print("FUSION F1:",train_fusion())
print("\n📌 Models saved to /kaggle/working/")
