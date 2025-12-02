import os
import json
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import numpy as np

# ML libraries
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from transformers import (
    RobertaTokenizerFast,
    AutoTokenizer,
    AutoModel,
    CLIPModel
)
import timm
import torch.nn as nn

# ------------------ User paths / config ------------------
IMAGE_ROOT = "/Users/yatharthnehva/Desktop/CrisisMMD_v2.0/data_image"
T1_FILE = "/Users/yatharthnehva/Desktop/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv"
T2_FILE = "/Users/yatharthnehva/Desktop/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
T3_FILE = "/Users/yatharthnehva/Desktop/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv"

T1_PATH = "/Users/yatharthnehva/Desktop/ALL_MODELS/T1_FUSION_FINAL.pt"
T2_PATH = "/Users/yatharthnehva/Desktop/ALL_MODELS/T2T_TEXT.pt"
T3_PATH = "/Users/yatharthnehva/Desktop/ALL_MODELS/T3_MULTIMODAL.pt"
T4_PATH = "/Users/yatharthnehva/Desktop/ALL_MODELS/T4_MULTIMODAL.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_IMG_SIZE = 224
CONV_IMG_SIZE = 256
MAX_LEN = 96

OPENROUTER_API_URL = "https://api.openrouter.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

# ------------------ Utility functions ------------------
def call_openrouter(api_key: str, prompt_messages: List[Dict[str, str]],
                    model: str = DEFAULT_MODEL, max_tokens: int = 512,
                    temperature: float = 0.2) -> Dict[str, Any]:

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    res = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    res.raise_for_status()
    return res.json()

print("🟢 DEVICE:", DEVICE)

# ------------------ Try loading sample dataset ------------------
try:
    T1 = pd.read_csv(T1_FILE, sep="\t")
    T2 = pd.read_csv(T2_FILE, sep="\t")
    T3 = pd.read_csv(T3_FILE, sep="\t")

    eval_df = (
        T1[["tweet_id","tweet_text","image","label"]].rename(columns={"label":"t1_label"})
        .merge(
            T2[["tweet_id","tweet_text","image","label","label_text","label_text_image"]],
            on=["tweet_id","tweet_text","image"]
        )
        .merge(
            T3[["tweet_id","tweet_text","image","label"]].rename(columns={"label":"t3_label"}),
            on=["tweet_id","tweet_text","image"]
        )
    ).sample(10).reset_index(drop=True)

    print(f"\n🔹 Loaded {len(eval_df)} aligned multimodal samples\n")

except Exception as e:
    print("Could not load local dataset, continuing. Error:", e)
    eval_df = None

# ------------------ Label mappers ------------------
T1_MAP = {"informative":1,"not_informative":0}

def map_t2(lbl: str) -> int:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
        "rescue_volunteering_or_donation_effort",
    ]: return 0
    elif lbl in ["infrastructure_and_utility_damage", "vehicle_damage"]:
        return 2
    else:
        return 1

T3_MAP = {"little_or_no_damage":0,"mild_damage":1,"severe_damage":2}

def map_t4(lbl: str) -> int:
    if lbl in ["affected_individuals","injured_or_dead_people","missing_or_found_people"]:
        return 0
    elif lbl == "rescue_volunteering_or_donation_effort":
        return 1
    else:
        return 2

# ------------------ Image transforms ------------------
clip_tf = transforms.Compose([
    transforms.Resize((CLIP_IMG_SIZE,CLIP_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.481,0.457,0.408],[0.269,0.261,0.276])
])

conv_tf = transforms.Compose([
    transforms.Resize((CONV_IMG_SIZE,CONV_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

# ------------------ Text encoding ------------------
def enc(tok,text):
    x = tok(text, return_tensors="pt", max_length=MAX_LEN,
            padding="max_length", truncation=True)
    return x["input_ids"].to(DEVICE), x["attention_mask"].to(DEVICE)

# ------------------ Model classes ------------------
class T1_FUSION(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = AutoModel.from_pretrained("distilroberta-base")
        self.img = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.img.vision_model.parameters():
            p.requires_grad=False
        self.txtp = nn.Linear(768,256)
        self.imgp = nn.Linear(768,256)
        self.fc   = nn.Sequential(
            nn.Linear(512,256), nn.GELU(), nn.Linear(256,2)
        )

    def forward(self,ids,mask,img):
        t = self.txt(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        with torch.no_grad():
            v = self.img.vision_model(img).pooler_output
        return self.fc(torch.cat([self.txtp(t), self.imgp(v)],1))

class T2Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt  = AutoModel.from_pretrained("distilbert-base-uncased")
        H = self.txt.config.hidden_size
        self.ln   = nn.LayerNorm(H)
        self.head = nn.Sequential(
            nn.Linear(H,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,3)
        )

    def forward(self,ids,mask):
        h = self.txt(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        h = self.ln(h)
        return self.head(h)

class T3T4(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        self.img = timm.create_model("convnext_tiny",pretrained=True,num_classes=0)
        d_img = self.img(torch.zeros(1,3,CONV_IMG_SIZE,CONV_IMG_SIZE)).shape[-1]

        self.txt = AutoModel.from_pretrained("distilbert-base-uncased")
        d_txt = self.txt.config.hidden_size
        self.txt_ln = nn.LayerNorm(d_txt)

        self.head   = nn.Sequential(
            nn.Linear(d_img+d_txt,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,64),
            nn.GELU(),
            nn.Linear(64,num_classes)
        )

    def forward(self,ids,mask,img):
        i = self.img(img)
        t = self.txt(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        t = self.txt_ln(t)
        return self.head(torch.cat([i,t],1))

# ------------------ Load models ------------------
LOAD_LOCAL_MODELS = True

try:
    t1 = T1_FUSION().to(DEVICE)
    if LOAD_LOCAL_MODELS and os.path.exists(T1_PATH):
        t1.load_state_dict(torch.load(T1_PATH,map_location=DEVICE))
    t1.eval()
    t1_tok = RobertaTokenizerFast.from_pretrained("distilroberta-base")

    t2 = T2Text().to(DEVICE)
    if LOAD_LOCAL_MODELS and os.path.exists(T2_PATH):
        t2.load_state_dict(torch.load(T2_PATH,map_location=DEVICE))
    t2.eval()
    t2_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    t3 = T3T4(3).to(DEVICE)
    if LOAD_LOCAL_MODELS and os.path.exists(T3_PATH):
        t3.load_state_dict(torch.load(T3_PATH,map_location=DEVICE))
    t3.eval()

    t4 = T3T4(3).to(DEVICE)
    if LOAD_LOCAL_MODELS and os.path.exists(T4_PATH):
        t4.load_state_dict(torch.load(T4_PATH,map_location=DEVICE))
    t4.eval()

except Exception as e:
    print("Warning: could not load all models:", e)
    t1 = t2 = t3 = t4 = None

# ------------------ Image conversion ------------------
def pil_to_tensor_clip(pil_img: Image.Image):
    return clip_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

def pil_to_tensor_conv(pil_img: Image.Image):
    return conv_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

T1_LABELS = {0:"not_informative",1:"informative"}
T2_LABELS = {0:"humanitarian",1:"non_informative",2:"structure"}
T3_LABELS = {0:"little_or_no_damage",1:"mild_damage",2:"severe_damage"}
T4_LABELS = {0:"people_affected",1:"rescue_needed",2:"no_human"}

# ------------------ Run all 4 models ------------------
def run_models_on(pil_img: Image.Image, tweet_text: str) -> Dict[str, List[str]]:

    outputs = {"caption": [], "damage": [], "humcat": [], "localization": []}

    # T1
    try:
        ids,mask = enc(t1_tok,tweet_text)
        img_t1 = pil_to_tensor_clip(pil_img)
        p = t1(ids,mask,img_t1).argmax().item()
        outputs["caption"] = [T1_LABELS[p]]*4
    except: outputs["caption"] = ["error"]*4

    # T2
    try:
        ids,mask = enc(t2_tok,tweet_text)
        p = t2(ids,mask).argmax().item()
        outputs["humcat"] = [T2_LABELS[p]]*4
    except: outputs["humcat"] = ["error"]*4

    # T3
    try:
        ids,mask = enc(t2_tok,tweet_text)
        img_t34 = pil_to_tensor_conv(pil_img)
        p = t3(ids,mask,img_t34).argmax().item()
        outputs["damage"] = [T3_LABELS[p]]*4
    except: outputs["damage"] = ["error"]*4

    # T4
    try:
        ids,mask = enc(t2_tok,tweet_text)
        img_t34 = pil_to_tensor_conv(pil_img)
        p = t4(ids,mask,img_t34).argmax().item()
        outputs["localization"] = [T4_LABELS[p]]*4
    except: outputs["localization"] = ["error"]*4

    return outputs

# ------------------ Architecture summary ------------------
ARCHITECTURES = """
T1: DistilRoBERTa + CLIP-ViT-B/32 fusion → 2-way informative classifier.
T2: DistilBERT text-only → 3-way humanitarian/non_informative/structure.
T3: ConvNeXt-Tiny image + DistilBERT text fusion → 3-way damage classification.
T4: ConvNeXt-Tiny + DistilBERT fusion → 3-way human rescue/people/no_human.
"""

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="CrisisMMD Assistant", layout="wide")
st.title("CrisisMMD 4-task assistant — models run on uploaded image + tweet")

with st.sidebar:
    st.header("API & Settings")
    api_key_input = st.text_input("OpenRouter API key", type="password")
    api_key = api_key_input or os.environ.get("OPENROUTER_API_KEY","")
    model_name = st.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0,1.0,0.2,0.05)
    max_tokens = st.number_input("Max tokens", 64,2048,512)

col1, col2 = st.columns([2,1])
with col1:
    uploaded_image = st.file_uploader("Upload satellite image/photo (RGB)", type=["png","jpg","jpeg"])
    uploaded_masks = st.file_uploader("Upload optional segmentation masks", accept_multiple_files=True, type=["png","jpg","jpeg"])
    original_tweet = st.text_area("Original tweet", height=160)
    disaster_type = st.text_input("Disaster type (e.g., flood)")

with col2:
    st.write("Device:", DEVICE)
    st.write("T1 Loaded:", t1 is not None)
    st.write("T2 Loaded:", t2 is not None)
    st.write("T3 Loaded:", t3 is not None)
    st.write("T4 Loaded:", t4 is not None)

# ------------------ Image preview ------------------
if uploaded_image:
    pil_img = Image.open(uploaded_image).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

# ------------------ RUN MODELS + CALL OPENROUTER ------------------
if st.button("Run models and generate response"):
    if not uploaded_image:
        st.error("Upload an image first.")
    elif not original_tweet.strip():
        st.error("Enter tweet text.")
    else:
        with st.spinner("Running models..."):
            task_outputs = run_models_on(pil_img, original_tweet)
            st.subheader("Model Outputs")
            st.json(task_outputs)

        user_lines = []
        user_lines.append(f"Disaster type: {disaster_type}")
        user_lines.append("Original tweet:")
        user_lines.append(original_tweet)
        user_lines.append("")
        user_lines.append("Model architectures:")
        user_lines.append(ARCHITECTURES)
        user_lines.append("")
        user_lines.append("Task model outputs:")

        for t, outs in task_outputs.items():
            user_lines.append(f"--- {t} ---")
            for i,o in enumerate(outs):
                user_lines.append(f"Output {i+1}: {o}")

        user_content = "\n".join(user_lines)

        messages = [
            {"role":"system","content":"You are a crisis-response assistant. Produce structured, compassionate, actionable insights."},
            {"role":"user","content":user_content}
        ]

        if not api_key:
            st.error("Missing OpenRouter API key.")
        else:
            with st.spinner("Calling OpenRouter..."):
                res = call_openrouter(api_key, messages, model=model_name, max_tokens=int(max_tokens), temperature=float(temperature))

                content = res.get("choices",[{}])[0].get("message",{}).get("content","")
                st.subheader("Assistant Response")
                st.text_area("Response", content, height=400)

            # Optional structured JSON
            if st.checkbox("Request structured JSON output"):
                fu = [
                    {"role":"system","content":"Return ONLY valid JSON."},
                    {"role":"user","content":"Return JSON with: summary, suggested_hashtags[], short_reply, instructions_for_public, metadata{}."}
                ]
                fu_res = call_openrouter(api_key, messages+fu, model=model_name, max_tokens=512, temperature=0.0)

                try:
                    fu_content = fu_res["choices"][0]["message"]["content"]
                    st.json(json.loads(fu_content))
                except:
                    st.text_area("Raw JSON response", json.dumps(fu_res,indent=2))

# ------------------ Download prompt ------------------
if st.button("Download assembled prompt JSON"):
    try:
        user_lines = []
        user_lines.append(f"Disaster type: {disaster_type}")
        user_lines.append("Original tweet:")
        user_lines.append(original_tweet)
        user_lines.append("")
        user_lines.append("Model architecture definitions:")
        user_lines.append(ARCHITECTURES)
        user_lines.append("")

        try: task_outputs
        except: task_outputs = {"caption":[""],"damage":[""],"humcat":[""],"localization":[""]}

        user_lines.append("Task model outputs:")
        for t, outs in task_outputs.items():
            user_lines.append(f"--- {t} ---")
            for i,o in enumerate(outs):
                user_lines.append(f"Output {i+1}: {o}")

        messages = [
            {"role":"system","content":"Crisis-response assistant."},
            {"role":"user","content":"\n".join(user_lines)}
        ]

        buff = BytesIO()
        buff.write(json.dumps(messages, indent=2).encode())
        buff.seek(0)

        st.download_button("Download JSON", buff, file_name="crisismmd_prompt.json")

    except Exception as e:
        st.error(f"Failed: {e}")

# ------------------ System helper ------------------
def system_msg_short():
    return "Crisis response assistant. Return brief, factual, compassionate outputs."

