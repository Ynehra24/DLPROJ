import os
import json
import base64
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
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

# ------------------ User paths / config (edit to your environment) ------------------
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    res = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    res.raise_for_status()
    return res.json()

# ------------------ Model & data code (from user) ------------------
print("🟢 DEVICE:", DEVICE)

# Load dataset intersection (small sample used only for reference)
try:
    T1 = pd.read_csv(T1_FILE, sep="	")
    T2 = pd.read_csv(T2_FILE, sep="	")
    T3 = pd.read_csv(T3_FILE, sep="	")
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
    print(f"\n🔹 Loaded {len(eval_df)} aligned multimodal samples\n"")
except Exception as e:
    print("Could not load local dataset (continuing). Error:", e)
    eval_df = None

# Label mappers
T1_MAP = {"informative":1,"not_informative":0}

def map_t2(lbl: str) -> int:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
        "rescue_volunteering_or_donation_effort",
    ]:
        return 0
    elif lbl in [
        "infrastructure_and_utility_damage",
        "vehicle_damage",
    ]:
        return 2
    else:
        return 1

T3_MAP = {"little_or_no_damage":0,"mild_damage":1,"severe_damage":2}

def map_t4(lbl: str) -> int:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
    ]:
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

# Encoding helper for text

def enc(tok,text):
    x=tok(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True
    )
    return x["input_ids"].to(DEVICE),x["attention_mask"].to(DEVICE)

# ------------------ Model classes (as provided) ------------------
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
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,2)
        )
    def forward(self,ids,mask,img):
        t = self.txt(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        with torch.no_grad():
            v = self.img.vision_model(img).pooler_output
        return self.fc(torch.cat([self.txtp(t),self.imgp(v)],1))

class T2Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt  = AutoModel.from_pretrained("distilbert-base-uncased")
        H = self.txt.config.hidden_size
        self.ln   = nn.LayerNorm(H)
        self.head = nn.Sequential(
            nn.Linear(H,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,3)
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

# ------------------ Instantiate & load weights (if available) ------------------
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
    print("Warning: could not instantiate/load all models locally:", e)
    t1 = t2 = t3 = t4 = None

# ------------------ Helpers to run models on uploaded image + tweet ------------------

def pil_to_tensor_clip(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    return clip_tf(img).unsqueeze(0).to(DEVICE)

def pil_to_tensor_conv(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    return conv_tf(img).unsqueeze(0).to(DEVICE)

T1_LABELS = {0: "not_informative", 1: "informative"}
T2_LABELS = {0: "humanitarian", 1: "non_informative", 2: "structure"}
T3_LABELS = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
T4_LABELS = {0: "people_affected", 1: "rescue_needed", 2: "no_human"}


def run_models_on(pil_img: Image.Image, tweet_text: str) -> Dict[str, List[str]]:
    """Run T1-T4 on the provided PIL image and tweet text. Returns dict of task -> 4 outputs (strings).
    For simplicity we repeat the single prediction across 4 slots so UI/consumers expecting 4 outputs remain compatible."""
    outputs = {"caption": [], "damage": [], "humcat": [], "localization": []}

    if t1 is not None:
        try:
            ids,mask = enc(t1_tok,tweet_text)
            img_t1 = pil_to_tensor_clip(pil_img)
            with torch.no_grad():
                p = t1(ids,mask,img_t1).argmax().item()
            outputs["caption"] = [T1_LABELS.get(p,str(p))]*4
        except Exception as e:
            outputs["caption"] = [f"error:{e}"]*4
    else:
        outputs["caption"] = ["model_unavailable"]*4

    if t2 is not None:
        try:
            ids,mask = enc(t2_tok,tweet_text)
            with torch.no_grad():
                p = t2(ids,mask).argmax().item()
            outputs["humcat"] = [T2_LABELS.get(p,str(p))]*4
        except Exception as e:
            outputs["humcat"] = [f"error:{e}"]*4
    else:
        outputs["humcat"] = ["model_unavailable"]*4

    if t3 is not None:
        try:
            ids,mask = enc(t2_tok,tweet_text)
            img_t34 = pil_to_tensor_conv(pil_img)
            with torch.no_grad():
                p = t3(ids,mask,img_t34).argmax().item()
            outputs["damage"] = [T3_LABELS.get(p,str(p))]*4
        except Exception as e:
            outputs["damage"] = [f"error:{e}"]*4
    else:
        outputs["damage"] = ["model_unavailable"]*4

    if t4 is not None:
        try:
            ids,mask = enc(t2_tok,tweet_text)
            img_t34 = pil_to_tensor_conv(pil_img)
            with torch.no_grad():
                p = t4(ids,mask,img_t34).argmax().item()
            outputs["localization"] = [T4_LABELS.get(p,str(p))]*4
        except Exception as e:
            outputs["localization"] = [f"error:{e}"]*4
    else:
        outputs["localization"] = ["model_unavailable"]*4

    return outputs

# ------------------ Build architecture summary string ------------------
ARCHITECTURES = """
T1: DistilRoBERTa (text) + CLIP-ViT-B/32 (vision) fusion. Text proj 768->256, Image proj 768->256, merged -> 2-way classifier.
T2: DistilBERT (text-only) -> LayerNorm -> MLP -> 3-way humanitarian/non_informative/structure.
T3: ConvNeXt-Tiny (image features) + DistilBERT (text) fusion -> 3-way damage classifier.
T4: ConvNeXt-Tiny + DistilBERT fusion (same architecture as T3) -> 3-way people_affected/rescue_needed/no_human.
"""

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="CrisisMMD Streamlit — OpenRouter assistant (integrated)", layout="wide")
st.title("CrisisMMD 4-task assistant — models run on uploaded image + tweet")

st.markdown("Upload a satellite image (or photo), segmentation mask(s) (optional), and paste the original tweet. The backend will run your 4 models on that image + text, assemble outputs, and call an OpenRouter model to produce a crisis response.")

with st.sidebar:
    st.header("API & settings")
    api_key_input = st.text_input("OpenRouter API key (or leave empty to use OPENROUTER_API_KEY env var)", type="password")
    api_key = api_key_input.strip() or os.environ.get("OPENROUTER_API_KEY", "")
    model_name = st.selectbox("Model", options=[DEFAULT_MODEL, "gpt-4o-mini", "gpt-4o"], index=0, key="model_select")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=512, step=64)

col1, col2 = st.columns([2,1])
with col1:
    uploaded_image = st.file_uploader("Upload satellite image / photo (RGB)", type=["png","jpg","jpeg"])
    uploaded_masks = st.file_uploader("Upload mask(s) (optional)", accept_multiple_files=True, type=["png","jpg","jpeg"])
    original_tweet = st.text_area("Original tweet (text)", height=160)
    disaster_type = st.text_input("Disaster type (e.g., flood, earthquake)")

with col2:
    st.markdown("### Backend models & status")
    st.text("Device: {}".format(DEVICE))
    st.text("T1 loaded: {}".format('yes' if t1 is not None else 'no'))
    st.text("T2 loaded: {}".format('yes' if t2 is not None else 'no'))
    st.text("T3 loaded: {}".format('yes' if t3 is not None else 'no'))
    st.text("T4 loaded: {}".format('yes' if t4 is not None else 'no'))

# preview image + masks
if uploaded_image:
    try:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
        masks_list = []
        for m in uploaded_masks:
            try:
                masks_list.append(Image.open(m))
            except Exception:
                st.warning(f"Could not open mask: {m.name}")
        for i, m in enumerate(masks_list):
            st.image(m, caption=f"Mask {i}", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")

# run inference and generate
if st.button("Run models and generate response"):
    if not uploaded_image:
        st.error("Please upload an image first.")
    elif not original_tweet.strip():
        st.error("Please paste the original tweet text.")
    else:
        with st.spinner("Running models on uploaded image + tweet..."):
            try:
                task_outputs = run_models_on(pil_img, original_tweet)
                st.success("Models ran — outputs below")
                st.subheader("Task model outputs")
                st.json(task_outputs)

                # assemble prompt and call OpenRouter
                system_msg = (
                    "You are an assistant specialized in crisis/social media multimodal analysis. "
                    "Given model architecture details, segmentation masks, disaster type, original tweet and outputs from different task models, produce appropriate, actionable, and compassionate responses. "
                    "Possible responses include: (1) an incident summary, (2) suggested hashtags, (3) a short reply for first responders, "
                    "(4) instructions for affected people, (5) metadata to add to downstream pipelines (confidence, probable affected count), and (6) an explanation of why the reply was chosen."
                )

                user_lines = []
                user_lines.append(f"Disaster type: {disaster_type or '(unknown)'}")
                user_lines.append("Original tweet:")
                user_lines.append(original_tweet)
                user_lines.append("")
                user_lines.append("Model architecture definitions (raw):")
                user_lines.append(ARCHITECTURES)
                user_lines.append("")
                user_lines.append("Task model outputs:")
                for task_name, outputs in task_outputs.items():
                    user_lines.append(f"--- {task_name} ---")
                    for i, out in enumerate(outputs):
                        user_lines.append(f"Output {i+1}: {out}")

                user_content = ""
".join(user_lines)
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ]

                if not api_key:
                    st.warning("No OpenRouter API key provided. Set it in the sidebar or in env var OPENROUTER_API_KEY to call the model.")
                else:
                    with st.spinner("Calling OpenRouter..."):
                        res = call_openrouter(api_key=api_key, prompt_messages=messages, model=model_name, max_tokens=int(max_tokens), temperature=float(temperature))
                        try:
                            content = res.get("choices", [])[0].get("message", {}).get("content", "")
                        except Exception:
                            content = json.dumps(res, indent=2)
                        st.subheader("Assistant response")
                        st.text_area("Response", value=content, height=400)

                        # optional structured JSON followup
                        if st.checkbox("Also request structured JSON (extra call)"):
                            fu = [
                                {"role": "system", "content": "You are an assistant that returns ONLY valid JSON as a single object in your content."},
                                {"role": "user", "content": ""Based on the previous messages, return a single JSON object with keys: summary, suggested_hashtags (array), short_reply (string), instructions_for_public (string), metadata (object)."}
                            ]
                            fu = messages + fu
                            fu_res = call_openrouter(api_key=api_key, prompt_messages=fu, model=model_name, max_tokens=512, temperature=0.0)
                            try:
                                fu_content = fu_res.get("choices", [])[0].get("message", {}).get("content", "")
                                st.json(json.loads(fu_content))
                            except Exception:
                                st.text_area("Structured JSON (raw)", value=str(fu_res), height=200)

            except Exception as e:
                st.exception(e)

st.markdown("---")
st.markdown("**How to run:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run streamlit_crisismmd_openrouter.py`
3. Provide your OpenRouter API key in the sidebar or set the environment variable `OPENROUTER_API_KEY`." )

# Allow user to download the assembled prompt/messages
if st.button("Download assembled prompt (JSON)"):
    try:
        # Recreate last used variables if available
        # Note: This button will use current UI values
        user_lines = []
        user_lines.append(f"Disaster type: {disaster_type or '(unknown)'}")
        user_lines.append("Original tweet:")
        user_lines.append(original_tweet or "")
        user_lines.append("")
        user_lines.append("Model architecture definitions (raw):")
        user_lines.append(ARCHITECTURES)
        user_lines.append("")
        # attempt to include last task_outputs variable if present (not stored globally)
        # we'll include an empty placeholder if not present
        try:
            task_outputs
        except NameError:
            task_outputs = {"caption":[""],"damage":[""],"humcat":[""],"localization":[""]}

        user_lines.append("Task model outputs:")
        for task_name, outputs in task_outputs.items():
            user_lines.append(f"--- {task_name} ---")
            for i, out in enumerate(outputs):
                user_lines.append(f"Output {i+1}: {out}")

        messages = [
            {"role": "system", "content": system_msg_short()},
            {"role": "user", "content": "
".join(user_lines)}
        ]
        buff = BytesIO()
        buff.write(json.dumps(messages, indent=2).encode("utf-8"))
        buff.seek(0)
        st.download_button("Download prompt JSON", buff, file_name="crisismmd_prompt.json")
    except Exception as e:
        st.error(f"Could not assemble prompt: {e}")


def system_msg_short():
    return (
        "You are a crisis-response assistant. Produce brief, factual, and compassionate outputs. If asked for JSON, return only valid JSON with no additional commentary."
    )
