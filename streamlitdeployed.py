# streamlitdeployed.py
# Full integrated CrisisMMD Streamlit app with T1-T4 model inference + OpenRouter plain-text output.
# NOTES:
# - Put your OpenRouter API key in Streamlit secrets as OPENROUTER_API_KEY
# - Place model weight files at the paths defined below (or modify paths)
# - If weights are missing on the host, the app falls back to safe stubs

import os
import json
import re
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import numpy as np

# Page config MUST be first Streamlit UI call
st.set_page_config(page_title="CrisisMMD Assistant (Integrated)", layout="wide")

# ---------------------------
# Light imports (wrapped)
# ---------------------------
IMPORT_ERROR_MSG = None
# try to import heavy libs but do NOT call streamlit functions here
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    # RobertaTokenizerFast might not exist in some transformer builds; we'll fallback later
    try:
        from transformers import AutoTokenizer, AutoModel, CLIPModel, RobertaTokenizerFast
    except Exception:
        from transformers import AutoTokenizer, AutoModel, CLIPModel
        RobertaTokenizerFast = None
    import timm
except Exception as e:
    IMPORT_ERROR_MSG = str(e)

# ---------------------------
# Safe DEVICE selection (do not reference torch if not present)
# ---------------------------
if "torch" in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

CLIP_IMG_SIZE = 224
CONV_IMG_SIZE = 256
MAX_LEN = 96

# Local model weight paths - change if different
T1_PATH = "T1_FUSION_FINAL.pt"
T2_PATH = "T2T_TEXT.pt"
T3_PATH = "T3_MULTIMODAL.pt"
T4_PATH = "T4_MULTIMODAL.pt"

# OpenRouter endpoint (cloud-compatible)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model for the OpenRouter call - changed to Qwen per request
DEFAULT_LLM_MODEL = "qwen/qwen3-32b"

# ---------------------------
# Label mappings & helpers
# ---------------------------
T1_LABELS = {0: "not_informative", 1: "informative"}
T2_LABELS = {0: "humanitarian", 1: "non_informative", 2: "structure"}
T3_LABELS = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
T4_LABELS = {0: "people_affected", 1: "rescue_needed", 2: "no_human"}

def map_t2(lbl: str) -> int:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
        "rescue_volunteering_or_donation_effort",
    ]:
        return 0
    elif lbl in ["infrastructure_and_utility_damage", "vehicle_damage"]:
        return 2
    return 1

def map_t4(lbl: str) -> int:
    if lbl in [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
    ]:
        return 0
    elif lbl == "rescue_volunteering_or_donation_effort":
        return 1
    return 2

# ---------------------------
# Image transforms (only if torch available)
# ---------------------------
if "torch" in globals():
    clip_tf = transforms.Compose([
        transforms.Resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.481,0.457,0.408],[0.269,0.261,0.276])
    ])
    conv_tf = transforms.Compose([
        transforms.Resize((CONV_IMG_SIZE, CONV_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
else:
    clip_tf = conv_tf = None

def pil_to_tensor_clip(pil_img: Image.Image):
    if clip_tf is None:
        raise RuntimeError("torch not available")
    return clip_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

def pil_to_tensor_conv(pil_img: Image.Image):
    if conv_tf is None:
        raise RuntimeError("torch not available")
    return conv_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

# ---------------------------
# Model classes (same as your architectures)
# ---------------------------
if "torch" in globals():
    class T1_FUSION(nn.Module):
        def __init__(self):
            super().__init__()
            self.txt = AutoModel.from_pretrained("distilroberta-base")
            self.img = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # freeze the vision tower as in original setup
            for p in self.img.vision_model.parameters():
                p.requires_grad = False
            self.txtp = nn.Linear(768,256)
            self.imgp = nn.Linear(768,256)
            self.fc   = nn.Sequential(
                nn.Linear(512,256),
                nn.GELU(),
                nn.Linear(256,2)
            )
        def forward(self, ids, mask, img):
            t = self.txt(input_ids=ids, attention_mask=mask).last_hidden_state[:,0]
            with torch.no_grad():
                v = self.img.vision_model(img).pooler_output
            return self.fc(torch.cat([self.txtp(t), self.imgp(v)], dim=1))

    class T2Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.txt = AutoModel.from_pretrained("distilbert-base-uncased")
            H = self.txt.config.hidden_size
            self.ln = nn.LayerNorm(H)
            self.head = nn.Sequential(
                nn.Linear(H,256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256,3)
            )
        def forward(self, ids, mask):
            h = self.txt(input_ids=ids, attention_mask=mask).last_hidden_state[:,0]
            h = self.ln(h)
            return self.head(h)

    class T3T4(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.img = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            # compute d_img dynamically (safe since torch exists)
            with torch.no_grad():
                d_img = self.img(torch.zeros(1,3,CONV_IMG_SIZE,CONV_IMG_SIZE)).shape[-1]
            self.txt = AutoModel.from_pretrained("distilbert-base-uncased")
            d_txt = self.txt.config.hidden_size
            self.txt_ln = nn.LayerNorm(d_txt)
            self.head = nn.Sequential(
                nn.Linear(d_img + d_txt, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256,64),
                nn.GELU(),
                nn.Linear(64, num_classes)
            )
        def forward(self, ids, mask, img):
            i = self.img(img)
            t = self.txt(input_ids=ids, attention_mask=mask).last_hidden_state[:,0]
            t = self.txt_ln(t)
            return self.head(torch.cat([i, t], dim=1))
else:
    # placeholder classes if torch missing to avoid NameError on references (they won't be used)
    T1_FUSION = T2Text = T3T4 = None

# ---------------------------
# Instantiate models and load weights (attempt)
# ---------------------------
MODELS_AVAILABLE = False
t1 = t2 = t3 = t4 = None
t1_tok = t2_tok = None

if "torch" in globals():
    try:
        # instantiate architectures
        t1 = T1_FUSION().to(DEVICE)
        t2 = T2Text().to(DEVICE)
        t3 = T3T4(3).to(DEVICE)
        t4 = T3T4(3).to(DEVICE)

        # tokenizers
        try:
            if RobertaTokenizerFast is not None:
                t1_tok = RobertaTokenizerFast.from_pretrained("distilroberta-base", use_fast=True)
            else:
                t1_tok = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True)
        except Exception:
            t1_tok = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True)
        t2_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

        # load local weights if present
        def safe_load(model, path):
            if path and os.path.exists(path):
                try:
                    sd = torch.load(path, map_location=DEVICE)
                    model.load_state_dict(sd)
                    return True, None
                except Exception as e:
                    return False, str(e)
            return False, "path_not_found"

        loaded_ok = []
        ok, err = safe_load(t1, T1_PATH)
        loaded_ok.append(("T1", ok, err))
        ok, err = safe_load(t2, T2_PATH)
        loaded_ok.append(("T2", ok, err))
        ok, err = safe_load(t3, T3_PATH)
        loaded_ok.append(("T3", ok, err))
        ok, err = safe_load(t4, T4_PATH)
        loaded_ok.append(("T4", ok, err))

        # set eval mode
        for m in (t1, t2, t3, t4):
            if m is not None:
                m.eval()

        # check if tokenizers and models exist
        MODELS_AVAILABLE = all(x is not None for x in (t1, t2, t3, t4, t1_tok, t2_tok))
    except Exception as e:
        MODELS_AVAILABLE = False

# ---------------------------
# Helper: text encoding
# ---------------------------
def enc(tok, text):
    x = tok(text, return_tensors="pt", max_length=MAX_LEN, padding="max_length", truncation=True)
    return x["input_ids"].to(DEVICE), x["attention_mask"].to(DEVICE)

# ---------------------------
# Run models on a given PIL image and tweet text
# Returns dict of task -> list of 4 outputs (strings)
# ---------------------------
def run_models_on(pil_img: Image.Image, tweet_text: str):
    outputs = {"caption": [], "humcat": [], "damage": [], "localization": []}

    # If models are not available, return deterministic stubs
    if not MODELS_AVAILABLE or any(m is None for m in (t1,t2,t3,t4,t1_tok,t2_tok)):
        outputs["caption"] = ["informative"]*4
        outputs["humcat"] = ["humanitarian"]*4
        outputs["damage"] = ["mild_damage"]*4
        outputs["localization"] = ["people_affected"]*4
        return outputs

    # T1 (text+clip)
    try:
        i_ids, i_mask = enc(t1_tok, tweet_text)
        img_t1 = pil_to_tensor_clip(pil_img)
        with torch.no_grad():
            pred = t1(i_ids, i_mask, img_t1).argmax().item()
        outputs["caption"] = [T1_LABELS.get(pred, str(pred))]*4
    except Exception:
        outputs["caption"] = ["error"]*4

    # T2 (text-only)
    try:
        ids2, mask2 = enc(t2_tok, tweet_text)
        with torch.no_grad():
            pred = t2(ids2, mask2).argmax().item()
        outputs["humcat"] = [T2_LABELS.get(pred, str(pred))]*4
    except Exception:
        outputs["humcat"] = ["error"]*4

    # T3 (convnext + text)
    try:
        ids3, mask3 = enc(t2_tok, tweet_text)
        img_t34 = pil_to_tensor_conv(pil_img)
        with torch.no_grad():
            pred = t3(ids3, mask3, img_t34).argmax().item()
        outputs["damage"] = [T3_LABELS.get(pred, str(pred))]*4
    except Exception:
        outputs["damage"] = ["error"]*4

    # T4 (convnext + text)
    try:
        ids4, mask4 = enc(t2_tok, tweet_text)
        img_t44 = pil_to_tensor_conv(pil_img)
        with torch.no_grad():
            pred = t4(ids4, mask4, img_t44).argmax().item()
        outputs["localization"] = [T4_LABELS.get(pred, str(pred))]*4
    except Exception:
        outputs["localization"] = ["error"]*4

    return outputs

# ---------------------------
# Build architecture summary string (full)
# ---------------------------
ARCHITECTURES = (
    "T1: DistilRoBERTa (text) + CLIP-ViT-B/32 (vision) fusion. Text proj 768->256, Image proj 768->256, merged -> 2-way classifier.\n"
    "T2: DistilBERT (text-only) -> LayerNorm -> MLP -> 3-way humanitarian/non_informative/structure.\n"
    "T3: ConvNeXt-Tiny (image features) + DistilBERT (text) fusion -> 3-way damage classifier.\n"
    "T4: ConvNeXt-Tiny + DistilBERT fusion (same architecture as T3) -> 3-way people_affected/rescue_needed/no_human.\n"
)

# ---------------------------
# OpenRouter call (valid request body)
# Enforce raw text via system msg & appended user constraint; sanitize response after receive
# ---------------------------
def call_openrouter_valid(api_key: str, messages: List[Dict[str,str]], model: str, max_tokens: int, temperature: float):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # extract content
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Sanitize: remove markdown artifacts and disallowed characters
    content = sanitize_raw_text(content)
    return content

def sanitize_raw_text(text: str) -> str:
    if text is None:
        return ""
    # Remove code fences first
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove markdown-like tokens but be conservative to keep sentences
    text = text.replace("**", "").replace("__", "")
    # Remove standalone asterisks used for emphasis or bullets (but keep multiplication asterisks unlikely here)
    text = re.sub(r"(?m)^\s*\*\s*", "", text)
    text = text.replace("`", "")
    # Remove leading heading markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove common bullet characters at start of lines
    text = re.sub(r"(?m)^[\-\u2022]\s*", "", text)
    # Replace markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove footnote markers like [1], [^1]
    text = re.sub(r"\[\^?\d+\]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CrisisMMD 4-task assistant — Integrated models + OpenRouter (Raw Text)")

with st.sidebar:
    st.header("API & Model Settings")
    # Use secrets: user should put OPENROUTER_API_KEY into Streamlit secrets.toml
    secret_api = ""
    try:
        # st.secrets may raise if not configured; use .get safely
        secret_api = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        secret_api = os.environ.get("OPENROUTER_API_KEY", "")
    st.write("Secret API available:", bool(secret_api))
    sidebar_key = st.text_input("OpenRouter API key (optional override)", type="password")
    # ensure qwen is the default first option
    selected_model = st.selectbox("OpenRouter model", [DEFAULT_LLM_MODEL, "gpt-4o-mini", "gpt-4o"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.number_input("Max tokens", 64, 2048, 512)

col1, col2 = st.columns([2,1])
with col1:
    uploaded_image = st.file_uploader("Upload satellite / disaster image (RGB)", type=["png","jpg","jpeg"])
    uploaded_masks = st.file_uploader("Upload segmentation mask(s) (optional)", accept_multiple_files=True, type=["png","jpg","jpeg"])
    original_tweet = st.text_area("Original tweet text", height=160)
    disaster_type = st.text_input("Disaster type (e.g., flood)")

with col2:
    st.write("Device:", str(DEVICE))
    st.write("Torch available:", "torch" in globals())
    if IMPORT_ERROR_MSG:
        st.write("Optional import errors (ignored):")
        st.code(IMPORT_ERROR_MSG)
    # show whether model weights were found on disk (informational)
    mw_status = {
        "t1": os.path.exists(T1_PATH),
        "t2": os.path.exists(T2_PATH),
        "t3": os.path.exists(T3_PATH),
        "t4": os.path.exists(T4_PATH)
    }
    st.write("Model weight files present (paths):")
    st.json(mw_status)
    st.write("MODELS_AVAILABLE:", MODELS_AVAILABLE)

# preview images and masks
pil_img = None
mask_infos = []
if uploaded_image:
    try:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
    except Exception as e:
        st.error("Could not read uploaded image: " + str(e))

if uploaded_masks:
    for m in uploaded_masks:
        try:
            mi = Image.open(m).convert("L")
            mask_infos.append({"name": getattr(m, "name", "mask"), "size": mi.size})
            st.image(mi, caption=f"Mask: {getattr(m,'name','mask')}", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not open mask {getattr(m,'name','')}: {e}")

# ---------------------------
# Run inference & call OpenRouter
# ---------------------------
if st.button("Run models and get response"):
    api_key = (sidebar_key.strip() or secret_api or os.environ.get("OPENROUTER_API_KEY","")).strip()
    if not api_key:
        st.error("OpenRouter API key missing. Add to Streamlit secrets (OPENROUTER_API_KEY) or paste in sidebar.")
        st.stop()
    if pil_img is None:
        st.error("Please upload an image.")
        st.stop()
    if not original_tweet.strip():
        st.error("Please paste the original tweet text.")
        st.stop()

    with st.spinner("Running task models..."):
        task_outputs = run_models_on(pil_img, original_tweet)
    st.success("Models ran (or stubs used).")
    st.subheader("Task model outputs")
    st.json(task_outputs)

    # Prepare prompt: architectures, mask summary, outputs, disaster type, tweet
    prompt_lines = []
    prompt_lines.append("Disaster type: " + (disaster_type or "(unknown)"))
    prompt_lines.append("Original tweet:")
    prompt_lines.append(original_tweet)
    prompt_lines.append("")
    prompt_lines.append("Segmentation masks summary:")
    if mask_infos:
        for mi in mask_infos:
            prompt_lines.append(f"{mi.get('name','mask')} size {mi.get('size')}")
    else:
        prompt_lines.append("(no masks provided)")
    prompt_lines.append("")
    prompt_lines.append("Model architecture definitions (raw):")
    prompt_lines.append(ARCHITECTURES)
    prompt_lines.append("")
    prompt_lines.append("Task model outputs:")
    for tname, outs in task_outputs.items():
        prompt_lines.append(f"--- {tname} ---")
        for i,o in enumerate(outs):
            prompt_lines.append(f"Output {i+1}: {o}")
    # final instruction requiring plain raw text
    prompt_lines.append("")
    prompt_lines.append("The response MUST be plain text only. Do not use markdown, headings, bullets, numbering, emojis, asterisks, or any formatting characters. Use only sentences and newlines.")

    user_content = "\n".join(prompt_lines)

    system_msg = (
        "You are an assistant specialized in crisis/social media multimodal analysis. "
        "Respond ONLY in raw plain text. Do not use markdown, headings, bullets, numbering, symbols, emojis, or formatting of any kind. "
        "No asterisks, no hashes, no hyphens for list markers, no code blocks. Output must be plain sentences separated only by newlines."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

    # call OpenRouter
    try:
        with st.spinner("Calling OpenRouter..."):
            llm_text = call_openrouter_valid(api_key=api_key, messages=messages, model=selected_model, max_tokens=int(max_tokens), temperature=float(temp))
        st.subheader("Assistant response (raw text)")
        st.text_area("Response (raw)", llm_text, height=400)
    except requests.exceptions.HTTPError as he:
        # show both status and server text if available
        server_text = ""
        try:
            server_text = he.response.text
        except Exception:
            server_text = ""
        st.error(f"OpenRouter HTTP error: {he} - {server_text}")
    except Exception as e:
        st.error("OpenRouter error: " + str(e))

# ---------------------------
# Prompt download
# ---------------------------
if st.button("Download assembled prompt (JSON)"):
    try:
        # reuse last-known variables if present; otherwise return a minimal prompt
        try:
            task_outputs
        except NameError:
            task_outputs = {"caption":[""],"humcat":[""],"damage":[""],"localization":[""]}

        lines = []
        lines.append("Disaster type: " + (disaster_type or "(unknown)"))
        lines.append("Original tweet:")
        lines.append(original_tweet or "")
        lines.append("")
        lines.append("Model architectures:")
        lines.append(ARCHITECTURES)
        lines.append("")
        lines.append("Task model outputs:")
        for tname, outs in task_outputs.items():
            lines.append(f"--- {tname} ---")
            for i,o in enumerate(outs):
                lines.append(f"Output {i+1}: {o}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(lines)}
        ]

        buff = BytesIO()
        buff.write(json.dumps(messages, indent=2).encode("utf-8"))
        buff.seek(0)
        st.download_button("Download prompt JSON", buff, file_name="crisismmd_prompt.json")
    except Exception as e:
        st.error("Could not assemble prompt: " + str(e))

# ---------------------------
# End
# ---------------------------
