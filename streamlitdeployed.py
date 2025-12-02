import os
import json
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------
# IMPORTANT: Streamlit page config MUST be first
# ----------------------------------------------------------------------
st.set_page_config(page_title="CrisisMMD Assistant", layout="wide")

# ----------------------------------------------------------------------
# Light imports (safe on Streamlit Cloud)
# ----------------------------------------------------------------------
try:
    import torch
    from torchvision import transforms
    from transformers import AutoTokenizer, AutoModel, CLIPModel
    import timm
    import torch.nn as nn
except Exception as e:
    st.write("Optional model imports failed (normal on Streamlit Cloud):")
    st.code(str(e))

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DEVICE = "cpu"
CLIP_IMG_SIZE = 224
CONV_IMG_SIZE = 256
MAX_LEN = 96

# Working OpenRouter endpoint (Streamlit Cloud compatible)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"

# Load the secret key from Streamlit secrets
SECRET_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

def get_api_key(user_input):
    if user_input.strip():
        return user_input.strip()
    return SECRET_KEY

# ----------------------------------------------------------------------
# FULL ARCHITECTURES (your exact descriptions)
# ----------------------------------------------------------------------
ARCHITECTURES = """
T1 — Informative vs Not Informative
Text: DistilRoBERTa-base, CLS token used (768d)
Image: CLIP ViT-B/32 vision tower (frozen, 768d)
Text projection: 768 → 256
Image projection: 768 → 256
Fusion: concat 512 → MLP → 2-class output

T2 — Humanitarian Category (3-class)
DistilBERT-base-uncased text encoder
CLS → LayerNorm → MLP(256) → 3-class head
Classes: humanitarian, non_informative, structure

T3 — Damage Severity (3-class)
Image: ConvNeXt-Tiny pretrained
Text: DistilBERT-base-uncased
Fusion: img + text → 256 → GELU → Dropout → 64 → GELU → 3-class head
Classes: little_or_no_damage, mild_damage, severe_damage

T4 — Human Presence / Rescue (3-class)
Same architecture as T3
Classes: people_affected, rescue_needed, no_human
"""

# ----------------------------------------------------------------------
# RAW TEXT–ONLY OpenRouter call (all 4 patches)
# ----------------------------------------------------------------------
def call_openrouter(api_key: str,
                    messages: List[Dict[str, str]],
                    model: str,
                    max_tokens: int,
                    temperature: float):

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
        "HTTP-Referer": "https://your-app.streamlit.app",
        "X-Title": "CrisisMMD Streamlit App"
    }

    # PATCH 3: Add provider format (disable markdown)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "provider": {"format": "text"}
    }

    r = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Extract text
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # PATCH 4: Sanitizer to remove common formatting characters
    forbidden = ["*", "#", "-", "`", "_", "•", ">", "|", "~"]
    for f in forbidden:
        content = content.replace(f, "")

    # Replace double spaces and leading/trailing whitespace
    content = content.strip()
    return content


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("CrisisMMD 4-Task Assistant (Raw Text Output)")

with st.sidebar:
    st.header("API Settings")

    st.write("Secret Key Loaded:", bool(SECRET_KEY))

    user_api_key = st.text_input("OpenRouter API key (optional override)", type="password")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", DEFAULT_MODEL])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.number_input("Max Tokens", min_value=64, max_value=2048, value=512)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader("Upload image (RGB)", type=["png", "jpg", "jpeg"])
    uploaded_masks = st.file_uploader("Upload segmentation masks (optional)", accept_multiple_files=True)
    tweet_text = st.text_area("Original Tweet", height=160)
    disaster_type = st.text_input("Disaster Type (e.g., flood, earthquake)")

with col2:
    st.write("Torch available:", "torch" in globals())
    st.write("Device:", DEVICE)


# Image preview
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)


# ----------------------------------------------------------------------
# Run: Generate response
# ----------------------------------------------------------------------
if st.button("Generate Crisis Response"):

    api_key = get_api_key(user_api_key)

    if not api_key:
        st.error("You must set OPENROUTER_API_KEY in Streamlit secrets or sidebar.")
        st.stop()

    if not uploaded_image:
        st.error("Please upload an image.")
        st.stop()

    if not tweet_text.strip():
        st.error("Enter tweet text.")
        st.stop()

    # Stub model outputs for now
    task_outputs = {
        "caption": ["informative"] * 4,
        "humcat": ["humanitarian"] * 4,
        "damage": ["mild_damage"] * 4,
        "localization": ["people_affected"] * 4
    }

    st.subheader("Model Outputs (Temporary)")
    st.json(task_outputs)

    # Construct user prompt
    lines = []
    lines.append("Disaster Type: " + disaster_type)
    lines.append("Original Tweet:")
    lines.append(tweet_text)
    lines.append("")
    lines.append("Model Architectures:")
    lines.append(ARCHITECTURES)
    lines.append("")
    lines.append("Task Model Outputs:")

    for k, outs in task_outputs.items():
        lines.append(f"{k}:")
        for i, o in enumerate(outs):
            lines.append(f"{o}")

    # PATCH 2: Force raw text requirement in user content
    lines.append("")
    lines.append("The output must be plain text only. Do not use markdown. Do not use bold. Do not use lists. Do not use formatting symbols. Use only plain sentences.")

    user_content = "\n".join(lines)

    # PATCH 1: Strong system instructions
    messages = [
        {
            "role": "system",
            "content": "Respond ONLY in raw plain text. No markdown, no headings, no bullet points, no asterisks, no hyphens, no hashes, no code blocks, no formatting of any kind. Only plain sentences."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    try:
        with st.spinner("Calling OpenRouter..."):
            response_text = call_openrouter(
                api_key=api_key,
                messages=messages,
                model=model_name,
                max_tokens=int(max_tokens),
                temperature=float(temperature)
            )

        st.subheader("Assistant Response (Raw Text Only)")
        st.text_area("Response", response_text, height=400)

    except Exception as e:
        st.error("OpenRouter Error: " + str(e))
