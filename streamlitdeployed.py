import os
import json
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import numpy as np

# Light imports only (safe on Streamlit Cloud)
try:
    import torch
    from torchvision import transforms
    from transformers import AutoTokenizer, AutoModel, CLIPModel
    import timm
    import torch.nn as nn
except Exception as e:
    import traceback
    err = traceback.format_exc()
    st.write("Import error detected:")
    st.code(err)
    # Continue, UI will still render

# ------------------------------------------------------------------
# Streamlit page config (must be FIRST)
# ------------------------------------------------------------------
st.set_page_config(page_title="CrisisMMD Assistant", layout="wide")

# ------------------------------------------------------------------
# Minimal configuration
# ------------------------------------------------------------------
DEVICE = "cpu"
CLIP_IMG_SIZE = 224
CONV_IMG_SIZE = 256
MAX_LEN = 96

OPENROUTER_API_URL = "https://api.openrouter.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

# ------------------------------------------------------------------
# OpenRouter wrapper
# ------------------------------------------------------------------
def call_openrouter(api_key: str, messages: List[Dict[str, str]],
                    model: str = DEFAULT_MODEL,
                    max_tokens: int = 512,
                    temperature: float = 0.2):

    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + api_key}
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------
# Architecture summary (ASCII only)
# ------------------------------------------------------------------
ARCHITECTURES = """
T1: DistilRoBERTa + CLIP-ViT-B32 fusion -> binary informative.
T2: DistilBERT text-only -> 3-class humanitarian.
T3: ConvNeXt-Tiny image + DistilBERT text -> 3-class damage.
T4: ConvNeXt-Tiny + DistilBERT -> 3-class human presence.
"""

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.title("CrisisMMD 4-task Assistant (Clean Cloud-Friendly Version)")

with st.sidebar:
    st.header("API Settings")
    api_key_input = st.text_input("OpenRouter API key", type="password")
    api_key = api_key_input or os.environ.get("OPENROUTER_API_KEY", "")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", 64, 2048, 512)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader("Upload image (RGB)", type=["png", "jpg", "jpeg"])
    uploaded_masks = st.file_uploader("Upload segmentation masks (optional)", accept_multiple_files=True)
    tweet_text = st.text_area("Original Tweet Text", height=160)
    disaster_type = st.text_input("Disaster Type (example: earthquake)")

with col2:
    st.write("Device:", DEVICE)
    st.write("Torch imported:", "torch" in globals())

# Preview image
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

# ------------------------------------------------------------------
# Run-button
# ------------------------------------------------------------------
if st.button("Generate Response"):
    if not uploaded_image:
        st.error("Upload an image.")
    elif not tweet_text.strip():
        st.error("Enter tweet text.")
    else:
        st.success("Image and text accepted. Preparing prompt...")

        # Simple placeholder outputs
        stub_outputs = {
            "caption": ["informative"]*4,
            "humcat": ["humanitarian"]*4,
            "damage": ["mild_damage"]*4,
            "localization": ["people_affected"]*4
        }

        st.subheader("Model Outputs (Stub)")
        st.json(stub_outputs)

        user_lines = []
        user_lines.append("Disaster Type: " + str(disaster_type))
        user_lines.append("Original Tweet:")
        user_lines.append(tweet_text)
        user_lines.append("")
        user_lines.append("Model Architectures:")
        user_lines.append(ARCHITECTURES)
        user_lines.append("")
        user_lines.append("Task Model Outputs:")

        for name, outs in stub_outputs.items():
            user_lines.append("--- " + name + " ---")
            for i, out in enumerate(outs):
                user_lines.append("Output " + str(i+1) + ": " + out)

        user_content = "\n".join(user_lines)

        messages = [
            {"role": "system", "content": "You are a crisis-response assistant."},
            {"role": "user", "content": user_content}
        ]

        if not api_key:
            st.error("Missing OpenRouter API key.")
        else:
            with st.spinner("Calling OpenRouter..."):
                try:
                    response = call_openrouter(api_key, messages, model=model_name,
                                               max_tokens=int(max_tokens),
                                               temperature=float(temperature))
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    st.subheader("Assistant Response")
                    st.text_area("Response", content, height=400)
                except Exception as e:
                    st.error("OpenRouter Error: " + str(e))
