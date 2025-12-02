import os
import json
import base64
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# ------------------ Configuration ------------------
# This app calls OpenRouter's chat completions endpoint. You can set your
# OpenRouter API key in the environment variable OPENROUTER_API_KEY or paste
# it into the textbox in the app.
OPENROUTER_API_URL = "https://api.openrouter.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"  # change if you have a different router model

# ------------------ Helper functions ------------------

def call_openrouter(api_key: str, prompt_messages: List[Dict[str, str]],
                    model: str = DEFAULT_MODEL, max_tokens: int = 512,
                    temperature: float = 0.2) -> Dict[str, Any]:
    """Call the OpenRouter Chat Completions endpoint with given messages.
    prompt_messages should be a list like [{"role":"system","content":"..."}, ...]
    returns the parsed JSON response (or raises on error).
    """
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


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Overlay a single-channel mask (0 background, >0 mask) on an RGB image.
    The mask will be converted into a colored overlay.
    """
    image = image.convert("RGBA")
    mask = mask.convert("L").resize(image.size)

    # create color overlay from mask (we'll pick a strong color)
    color = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay = Image.composite(color, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)

    # blend overlay and original
    blended = Image.alpha_composite(image, ImageEnhance.Brightness(overlay).enhance(alpha))
    return blended.convert("RGB")


def display_segmentation(image: Image.Image, masks: List[Image.Image], titles: List[str]):
    st.image(image, caption="Original image", use_column_width=True)
    for i, m in enumerate(masks):
        over = overlay_mask_on_image(image, m, alpha=0.5)
        st.image(over, caption=f"Mask overlay: {titles[i] if i < len(titles) else i}", use_column_width=True)


def build_prompt(arch_defs: str, disaster_type: str, original_tweet: str,
                 model_outputs: Dict[str, List[str]], extra_instructions: str = "") -> List[Dict[str, str]]:
    """Construct messages for the chat model.
    model_outputs is a dict mapping task names -> list of outputs (or single-item list).
    """
    system_msg = (
        "You are an assistant specialized in crisis/social media multimodal analysis. "
        "Given model architecture details, segmentation masks, disaster type, original tweet and outputs from different task models, produce appropriate, actionable, and compassionate responses. "
        "Possible responses include: (1) an incident summary, (2) suggested hashtags, (3) a short reply for first responders, "
        "(4) instructions for affected people, (5) metadata to add to downstream pipelines (confidence, probable affected count), and (6) an explanation of why the reply was chosen."
    )

    # user-visible content
    user_lines = []
    user_lines.append(f"Disaster type: {disaster_type}")
    user_lines.append("Original tweet:")
    user_lines.append(original_tweet)
    user_lines.append("")
    user_lines.append("Model architecture definitions (raw):")
    user_lines.append(arch_defs)
    user_lines.append("")
    user_lines.append("Task model outputs:")
    for task_name, outputs in model_outputs.items():
        user_lines.append(f"--- {task_name} ---")
        for i, out in enumerate(outputs):
            user_lines.append(f"Output {i+1}: {out}")
    if extra_instructions:
        user_lines.append("")
        user_lines.append("Extra instructions:")
        user_lines.append(extra_instructions)

    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="CrisisMMD Streamlit — OpenRouter assistant", layout="wide")
st.title("CrisisMMD 4-task assistant (Streamlit + OpenRouter)")

st.markdown(
    "This app helps you assemble model metadata (architectures + outputs) and uses an OpenRouter chat model to generate actionable crisis responses. "
    "Set your OpenRouter API key below (or set env var OPENROUTER_API_KEY)."
)

# Sidebar: API key and options
with st.sidebar:
    st.header("API & settings")
    api_key_input = st.text_input("OpenRouter API key (or leave empty to use OPENROUTER_API_KEY env var)", type="password")
    api_key = api_key_input.strip() or os.environ.get("OPENROUTER_API_KEY", "")

    st.selectbox("Model", options=[DEFAULT_MODEL, "gpt-4o-mini", "gpt-4o"], index=0, key="model_select")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=512, step=64)
    st.markdown("---")
    st.markdown("**Upload image(s) and segmentation mask(s)**")
    uploaded_image = st.file_uploader("Upload segmented image (RGB) — usually from XBD", type=["png", "jpg", "jpeg"])
    uploaded_masks = st.file_uploader("Upload mask(s) (one mask per file, single-channel recommended)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Main inputs
st.subheader("Inputs")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("**Paste model architecture definitions**\n(Free-form text or JSON) — paste architectures for your task models here.")
    arch_defs = st.text_area("Architecture definitions", height=240, placeholder='Paste architecture JSON or text here')

    st.markdown("**Original tweet (text)**")
    original_tweet = st.text_area("Original tweet", height=120)

with col2:
    st.markdown("**Disaster type**")
    disaster_type = st.text_input("Disaster type (e.g., flood, earthquake)")

    st.markdown("**Task model outputs**\n(Enter outputs for each task model: Image-segmentation caption, damage classification, humanitarian categories, victim localization etc.)")
    task_names_str = st.text_input("Comma-separated task names (default: caption,damage,humcat,localization)", value="caption,damage,humcat,localization")

    # create dynamic inputs for each of the 4 outputs per task
    task_names = [t.strip() for t in task_names_str.split(",") if t.strip()]
    model_outputs: Dict[str, List[str]] = {}
    for tn in task_names:
        st.markdown(f"**Outputs for task: {tn}**")
        outs = [st.text_area(f"{tn} - output {i+1}", key=f"{tn}_{i}") for i in range(4)]
        model_outputs[tn] = outs

st.markdown("---")
extra_instructions = st.text_area("Extra instructions for the assistant (optional)")

# Visualization of uploaded image/masks
if uploaded_image:
    try:
        img = Image.open(uploaded_image).convert("RGB")
        masks_list = []
        for m in uploaded_masks:
            try:
                masks_list.append(Image.open(m))
            except Exception:
                st.warning(f"Could not open mask: {m.name}")
        st.subheader("Image & Masks preview")
        display_segmentation(img, masks_list, [m.name for m in uploaded_masks])
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")

# Trigger the assistant
if st.button("Generate response using OpenRouter"):
    if not api_key:
        st.error("No OpenRouter API key provided. Set it in the sidebar or in env var OPENROUTER_API_KEY.")
    else:
        with st.spinner("Calling OpenRouter..."):
            try:
                messages = build_prompt(arch_defs=arch_defs or "(none)", disaster_type=disaster_type or "(unknown)",
                                        original_tweet=original_tweet or "(none)", model_outputs=model_outputs,
                                        extra_instructions=extra_instructions)

                # use selected model/parameters
                selected_model = st.session_state.get("model_select", DEFAULT_MODEL)
                response_json = call_openrouter(api_key=api_key, prompt_messages=messages,
                                                model=selected_model, max_tokens=int(max_tokens), temperature=float(temperature))

                # parsing typical OpenRouter response structure
                # This block is defensive to handle variations in response format
                content = ""
                try:
                    # openrouter often returns choices[0].message.content
                    content = response_json.get("choices", [])[0].get("message", {}).get("content", "")
                except Exception:
                    content = json.dumps(response_json, indent=2)

                st.success("Response received")
                st.subheader("Assistant response")
                st.text_area("Response", value=content, height=400)

                st.subheader("Response as structured JSON (suggested)")
                # Ask the assistant to also produce structured JSON (we can ask a follow-up prompt)
                followup_prompt = [
                    {"role": "system", "content": system_msg_short()},
                    {"role": "user", "content": "Please return a JSON object with keys: summary, suggested_hashtags, short_reply, instructions_for_public, metadata. Use concise fields only."}
                ]
                # Defensive: call follow-up only if user wants it
                if st.checkbox("Also request structured JSON (extra call)"):
                    fu = [
                        {"role": "system", "content": "You are an assistant that returns ONLY valid JSON as a single object in your content."},
                        {"role": "user", "content": "Based on the previous messages, return a single JSON object with keys: summary, suggested_hashtags (array), short_reply (string), instructions_for_public (string), metadata (object)."}
                    ]
                    fu = messages + fu
                    fu_res = call_openrouter(api_key=api_key, prompt_messages=fu, model=selected_model, max_tokens=512, temperature=0.0)
                    try:
                        fu_content = fu_res.get("choices", [])[0].get("message", {}).get("content", "")
                        st.json(json.loads(fu_content))
                    except Exception:
                        st.text_area("Structured JSON (raw)", value=str(fu_res), height=200)

            except requests.HTTPError as he:
                st.error(f"OpenRouter API returned an HTTP error: {he} — {getattr(he, 'response', None)}")
            except Exception as e:
                st.exception(e)


# Small helper to provide a short system message for structured JSON follow-up
def system_msg_short():
    return (
        "You are a crisis-response assistant. Produce brief, factual, and compassionate outputs. If asked for JSON, return only valid JSON with no additional commentary."
    )

# Footer / run instructions
st.markdown("---")
st.markdown("**How to run:**\n1. Install dependencies: `pip install streamlit requests pillow numpy`\n2. Save this file and run: `streamlit run streamlit_crisismmd_openrouter.py`\n3. Provide your OpenRouter API key in the sidebar or set the environment variable `OPENROUTER_API_KEY`." )

# Optional: allow user to download the assembled prompt/messages
if st.button("Download assembled prompt (JSON)"):
    messages = build_prompt(arch_defs=arch_defs or "(none)", disaster_type=disaster_type or "(unknown)",
                            original_tweet=original_tweet or "(none)", model_outputs=model_outputs,
                            extra_instructions=extra_instructions)
    buff = BytesIO()
    buff.write(json.dumps(messages, indent=2).encode("utf-8"))
    buff.seek(0)
    st.download_button("Download prompt JSON", buff, file_name="crisismmd_prompt.json")
