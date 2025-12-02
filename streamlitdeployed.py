# streamlitdeployed.py
# Import probe for Streamlit Cloud - ASCII only, robust, prints tracebacks.
import importlib
import traceback
import sys
from typing import List, Tuple

import streamlit as st

st.set_page_config(page_title="Import Probe", layout="wide")
st.title("Import Probe — shows which module import fails on this environment")

# List modules to probe (adjust order if you like)
MODULES_TO_TEST: List[str] = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "transformers",
    "sklearn",
    "pandas",
    "numpy",
    "PIL",
    "requests",
    "streamlit",
]

def try_import(module_name: str) -> Tuple[bool, str]:
    """
    Try to import a module by name.
    Returns (success, message). Message is empty on success, otherwise traceback.
    """
    try:
        # handle special casing for PIL
        if module_name == "PIL":
            from PIL import Image  # type: ignore
        else:
            importlib.import_module(module_name)
        return True, ""
    except Exception:
        tb = traceback.format_exc()
        return False, tb

results = []
for m in MODULES_TO_TEST:
    success, msg = try_import(m)
    results.append((m, success, msg))

st.header("Import results")
ok = [r for r in results if r[1]]
bad = [r for r in results if not r[1]]

st.subheader("Successful imports")
if ok:
    for mod, _, _ in ok:
        st.write(f"- {mod} ✅")
else:
    st.write("None")

st.subheader("Failed imports (with traceback)")
if bad:
    for mod, _, tb in bad:
        st.write(f"### {mod} ❌")
        st.code(tb, language="python")
else:
    st.write("None")

st.markdown("---")
st.write("If an import fails with a long C-extension traceback, copy the first error lines and the last error lines and paste them into the logs or here so I can interpret them.")

# Extra: print a short sys.platform and python version
st.write("Runtime info:")
st.write(f"- python: {sys.version.replace(chr(10), ' ')}")
st.write(f"- platform: {sys.platform}")
