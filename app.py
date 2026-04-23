import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model_def import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────── MODEL ─────────────────────────
@st.cache_resource
def load_model():
    model = get_model()

    state = torch.load("best_model.pth", map_location=DEVICE)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # remove `_orig_mod.` prefix if present
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            new_state[k.replace("_orig_mod.", "")] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ───────────────────── PREPROCESS ────────────────────────
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# ───────────────────── INFERENCE ─────────────────────────
def predict(model, image):
    x = preprocess(image).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    return mask.squeeze().cpu().numpy()

# ───────────────────── VISUALS ───────────────────────────
def overlay(image, mask):
    image = np.array(image.resize((512, 512)))
    mask = (mask * 255).astype(np.uint8)

    overlay = image.copy()
    overlay[mask > 0] = [52, 168, 235]  # water = blue

    return overlay

# ───────────────────── UI ────────────────────────────────
st.title("Water Segmentation")

uploaded_file = st.file_uploader(
    "Upload image (png/jpg/jpeg/tif)",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    mask = predict(model, image)
    overlay_img = overlay(image, mask)

    # ─── Side-by-side layout ───
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Input", use_container_width=True)

    with col2:
        st.image(mask, caption="Binary Mask", clamp=True, use_container_width=True)

    with col3:
        st.image(overlay_img, caption="Overlay", use_container_width=True)
