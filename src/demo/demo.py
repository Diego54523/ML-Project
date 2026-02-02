import os
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms, datasets
import matplotlib.cm as cm

import joblib  # <-- NEW

# ==========================
# FIX PYTHONPATH (aggiunge src/)
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.Custom_CNN import CustomCNNFeatureExtractor


# ==========================
# CONFIG UI
# ==========================
st.set_page_config(page_title="MRI Classification Demo", layout="wide")

st.title("MRI Classification Demo")
st.caption("Carica un'immagine MRI e ottieni la predizione del modello. Dataset-mode supporta anche ground truth.")


# ==========================
# PATHS
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
RESULTS_DIR = PROJECT_ROOT / "results"

WEIGHTS_DIR = RESULTS_DIR / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "custom_cnn_classifier_BEST.pth"

# SVM artifacts
MODELS_DIR = RESULTS_DIR / "models"
SVM_MODEL_PATH = MODELS_DIR / "svm_best.joblib"
SVM_SCALER_PATH = MODELS_DIR / "svm_scaler.joblib"
SVM_META_PATH = MODELS_DIR / "svm_meta.json"  # opzionale
SVM_PLOT_PATH = (RESULTS_DIR / "plots" / "svm_pca_scatter.png")  # opzionale


# ==========================
# MODEL / DEVICE
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_DEFAULT = ["Mild", "Moderate", "Non", "Very Mild"]  # fallback


# ==========================
# TRANSFORMS (Custom CNN)
# ==========================
IMG_SIZE = (178, 208)
MEAN = [0.2954]
STD = [0.3207]

INFER_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ==========================
# UTILS
# ==========================
def list_weight_files(weights_dir: Path):
    if not weights_dir.exists():
        return []
    return sorted([p for p in weights_dir.glob("*.pth") if p.is_file()])


@st.cache_resource
def load_cnn_classifier(weights_path: str):
    """CNN end-to-end classifier (logits -> softmax)."""
    model = CustomCNNFeatureExtractor(num_classes=4)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_cnn_feature_extractor(weights_path: str):
    """Same CNN checkpoint, but decapitated for feature extraction."""
    model = CustomCNNFeatureExtractor(num_classes=4)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)

    # decapitation: output = features
    model.linear = torch.nn.Identity()

    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_svm_artifacts(model_path: str, scaler_path: str):
    svm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return svm, scaler


def pil_from_uploaded_file(uploaded_file) -> Image.Image:
    data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")


def predict_cnn(model, pil_img: Image.Image):
    x = INFER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    with torch.no_grad():
        logits = model(x)  # [1,4]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def extract_single_feature_vec(feat_model, pil_img: Image.Image):
    x = INFER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feat_model(x)  # [1, D]
    feat = feat.squeeze(0).detach().cpu().numpy()
    return feat


def predict_svm(feat_model, svm, scaler, pil_img: Image.Image):
    feat = extract_single_feature_vec(feat_model, pil_img)  # [D]
    feat_s = scaler.transform(feat.reshape(1, -1))          # [1, D]

    pred_idx = int(svm.predict(feat_s)[0])

    # Probabilities if available, else use decision_function scores (not calibrated)
    probs = None
    if hasattr(svm, "predict_proba"):
        probs = svm.predict_proba(feat_s).squeeze(0)
    elif hasattr(svm, "decision_function"):
        scores = svm.decision_function(feat_s)
        scores = np.array(scores).reshape(-1)
        # pseudo-prob: softmax on scores (ONLY for visualization)
        exp = np.exp(scores - scores.max())
        probs = exp / (exp.sum() + 1e-12)

    return pred_idx, probs


def find_last_conv_module(model: torch.nn.Module):
    last_name, last_mod = None, None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_name, last_mod = name, m
    return last_name, last_mod


def make_activation_heatmap_overlay(
    model: torch.nn.Module,
    pil_img: Image.Image,
    alpha: float = 0.4
) -> Image.Image | None:
    name, conv = find_last_conv_module(model)
    if conv is None:
        return None

    activ = {"fm": None}

    def hook_fn(module, inp, out):
        activ["fm"] = out.detach()

    handle = conv.register_forward_hook(hook_fn)

    try:
        x = INFER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(x)

        fm = activ["fm"]
        if fm is None:
            return None

        hm = fm[0].mean(dim=0)
        hm = torch.relu(hm)
        hm = hm - hm.min()
        if hm.max() > 0:
            hm = hm / hm.max()

        hm_np = hm.cpu().numpy()

        base = pil_img.convert("RGB").resize(IMG_SIZE)
        base_np = np.array(base).astype(np.float32) / 255.0

        hm_img = Image.fromarray((hm_np * 255).astype(np.uint8)).resize(IMG_SIZE)
        hm_resized = (np.array(hm_img).astype(np.float32) / 255.0)

        colored = cm.get_cmap("jet")(hm_resized)[..., :3]
        overlay = (1 - alpha) * base_np + alpha * colored
        overlay = np.clip(overlay, 0, 1)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        return Image.fromarray(overlay_uint8)

    finally:
        handle.remove()


# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Impostazioni")

weights_files = list_weight_files(WEIGHTS_DIR)
weights_options = [str(DEFAULT_WEIGHTS)] + [str(p) for p in weights_files if p != DEFAULT_WEIGHTS]

selected_weights = st.sidebar.selectbox(
    "Checkpoint Custom CNN (.pth)",
    options=weights_options if weights_options else [str(DEFAULT_WEIGHTS)],
    index=0
)

if not Path(selected_weights).exists():
    st.sidebar.error(
        f"Checkpoint non trovato:\n{selected_weights}\n\n"
        "Metti i pesi in results/weights/ oppure seleziona un file valido."
    )
    st.stop()

st.sidebar.write(f"Device: **{DEVICE}**")

classifier_choice = st.sidebar.radio(
    "Classificatore",
    ["Custom CNN (end-to-end)", "SVM su feature (Custom CNN)"],
    index=0
)

is_svm = classifier_choice.startswith("SVM")
show_confidences = not is_svm

mode = st.sidebar.radio("Modalità", ["Dataset mode (GT reale)", "Single image mode (GT N/A)"], index=0)

st.sidebar.subheader("Explainability")
show_heatmap = st.sidebar.checkbox("Mostra heatmap", value=False)
heatmap_alpha = st.sidebar.slider("Intensità heatmap", 0.0, 1.0, 0.4, 0.05)

# note: heatmap is meaningful for CNN features; for SVM it becomes "feature extractor activation overlay"
if classifier_choice.startswith("SVM"):
    st.sidebar.caption("Nota: la heatmap mostra attivazioni del feature extractor (non è una spiegazione diretta della decisione SVM).")


class_names = CLASS_NAMES_DEFAULT

# ==========================
# LOAD MODELS
# ==========================
cnn_model = load_cnn_classifier(selected_weights)
feat_model = load_cnn_feature_extractor(selected_weights)

svm_model, svm_scaler = None, None
if classifier_choice.startswith("SVM"):
    if not SVM_MODEL_PATH.exists() or not SVM_SCALER_PATH.exists():
        st.sidebar.error(
            "Artefatti SVM mancanti.\n"
            "Assicurati di avere:\n"
            f"- {SVM_MODEL_PATH}\n"
            f"- {SVM_SCALER_PATH}\n\n"
            "Esegui prima il training SVM."
        )
        st.stop()

    svm_model, svm_scaler = load_svm_artifacts(str(SVM_MODEL_PATH), str(SVM_SCALER_PATH))


# ==========================
# MAIN
# ==========================
col_left, col_right = st.columns([1, 1])

def run_inference(pil_img: Image.Image):
    """Returns (pred_idx, probs) based on selected classifier."""
    if classifier_choice.startswith("Custom CNN"):
        return predict_cnn(cnn_model, pil_img)
    else:
        return predict_svm(feat_model, svm_model, svm_scaler, pil_img)


if mode.startswith("Dataset mode"):
    st.subheader("Dataset mode")
    st.write("Seleziona una cartella dataset in formato ImageFolder (una sottocartella per classe).")

    load_dotenv()
    default_data_path = os.getenv("DATA_PATH", "")

    dataset_path = st.text_input("Percorso cartella dataset", value=default_data_path, placeholder=r"es. C:\...\MRI_dataset")
    if dataset_path and not Path(dataset_path).exists():
        st.error("Percorso dataset non valido.")
        st.stop()

    if dataset_path:
        ds = datasets.ImageFolder(root=dataset_path, transform=None)
        class_names = ds.classes

        st.success(f"Dataset caricato. Classi: {class_names}")

        samples = ds.samples
        if len(samples) == 0:
            st.error("Dataset vuoto.")
            st.stop()

        use_random = st.checkbox("Seleziona immagine random", value=True)
        if use_random:
            idx = np.random.randint(0, len(samples))
        else:
            display = [Path(p).name for p, _ in samples[:5000]]
            choice = st.selectbox("Scegli immagine", options=list(range(len(display))), format_func=lambda i: display[i])
            idx = int(choice)

        img_path, gt_label = samples[idx]
        gt_name = class_names[int(gt_label)]

        pil_img = Image.open(img_path).convert("RGB")

        heatmap_img = None
        if show_heatmap:
            # heatmap from CNN feature extractor (or classifier, similar)
            heatmap_img = make_activation_heatmap_overlay(feat_model, pil_img, alpha=heatmap_alpha)

        pred_idx, probs = run_inference(pil_img)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        with col_left:
            st.markdown("### Input")
            st.image(pil_img, caption=Path(img_path).name, use_container_width=True)
            if show_heatmap:
                if heatmap_img is None:
                    st.warning("Heatmap non disponibile: nessun layer Conv2d trovato nel modello.")
                else:
                    st.image(heatmap_img, caption="Heatmap (activation overlay)", use_container_width=True)

        with col_right:
            st.markdown("### Output")
            st.write(f"**Classificatore:** {classifier_choice}")
            st.write(f"**Ground Truth:** {gt_name}")
            st.write(f"**Predizione:** {pred_name}")

            if probs is None:
                st.info("Confidenze non disponibili (SVM senza probabilità).")
            else:
                if show_confidences:
                    st.write("**Confidenze:**")
                    for i, p in enumerate(probs):
                        name = class_names[i] if i < len(class_names) else f"class_{i}"
                        st.progress(float(p), text=f"{name}: {p:.3f}")

            if classifier_choice.startswith("SVM") and SVM_PLOT_PATH.exists():
                with st.expander("Visualizzazione SVM (PCA 2D)"):
                    st.image(str(SVM_PLOT_PATH), use_container_width=True)

else:
    st.subheader("Single image mode")
    uploaded = st.file_uploader("Carica un'immagine MRI", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Carica un file per vedere la predizione.")
        st.stop()

    pil_img = pil_from_uploaded_file(uploaded)

    heatmap_img = None
    if show_heatmap:
        heatmap_img = make_activation_heatmap_overlay(feat_model, pil_img, alpha=heatmap_alpha)

    pred_idx, probs = run_inference(pil_img)
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    with col_left:
        st.markdown("### Input")
        st.image(pil_img, caption=uploaded.name, use_container_width=True)
        if show_heatmap:
            if heatmap_img is None:
                st.warning("Heatmap non disponibile: nessun layer Conv2d trovato nel modello.")
            else:
                st.image(heatmap_img, caption="Heatmap (activation overlay)", use_container_width=True)

    with col_right:
        st.markdown("### Output")
        st.write(f"**Classificatore:** {classifier_choice}")
        st.write("**Ground Truth:** N/A (immagine caricata manualmente)")
        st.write(f"**Predizione:** {pred_name}")

        if probs is None:
            st.info("Confidenze non disponibili (SVM senza probabilità).")
        else:
            if show_confidences:
                st.write("**Confidenze:**")
                for i, p in enumerate(probs):
                    name = class_names[i] if i < len(class_names) else f"class_{i}"
                    st.progress(float(p), text=f"{name}: {p:.3f}")

        if classifier_choice.startswith("SVM") and SVM_PLOT_PATH.exists():
            with st.expander("Visualizzazione SVM (PCA 2D)"):
                st.image(str(SVM_PLOT_PATH), use_container_width=True)
