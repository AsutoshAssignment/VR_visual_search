
# =========================================================
# VISUAL PRODUCT SEARCH — Streamlit App
# =========================================================
# Pipeline:
#   Upload image
#   → YOLO detects ALL clothing items (multi-detection)
#   → User picks which item to search
#   → BLIP captions the crop
#   → CLIP fuses image + text embedding (α=0.7)
#   → HNSW ANN search → top-K candidates
#   → BLIP ITM re-ranks candidates
#   → Both raw (HNSW) and re-ranked (ITM) results shown in tabs
# =========================================================

import io
import hashlib
import clip
import faiss
import torch
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageDraw
from pathlib import Path
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
)
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG  — must be the very first Streamlit call
# =========================================================

st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0A0C10; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

.app-title {
    font-size: 1.9rem; font-weight: 600;
    color: #F0F0F0; letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 0.85rem; color: #5A6070;
    font-family: 'DM Mono', monospace;
}
.step-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #4A90D9; margin-bottom: 6px;
}
.caption-pill {
    display: inline-block; background: #1A2035;
    border: 1px solid #2A3550; border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem;
    color: #8AB4E8; font-style: italic; margin: 6px 0 10px;
}
.result-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; color: #4A90D9; margin-bottom: 2px;
}
.result-score {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
}
.score-itm  { color: #5BC8AF; }
.score-hnsw { color: #D4A94A; }
.divider { border: none; border-top: 1px solid #1A1E28; margin: 1rem 0; }

.stButton > button {
    background: #1A2540; border: 1px solid #2A3A5A;
    color: #8AB4E8; border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
}
.stButton > button:hover {
    background: #1E2E52; border-color: #4A90D9; color: #C8DAFF;
}
.stButton > button[kind="primary"] {
    background: #1E5FAA; border-color: #2A7AE0; color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
    background: #2468BB; border-color: #4A90D9;
}
.stSlider > div { padding: 0; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================

ROOT_DIR    = Path(__file__).resolve().parent.parent
CROPPED_DIR = ROOT_DIR / "data" / "cropped"
PROC_DIR    = ROOT_DIR / "data" / "processed"
EXP_C_DIR   = ROOT_DIR / "results" / "experiment_C"
MODELS_DIR  = ROOT_DIR / "models"

# =========================================================
# CONSTANTS
# =========================================================

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 only on CUDA; CPU cannot handle float16 ops in most torch builds
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

ALPHA    = 0.7
TOP_K    = 15
CONF_THR = 0.25
MAX_DET  = 10

# CLOTHING_CLASSES = {
#     0: "person", 24: "backpack", 25: "umbrella",
#     26: "handbag", 27: "tie", 28: "suitcase",
# }

# =========================================================
# MODEL LOADERS  (cached across sessions / reruns)
# =========================================================

@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    model = model.float()
    w = EXP_C_DIR / "clip_visual_finetuned.pth"
    if w.exists():
        model.visual.load_state_dict(torch.load(w, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss_index():
    for name in ("fashion_hnsw.index", "fashion_hnsw_best.index"):
        p = EXP_C_DIR / name
        if p.exists():
            return faiss.read_index(str(p))
    raise FileNotFoundError(
        f"HNSW index not found in {EXP_C_DIR}. "
        "Expected 'fashion_hnsw.index' or 'fashion_hnsw_best.index'."
    )


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo():
    for name in ("fashion_yolo.pt", "yolov8n.pt"):
        p = MODELS_DIR / name
        if p.exists():
            return YOLO(str(p))
    raise FileNotFoundError(
        f"No YOLO model found in {MODELS_DIR}. "
        "Expected 'fashion_yolo.pt' or 'yolov8n.pt'."
    )


@st.cache_resource(show_spinner="Loading BLIP captioning model…")
def load_blip_caption():
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE).eval()
    return proc, model


@st.cache_resource(show_spinner="Loading BLIP ITM model…")
def load_blip_itm():
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    # float16 only on CUDA — crashes on CPU
    model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco",
        torch_dtype=MODEL_DTYPE,
    ).to(DEVICE).eval()
    return proc, model


@st.cache_data(show_spinner="Loading gallery CSV…")
def load_gallery():
    path = PROC_DIR / "gallery.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gallery CSV not found at {path}")
    return pd.read_csv(path)


# =========================================================
# PIPELINE FUNCTIONS
# =========================================================

def _file_hash(uploaded_file) -> str:
    """Stable identity for an uploaded file based on its bytes."""
    uploaded_file.seek(0)
    digest = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest


def detect_clothing(image_pil: Image.Image, yolo_model) -> list[dict]:

    """
    Hybrid fashion detector.

    Strategy
    --------
    1. Run custom fashion YOLO
    2. If YOLO finds ONE dominant full-body region:
           create:
               - upper body
               - lower body
               - full outfit
    3. Else:
           use YOLO detections directly

    Returns
    -------
    list[dict]
    """

    W, H = image_pil.size

    img_np = np.array(image_pil)

    # =====================================================
    # YOLO INFERENCE
    # =====================================================

    res = yolo_model.predict(
        source=img_np,
        conf=CONF_THR,
        agnostic_nms=True,
        max_det=MAX_DET,
        verbose=False,
    )[0]

    # =====================================================
    # NO DETECTIONS
    # =====================================================

    if len(res.boxes) == 0:

        return [{
            "label": "full outfit",
            "confidence": 1.0,
            "bbox": [0, 0, W, H],
            "crop": image_pil.copy(),
        }]

    detections = []

    names   = res.names
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs   = res.boxes.conf.cpu().numpy()

    total_boxes = len(res.boxes)

    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for i in range(total_boxes):

        box = res.boxes.xyxy[i].cpu().numpy().astype(int)

        conf = float(confs[i])

        # -------------------------------------------------
        # FILTER WEAK DETECTIONS
        # -------------------------------------------------

        if conf < 0.35:
            continue

        cid = int(cls_ids[i])

        label = names.get(cid, f"item_{cid}")

        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(W, box[2])
        y2 = min(H, box[3])

        bw = x2 - x1
        bh = y2 - y1

        # -------------------------------------------------
        # FILTER TINY BOXES
        # -------------------------------------------------

        if bw < 20 or bh < 20:
            continue

        # =================================================
        # LARGE BODY DETECTION
        # =================================================

        large_detection = (
            bh > 0.55 * H or
            bw > 0.45 * W
        )

        # =================================================
        # HYBRID SPLIT MODE
        # Only when YOLO did NOT already find
        # multiple garment detections
        # =================================================

        if large_detection and total_boxes <= 2:

            upper_y2 = y1 + int(bh * 0.55)

            lower_y1 = y1 + int(bh * 0.45)

            upper_box = [
                x1,
                y1,
                x2,
                upper_y2,
            ]

            lower_box = [
                x1,
                lower_y1,
                x2,
                y2,
            ]

            full_box = [
                x1,
                y1,
                x2,
                y2,
            ]

            detections.extend([

                {
                    "label": "upper body",
                    "confidence": round(conf, 2),
                    "bbox": upper_box,
                    "crop": image_pil.crop(upper_box),
                },

                {
                    "label": "lower body",
                    "confidence": round(conf, 2),
                    "bbox": lower_box,
                    "crop": image_pil.crop(lower_box),
                },

                {
                    "label": "full outfit",
                    "confidence": round(conf, 2),
                    "bbox": full_box,
                    "crop": image_pil.crop(full_box),
                },

            ])

        # =================================================
        # DIRECT YOLO GARMENT DETECTION
        # =================================================

        else:

            bbox = [x1, y1, x2, y2]

            detections.append({

                "label": label,

                "confidence": round(conf, 2),

                "bbox": bbox,

                "crop": image_pil.crop(bbox),

            })

    # =====================================================
    # FALLBACK IF EVERYTHING FILTERED
    # =====================================================

    if not detections:

        return [{
            "label": "full outfit",
            "confidence": 1.0,
            "bbox": [0, 0, W, H],
            "crop": image_pil.copy(),
        }]

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique = []

    seen = set()

    for det in detections:

        key = (
            det["label"],
            tuple(det["bbox"]),
        )

        if key not in seen:

            unique.append(det)

            seen.add(key)

    return unique



def draw_detections(
    image_pil: Image.Image,
    detections: list[dict],
    selected_idx: int | None = None,
) -> Image.Image:
    img  = image_pil.copy()
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        sel   = (i == selected_idx)
        color = "#4A90D9" if sel else "#2A3550"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3 if sel else 1)
        lbl   = f"{det['label']} {det['confidence']:.0%}"
        tw    = len(lbl) * 6
        th    = 14
        tx    = x1 + 4
        ty = y1 + 4 if y1 < 25 else y1 - th - 2
        draw.rectangle([tx - 2, ty, tx + tw, ty + th + 2],
                       fill="#4A90D9" if sel else "#1A2035")
        draw.text((tx, ty + 1), lbl, fill="#FFFFFF")
    return img


@torch.no_grad()
def generate_caption(
    crop: Image.Image,
    proc: BlipProcessor,
    model: BlipForConditionalGeneration,
) -> str:
    inputs = proc(images=crop, return_tensors="pt").to(DEVICE)
    ids    = model.generate(**inputs, max_new_tokens=30, num_beams=3)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def embed_query(
    crop: Image.Image,
    caption: str,
    clip_model,
    preprocess,
) -> np.ndarray:
    img_t = preprocess(crop).unsqueeze(0).to(DEVICE)
    img_f = clip_model.encode_image(img_t).float()
    img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)

    tok   = clip.tokenize([caption], truncate=True).to(DEVICE)
    txt_f = clip_model.encode_text(tok).float()
    txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-8)

    fused = ALPHA * img_f + (1 - ALPHA) * txt_f
    fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)
    return fused.cpu().numpy().astype("float32")


def retrieve(
    query_emb: np.ndarray,
    index: faiss.Index,
) -> tuple[np.ndarray, np.ndarray]:
    q = query_emb.copy()
    faiss.normalize_L2(q)
    D, I = index.search(q, TOP_K)
    return I[0], D[0]


@torch.no_grad()
def itm_rerank(
    crop: Image.Image,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    gallery_df: pd.DataFrame,
    itm_proc: BlipProcessor,
    itm_model_obj: BlipForImageTextRetrieval,
) -> tuple[list[dict], list[dict]]:
    """
    Compute BLIP ITM cosine scores for all HNSW candidates.

    Returns
    -------
    reranked : list[dict]  — sorted by ITM score (best first)
    raw      : list[dict]  — original HNSW order, both scores attached
    """
    # Encode query image — cast to the model's dtype for consistency
    img_inp = itm_proc(images=crop, return_tensors="pt")
    pv      = img_inp["pixel_values"].to(DEVICE, dtype=MODEL_DTYPE)

    v_out    = itm_model_obj.vision_model(pixel_values=pv)
    img_feat = torch.nn.functional.normalize(
        itm_model_obj.vision_proj(
            v_out.last_hidden_state[:, 0, :].to(MODEL_DTYPE)
        ),
        dim=-1,
    )

    captions, valid_rows, valid_dists = [], [], []
    for pos, gi in enumerate(candidate_indices):
        if gi < 0 or gi >= len(gallery_df):
            continue
        row      = gallery_df.iloc[int(gi)]
        caption  = str(row.get("caption", "")) or "clothing item"
        captions.append(caption)
        valid_rows.append((int(gi), row))
        valid_dists.append(
            float(candidate_distances[pos])
            if pos < len(candidate_distances) else 0.0
        )

    if not captions:
        return [], []

    txt_inp  = itm_proc(
        text=captions, return_tensors="pt",
        padding=True, truncation=True, max_length=64,
    )
    q_out    = itm_model_obj.text_encoder(
        input_ids=txt_inp["input_ids"].to(DEVICE),
        attention_mask=txt_inp["attention_mask"].to(DEVICE),
    )
    txt_feat = torch.nn.functional.normalize(
        itm_model_obj.text_proj(
            q_out.last_hidden_state[:, 0, :].to(MODEL_DTYPE)
        ),
        dim=-1,
    )

    # img_feat: [1, D]   txt_feat: [N, D]  → scores: [N]
    itm_scores = (img_feat @ txt_feat.T).squeeze(0).float().cpu().numpy()

    raw = []
    for itm_s, dist, (gi, row) in zip(itm_scores, valid_dists, valid_rows):
        raw.append({
            "gallery_idx":    gi,
            "item_id":        row.get("item_id", ""),
            "image_path":     str(row.get("image_path", "")),
            "clothing_label": str(row.get("clothing_label", "full")),
            "ann_score":      dist,          # HNSW cosine similarity
            "itm_score":      float(itm_s),
        })

    reranked = sorted(raw, key=lambda x: x["itm_score"], reverse=True)
    return reranked, raw   # raw is already in HNSW order


def load_gallery_image(row_dict: dict) -> Image.Image | None:
    """
    Load a gallery result image, preferring a padded crop from the
    original source image (so side/back shots are not sliced).

    Resolution order:
      1. Original image + bbox with 20% padding  <- best quality
      2. Pre-cropped file under CROPPED_DIR/img/...
      3. Pre-cropped file under CROPPED_DIR/<label>/...
      4. Raw image_path treated as absolute path
    """
    img_rel = row_dict.get("image_path", "")
    if not img_rel:
        return None

    # ── Option 1: original image + padded bbox ────────────
    # image_path is "img/WOMEN/..." — strip the leading "img/" to get
    # the path relative to the images root, then try common layouts.
    rel_stripped = img_rel[4:] if img_rel.startswith("img/") else img_rel
    orig_candidates = [
        ROOT_DIR / "data" / "images" / rel_stripped,
        ROOT_DIR / "data" / "raw"    / rel_stripped,
        ROOT_DIR / "data" / "images" / img_rel,
        ROOT_DIR / "data" / "raw"    / img_rel,
    ]
    for orig_path in orig_candidates:
        if orig_path.exists():
            try:
                img = Image.open(orig_path).convert("RGB")
                x1 = row_dict.get("x1")
                y1 = row_dict.get("y1")
                x2 = row_dict.get("x2")
                y2 = row_dict.get("y2")
                if all(v is not None and not pd.isna(v)
                       for v in (x1, y1, x2, y2)):
                    W, H  = img.size
                    bw    = int(x2) - int(x1)
                    bh    = int(y2) - int(y1)
                    pad_x = int(bw * 0.20)   # 20% horizontal padding
                    pad_y = int(bh * 0.20)   # 20% vertical padding
                    cx1   = max(0, int(x1) - pad_x)
                    cy1   = max(0, int(y1) - pad_y)
                    cx2   = min(W, int(x2) + pad_x)
                    cy2   = min(H, int(y2) + pad_y)
                    img   = img.crop((cx1, cy1, cx2, cy2))
                return img
            except Exception:
                pass

    # ── Option 2: pre-cropped under CROPPED_DIR/img/... ───
    # Actual on-disk layout: data/cropped/img/WOMEN/...
    try:
        p = CROPPED_DIR / img_rel      # img_rel already starts with "img/"
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass

    # ── Option 3: pre-cropped under CROPPED_DIR/<label>/... ─
    try:
        label = row_dict.get("clothing_label", "full")
        rel   = Path(img_rel)
        for p in (
            CROPPED_DIR / label / rel,
            CROPPED_DIR / label / rel.name,
        ):
            if p.exists():
                return Image.open(p).convert("RGB")
    except Exception:
        pass

    # ── Option 4: treat image_path as absolute ────────────
    try:
        p = Path(img_rel)
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass

    return None


def render_result_grid(
    results: list[dict],
    score_key: str,
    score_label: str,
    score_css_class: str,
    n_cols: int = 5,
) -> None:
    """Shared grid renderer — works for both HNSW-order and ITM-order lists."""
    rows = [results[i : i + n_cols] for i in range(0, len(results), n_cols)]
    for r_idx, row_results in enumerate(rows):
        base = r_idx * n_cols
        cols = st.columns(n_cols)
        for j, (col, res) in enumerate(zip(cols, row_results)):
            rank = base + j + 1
            with col:
                img = load_gallery_image(res)
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:90px;background:#1A1E28;border-radius:8px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#3A4050;font-size:11px">no image</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="result-rank">#{rank}</div>'
                    f'<div class="result-score {score_css_class}">'
                    f'{score_label} {res[score_key]:.3f}</div>',
                    unsafe_allow_html=True,
                )


# =========================================================
# SESSION STATE INITIALISATION
# =========================================================

def init_state() -> None:
    defaults = {
        "file_hash":         None,   # MD5 of uploaded file bytes — stable identity
        "image_pil":         None,   # PIL.Image kept in state to avoid re-reads
        "detections":        [],
        "selected_det_idx":  None,
        "caption":           "",
        "results_reranked":  [],
        "results_raw":       [],
        "search_done":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =========================================================
# LOAD MODELS  (each loader has its own spinner label;
#               @st.cache_resource means they only run once
#               per server process, not per user session)
# =========================================================

clip_model, clip_preprocess = load_clip()
faiss_index                 = load_faiss_index()
yolo_model                  = load_yolo()
cap_proc, cap_model         = load_blip_caption()
itm_proc, itm_model_obj     = load_blip_itm()
gallery_df                  = load_gallery()

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:0.25rem">
  <span class="app-title">Visual Product Search</span>
  <span class="app-subtitle">CLIP · BLIP · HNSW · DeepFashion</span>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT  —  left panel : upload + detect + search
#            right panel: results
# =========================================================

left, right = st.columns([1, 1.6], gap="large")

# ─── LEFT PANEL ──────────────────────────────────────────
with left:

    st.markdown('<div class="step-label">Step 1 — Upload image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # ── Stable file identity via MD5 hash ────────────
        current_hash = _file_hash(uploaded)
        if st.session_state["file_hash"] != current_hash:
            # New file uploaded — reset all downstream state
            uploaded.seek(0)
            image_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.session_state.update({
                "file_hash":        current_hash,
                "image_pil":        image_pil,
                "detections":       [],
                "selected_det_idx": None,
                "caption":          "",
                "results_reranked": [],
                "results_raw":      [],
                "search_done":      False,
            })

        image_pil = st.session_state["image_pil"]

        # ── Run YOLO detection once per image ────────────
        if not st.session_state["detections"]:
            with st.spinner("Detecting clothing items…"):
                st.session_state["detections"] = detect_clothing(
                    image_pil, yolo_model
                )

        detections = st.session_state["detections"]
        sel_idx    = st.session_state["selected_det_idx"]

        # ── Annotated image preview ───────────────────────
        st.image(
            draw_detections(image_pil, detections, selected_idx=sel_idx),
            use_container_width=True,
        )

        # ── Item selection ────────────────────────────────
        st.markdown(
            '<div class="step-label" style="margin-top:12px">'
            'Step 2 — Choose clothing item</div>',
            unsafe_allow_html=True,
        )

        cols_per_row = 3
        det_rows = [
            detections[i : i + cols_per_row]
            for i in range(0, len(detections), cols_per_row)
        ]

        for r_idx, row_dets in enumerate(det_rows):
            base_idx = r_idx * cols_per_row
            row_cols = st.columns(len(row_dets))
            for j, (col, det) in enumerate(zip(row_cols, row_dets)):
                gidx   = base_idx + j
                is_sel = st.session_state["selected_det_idx"] == gidx
                with col:
                    thumb = det["crop"].copy()
                    thumb.thumbnail((120, 120))
                    st.image(thumb, use_container_width=True)
                    btn_label = f"✓ {det['label']}" if is_sel else det["label"]
                    if st.button(
                        btn_label,
                        key=f"det_{gidx}",
                        type="primary" if is_sel else "secondary",
                    ):
                        # Only update state; let the natural rerun handle the rest
                        st.session_state.update({
                            "selected_det_idx": gidx,
                            "caption":          "",
                            "results_reranked": [],
                            "results_raw":      [],
                            "search_done":      False,
                        })
                        # st.rerun() is safe here because we just changed state
                        st.rerun()

        # ── Search button ─────────────────────────────────
        if st.session_state["selected_det_idx"] is not None:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="step-label">Step 3 — Search</div>',
                        unsafe_allow_html=True)

            chosen = detections[st.session_state["selected_det_idx"]]
            st.markdown(
                f"Searching for **{chosen['label']}** "
                f"(conf {chosen['confidence']:.0%})"
            )

            top_k_val = st.slider(
                "Results to show",
                min_value=5,
                max_value=TOP_K,
                value=10,
                step=5,
                key="top_k_slider",
            )

            if st.button("🔍 Search", type="primary", use_container_width=True):
                crop = chosen["crop"]

                with st.spinner("Generating caption…"):
                    caption = generate_caption(crop, cap_proc, cap_model)
                    st.session_state["caption"] = caption

                with st.spinner("Embedding & retrieving…"):
                    q_emb           = embed_query(crop, caption, clip_model, clip_preprocess)
                    cand_idx, dists = retrieve(q_emb, faiss_index)

                with st.spinner("Re-ranking with BLIP ITM…"):
                    reranked, raw = itm_rerank(
                        crop, cand_idx, dists,
                        gallery_df, itm_proc, itm_model_obj,
                    )

                st.session_state.update({
                    "results_reranked": reranked[:top_k_val],
                    "results_raw":      raw[:top_k_val],
                    "search_done":      True,
                })
                # Rerun so the right panel renders the new results immediately
                st.rerun()

# ─── RIGHT PANEL ─────────────────────────────────────────
with right:

    if st.session_state["search_done"]:
        reranked = st.session_state["results_reranked"]
        raw      = st.session_state["results_raw"]

        if st.session_state["caption"]:
            st.markdown(
                f'<div class="caption-pill">"{st.session_state["caption"]}"</div>',
                unsafe_allow_html=True,
            )

        tab_itm, tab_hnsw = st.tabs([
            "✦ After ITM re-ranking",
            "◈ Before re-ranking  (HNSW order)",
        ])

        with tab_itm:
            st.markdown(
                '<div class="step-label" style="margin-bottom:10px">'
                'Sorted by BLIP ITM score — semantic match quality</div>',
                unsafe_allow_html=True,
            )
            if reranked:
                render_result_grid(reranked, "itm_score", "ITM", "score-itm")
            else:
                st.info("No results.")

        with tab_hnsw:
            st.markdown(
                '<div class="step-label" style="margin-bottom:10px">'
                'Raw HNSW retrieval — cosine similarity in embedding space</div>',
                unsafe_allow_html=True,
            )
            if raw:
                render_result_grid(raw, "ann_score", "ANN", "score-hnsw")
            else:
                st.info("No results.")

        # ── Rank-shift comparison table ───────────────────
        if reranked and raw:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="step-label">Rank shift after ITM re-ranking</div>',
                unsafe_allow_html=True,
            )

            hnsw_rank = {r["gallery_idx"]: i + 1 for i, r in enumerate(raw)}
            itm_rank  = {r["gallery_idx"]: i + 1 for i, r in enumerate(reranked)}

            table_rows = []
            for r in reranked:
                gi   = r["gallery_idx"]
                h_rk = hnsw_rank.get(gi, "—")
                i_rk = itm_rank.get(gi,  "—")
                if isinstance(h_rk, int) and isinstance(i_rk, int):
                    delta = h_rk - i_rk
                    shift = (
                        f"▲ {delta}"    if delta > 0 else
                        f"▼ {abs(delta)}" if delta < 0 else "—"
                    )
                else:
                    shift = "—"
                table_rows.append({
                    "ITM rank":  i_rk,
                    "HNSW rank": h_rk,
                    "Shift":     shift,
                    "ITM score": f"{r['itm_score']:.3f}",
                    "ANN score": f"{r['ann_score']:.3f}",
                })

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # ── Empty state placeholder ───────────────────────
        st.markdown("""
        <div style="height:60vh;display:flex;flex-direction:column;
             align-items:center;justify-content:center;
             text-align:center;gap:16px;">
          <div style="font-size:3rem;">🔍</div>
          <div style="font-size:1rem;font-weight:500;color:#3A4570;">
            Upload an image to begin
          </div>
          <div style="font-size:0.8rem;color:#2A3050;max-width:300px;line-height:1.7">
            YOLO detects clothing items → pick which to search →
            CLIP + BLIP retrieves and re-ranks similar products
          </div>
        </div>
        """, unsafe_allow_html=True)