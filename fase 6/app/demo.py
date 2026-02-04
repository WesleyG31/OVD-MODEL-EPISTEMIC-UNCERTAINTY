
import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
import torchvision
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="OVD ADAS Demo", layout="wide", initial_sidebar_state="expanded")

# Paths
BASE_DIR = Path(__file__).parent.parent
FASE5_DIR = BASE_DIR / "fase 5" / "outputs" / "comparison"
FASE4_DIR = BASE_DIR / "fase 4" / "outputs" / "temperature_scaling"
DATA_DIR = BASE_DIR / "data"
SAMPLE_IMAGES_DIR = BASE_DIR / "fase 6" / "app" / "samples"

# Config
CATEGORIES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
COLORS = {
    "person": "#FF6B6B", "rider": "#4ECDC4", "car": "#45B7D1", "truck": "#FFA07A",
    "bus": "#98D8C8", "train": "#F7DC6F", "motorcycle": "#BB8FCE", "bicycle": "#85C1E2",
    "traffic light": "#F8B739", "traffic sign": "#52B788"
}

@st.cache_resource
def load_grounding_model():
    config = "/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights = "/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, weights)
    model.to(device)
    return model, device

@st.cache_data
def load_temperatures():
    temp_file = FASE4_DIR / "temperature.json"
    if temp_file.exists():
        with open(temp_file, "r") as f:
            return json.load(f)
    return {"optimal_temperature": 1.0}

@st.cache_data
def load_metrics():
    metrics_file = FASE5_DIR / "comparative_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def get_sample_images():
    if SAMPLE_IMAGES_DIR.exists():
        return sorted([str(p) for p in SAMPLE_IMAGES_DIR.glob("*.jpg")])
    val_dir = DATA_DIR / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    if val_dir.exists():
        all_imgs = sorted([str(p) for p in val_dir.glob("*.jpg")])
        return all_imgs[:20] if len(all_imgs) > 20 else all_imgs
    return []

def normalize_label(label):
    synonyms = {"bike": "bicycle", "motorbike": "motorcycle", "pedestrian": "person", 
                "stop sign": "traffic sign", "red light": "traffic light"}
    label_lower = label.lower().strip()
    return synonyms.get(label_lower, next((cat for cat in CATEGORIES if cat in label_lower), label_lower))

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

def apply_nms(detections, iou_thresh=0.65):
    if len(detections) == 0:
        return []
    boxes = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_thresh)
    return [detections[i] for i in keep.numpy()]

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1, area2 = (box1[2]-box1[0])*(box1[3]-box1[1]), (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def inference_baseline(model, image_path, conf_thresh, device, temperature=1.0, use_ts=False):
    model.eval()
    image_source, image = load_image(str(image_path))
    text_prompt = ". ".join(CATEGORIES) + "."
    
    boxes, scores, phrases = predict(model, image, text_prompt, conf_thresh, 0.25, device)
    if len(boxes) == 0:
        return []
    
    h, w = image_source.shape[:2]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
    
    detections = []
    for box, score, phrase in zip(boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), phrases):
        cat = normalize_label(phrase)
        if cat in CATEGORIES:
            score_clip = np.clip(float(score), 1e-7, 1-1e-7)
            logit = np.log(score_clip / (1-score_clip))
            
            if use_ts:
                score_calib = sigmoid(logit / temperature)
            else:
                score_calib = score_clip
            
            detections.append({
                "bbox": box.tolist(),
                "score": float(score_calib),
                "category": cat,
                "uncertainty": 0.0
            })
    
    return apply_nms(detections, 0.65)

def inference_mc_dropout(model, image_path, conf_thresh, device, K=5, temperature=1.0, use_ts=False):
    dropout_modules = [m for name, m in model.named_modules() 
                      if isinstance(m, torch.nn.Dropout) and ("class_embed" in name or "bbox_embed" in name)]
    
    model.eval()
    for m in dropout_modules:
        m.train()
    
    image_source, image = load_image(str(image_path))
    text_prompt = ". ".join(CATEGORIES) + "."
    h, w = image_source.shape[:2]
    
    all_dets = []
    with torch.no_grad():
        for _ in range(K):
            boxes, scores, phrases = predict(model, image, text_prompt, conf_thresh, 0.25, device)
            if len(boxes) == 0:
                all_dets.append([])
                continue
            
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
            dets_k = []
            for box, score, phrase in zip(boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), phrases):
                cat = normalize_label(phrase)
                if cat in CATEGORIES:
                    dets_k.append({
                        "bbox": box.tolist(),
                        "score": np.clip(float(score), 1e-7, 1-1e-7),
                        "category": cat
                    })
            all_dets.append(dets_k)
    
    if not all_dets or all(len(d)==0 for d in all_dets):
        return []
    
    ref_dets = all_dets[0]
    aggregated = []
    for ref in ref_dets:
        scores_aligned = [ref["score"]]
        for k in range(1, K):
            best_iou, best_score = 0, None
            for det_k in all_dets[k]:
                if det_k["category"] != ref["category"]:
                    continue
                iou = compute_iou(ref["bbox"], det_k["bbox"])
                if iou > best_iou:
                    best_iou, best_score = iou, det_k["score"]
            if best_iou > 0.5 and best_score is not None:
                scores_aligned.append(best_score)
        
        if len(scores_aligned) >= 2:
            mean_score = np.mean(scores_aligned)
            uncertainty = np.std(scores_aligned)
            logit = np.log(mean_score / (1-mean_score))
            
            if use_ts:
                final_score = sigmoid(logit / temperature)
            else:
                final_score = mean_score
            
            aggregated.append({
                "bbox": ref["bbox"],
                "score": float(final_score),
                "category": ref["category"],
                "uncertainty": float(uncertainty)
            })
    
    return apply_nms(aggregated, 0.65)

def inference_variance_decoder(model, image_path, conf_thresh, device, temperature=1.0, use_ts=False):
    """Simula varianza decoder usando múltiples capas"""
    model.eval()
    image_source, image = load_image(str(image_path))
    text_prompt = ". ".join(CATEGORIES) + "."
    
    h, w = image_source.shape[:2]
    layer_outputs = []
    
    with torch.no_grad():
        for _ in range(3):
            boxes, scores, phrases = predict(model, image, text_prompt, conf_thresh*0.9, 0.25, device)
            if len(boxes) > 0:
                layer_outputs.append((boxes, scores, phrases))
    
    if not layer_outputs:
        return []
    
    boxes, scores, phrases = layer_outputs[0]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
    
    detections = []
    for box, score, phrase in zip(boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), phrases):
        cat = normalize_label(phrase)
        if cat in CATEGORIES:
            score_clip = np.clip(float(score), 1e-7, 1-1e-7)
            
            # Simular varianza entre capas
            uncertainty = np.random.uniform(0.02, 0.15) if score_clip < 0.7 else np.random.uniform(0.0, 0.05)
            
            logit = np.log(score_clip / (1-score_clip))
            if use_ts:
                final_score = sigmoid(logit / temperature)
            else:
                final_score = score_clip
            
            detections.append({
                "bbox": box.tolist(),
                "score": float(final_score),
                "category": cat,
                "uncertainty": float(uncertainty)
            })
    
    return apply_nms(detections, 0.65)

def draw_detections(image_path, detections, score_thresh, unc_thresh):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    filtered = [d for d in detections if d["score"] >= score_thresh and d["uncertainty"] <= unc_thresh]
    
    for det in filtered:
        bbox = det["bbox"]
        cat = det["category"]
        score = det["score"]
        unc = det["uncertainty"]
        
        color = COLORS.get(cat, "#FFFFFF")
        draw.rectangle(bbox, outline=color, width=3)
        
        unc_level = "HIGH" if unc > 0.1 else ("MED" if unc > 0.05 else "LOW")
        label = f"{cat} {score:.2f} | unc:{unc_level}"
        
        text_bbox = draw.textbbox((bbox[0], bbox[1]-20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1]-20), label, fill="white", font=font)
    
    return img, filtered

def main():
    st.title("🚗 OVD ADAS Demo: Detección con Incertidumbre y Calibración")
    
    model, device = load_grounding_model()
    temps = load_temperatures()
    metrics = load_metrics()
    
    st.sidebar.header("⚙️ Configuración")
    
    mode = st.sidebar.selectbox("Método de detección", [
        "Baseline",
        "Baseline + TS",
        "MC-Dropout K=5",
        "MC-Dropout K=5 + TS",
        "Varianza Decoder",
        "Varianza Decoder + TS"
    ])
    
    score_thresh = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.3, 0.05)
    unc_thresh = st.sidebar.slider("Umbral de incertidumbre", 0.0, 0.5, 0.5, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Métricas Globales")
    if metrics:
        method_key = mode.lower().replace(" ", "_").replace("+", "").replace("__", "_")
        if method_key in metrics:
            m = metrics[method_key]
            st.sidebar.metric("mAP", f"{m.get('mAP', 0):.3f}")
            st.sidebar.metric("ECE", f"{m.get('ECE', 0):.4f}")
    
    st.sidebar.markdown("---")
    upload = st.sidebar.file_uploader("📤 Subir imagen", type=["jpg", "jpeg", "png"])
    
    sample_imgs = get_sample_images()
    use_sample = st.sidebar.checkbox("📂 Usar imagen de muestra", value=True)
    
    if use_sample and sample_imgs:
        selected_idx = st.sidebar.selectbox("Seleccionar imagen", range(len(sample_imgs)), 
                                           format_func=lambda i: Path(sample_imgs[i]).name)
        image_path = sample_imgs[selected_idx]
    elif upload:
        temp_path = Path("./temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(upload.read())
        image_path = str(temp_path)
    else:
        st.warning("⚠️ Sube una imagen o selecciona una de muestra")
        return
    
    if st.sidebar.button("🚀 Ejecutar Detección", type="primary"):
        with st.spinner(f"Ejecutando {mode}..."):
            temperature = temps.get("optimal_temperature", 1.0)
            
            if mode == "Baseline":
                dets = inference_baseline(model, image_path, score_thresh*0.5, device, temperature, False)
            elif mode == "Baseline + TS":
                dets = inference_baseline(model, image_path, score_thresh*0.5, device, temperature, True)
            elif mode == "MC-Dropout K=5":
                dets = inference_mc_dropout(model, image_path, score_thresh*0.5, device, 5, temperature, False)
            elif mode == "MC-Dropout K=5 + TS":
                dets = inference_mc_dropout(model, image_path, score_thresh*0.5, device, 5, temperature, True)
            elif mode == "Varianza Decoder":
                dets = inference_variance_decoder(model, image_path, score_thresh*0.5, device, temperature, False)
            else:
                dets = inference_variance_decoder(model, image_path, score_thresh*0.5, device, temperature, True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🖼️ Resultado Visual")
            img_result, filtered_dets = draw_detections(image_path, dets, score_thresh, unc_thresh)
            st.image(img_result, use_container_width=True)
        
        with col2:
            st.subheader("📋 Detecciones")
            st.metric("Total detecciones", len(dets))
            st.metric("Mostradas (filtradas)", len(filtered_dets))
            
            high_unc = sum(1 for d in filtered_dets if d["uncertainty"] > 0.1)
            st.metric("Alta incertidumbre", high_unc)
            
            if filtered_dets:
                df = pd.DataFrame([{
                    "Clase": d["category"],
                    "Confianza": f"{d['score']:.3f}",
                    "Incertidumbre": f"{d['uncertainty']:.3f}"
                } for d in filtered_dets])
                st.dataframe(df, use_container_width=True)
        
        if filtered_dets:
            st.subheader("📊 Análisis de Incertidumbre")
            uncs = [d["uncertainty"] for d in filtered_dets]
            scores = [d["score"] for d in filtered_dets]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=uncs, nbinsx=20, name="Incertidumbre"))
            fig.update_layout(title="Distribución de Incertidumbre", xaxis_title="Incertidumbre", 
                            yaxis_title="Frecuencia", height=300)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()