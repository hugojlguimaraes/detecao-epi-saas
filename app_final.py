# app_final.py (Streamlit 1.36.0 compatível)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

# ====== Configuração da página ======
st.set_page_config(
    page_title="EPI Detection - CCTV",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Detector Class ======
class EPIDetector:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None

        # mapeamento das classes do modelo
        self.epi_classes = {
            8: "glasses",
            9: "gloves",
            10: "helmet",
            14: "boots",
            15: "safety-suit",
            16: "safety-vest"
        }

    def detect_epis(self, frame, confidence=0.5):
        """Recebe PIL.Image ou np.ndarray"""
        if self.model is None:
            return None
        try:
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(frame)
            else:
                pil_image = frame
            results = self.model(pil_image, conf=float(confidence), verbose=False)
            return results[0] if results else None
        except Exception as e:
            st.error(f"Erro na detecção: {e}")
            return None

# ====== Função de desenho ======
def draw_detections_pil(pil_image, results, required_epis, confidence, epi_classes):
    if results is None or not hasattr(results, 'boxes'):
        return pil_image, [], list(required_epis or []), []

    boxes = getattr(results, 'boxes', [])
    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []

    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)

    # identificar EPIs detectados
    for box in boxes:
        try:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
        except Exception:
            continue
        if conf < confidence:
            continue
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            detected_epis.add(epi_name)
            if epi_name in missing_epis:
                missing_epis.remove(epi_name)

    # desenhar boxes
    for box in boxes:
        try:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            continue

        if conf < confidence:
            continue

        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            color = "green"
            label = f"{epi_name} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1-18)), label, fill=color)
        elif cls_id == 0:  # pessoa
            person_missing_epis = list(missing_epis.copy())
            color = "red" if person_missing_epis else "blue"
            label = "PESSOA" if not person_missing_epis else f"MISSING: {', '.join(person_missing_epis)}"
            if person_missing_epis:
                people_without_epi.append(person_missing_epis)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1-18)), label, fill=color)

    return draw_image, list(detected_epis), list(missing_epis), people_without_epi

# ====== Função principal de processamento ======
def process_image_file(pil_image, detector, required_epis, confidence):
    results = detector.detect_epis(pil_image, confidence)
    processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
        pil_image, results, required_epis, confidence, detector.epi_classes
    )

    # Layout em colunas
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Imagem Original")
        st.image(pil_image, use_column_width=True)  # <--- corrigido

    with col2:
        st.subheader("🎯 Imagem Processada")
        st.image(processed_image, use_column_width=True)  # <--- corrigido

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("EPIs Detectados", len(detected_epis))
        if detected_epis:
            st.write("✅ " + ", ".join(detected_epis))
    with col4:
        st.metric("EPIs Faltantes", len(missing_epis))
        if missing_epis:
            st.write("❌ " + ", ".join(missing_epis))
    with col5:
        st.metric("Pessoas sem EPI", len(people_without_epi))

# ====== Webcam com st.camera_input ======
def process_webcam_camera_input(detector, required_epis, confidence):
    st.header("🔴 WEBCAM (captura)")
    st.write("Tire uma foto para processar a detecção.")
    img_file_buffer = st.camera_input("Use a webcam")
    if img_file_buffer is not None:
        try:
            pil_image = Image.open(img_file_buffer).convert("RGB")
            process_image_file(pil_image, detector, required_epis, confidence)
        except Exception as e:
            st.error(f"Erro ao ler imagem da câmera: {e}")

# ====== Carregar modelo ======
@st.cache_resource
def load_model():
    model_paths = ["models/best.pt", "best.pt", "yolov8n.pt"]
    for path in model_paths:
        if os.path.exists(path):
            detector = EPIDetector(path)
            if detector.model is not None:
                return detector
    return EPIDetector("yolov8n.pt")

# ====== Main ======
def main():
    st.title("🛡️ SISTEMA DE DETECÇÃO DE EPI")

    # Sidebar
    st.sidebar.header("⚙️ CONFIGURAÇÕES")
    required_epis = st.sidebar.multiselect(
        "EPIs Obrigatórios:",
        ["helmet", "gloves", "safety-vest", "safety-suit", "glasses", "boots"],
        default=["helmet", "gloves"]
    )
    confidence = st.sidebar.slider("Confiança mínima:", 0.1, 0.9, 0.3, 0.05)
    image_source = st.sidebar.radio("Fonte de imagem:", ["Imagem", "Webcam"], index=0)

    # Carregar modelo
    detector = load_model()
    if detector.model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique os weights.")
        return

    st.header("🖼️ PROCESSAMENTO")
    if image_source == "Imagem":
        uploaded_file = st.file_uploader(
            "📤 Faça upload de uma imagem",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            process_image_file(pil_image, detector, required_epis, confidence)
    else:
        process_webcam_camera_input(detector, required_epis, confidence)

if __name__ == "__main__":
    main()
