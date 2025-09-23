# app_final.py (corrigido)
import os
# Previne alguns problemas de bibliotecas paralelas
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import defensivo do ultralytics para evitar registro de signal handlers
try:
    # Import direto do engine/model evita carregar o "hub" que registra signal handlers
    from ultralytics.yolo.engine.model import YOLO
except Exception as e:
    import warnings
    warnings.warn(f"Import direto de ultralytics falhou ({e}). Tentando import padrão.")
    from ultralytics import YOLO

import streamlit as st
import numpy as np
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Criar pasta de imagens se não existir
if not os.path.exists("imagens"):
    os.makedirs("imagens")

def inserir_ocorrencia_arquivo(conforme, setores_id, cameras_id, tipos_epi_id, nome_arquivo, tipo_arquivo):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="RootMB@2025",
            database="epidetector",
            port=3306
        )

        if conn.is_connected():
            cursor = conn.cursor()
            sql_ocorrencia = """
                INSERT INTO ocorrencias (conforme, setores_id, cameras_id, tipos_epi_id)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql_ocorrencia, (conforme, setores_id, cameras_id, tipos_epi_id))
            ocorrencia_id = cursor.lastrowid

            sql_arquivo = """
                INSERT INTO arquivos (nome_arquivo, tipo_arquivo, ocorrencias_id)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql_arquivo, (nome_arquivo, tipo_arquivo, ocorrencia_id))

            conn.commit()
            print(f"✅ Ocorrência e arquivo inseridos com sucesso. Ocorrência ID: {ocorrencia_id}")

    except Error as e:
        print(f"❌ Erro ao conectar ou inserir: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


# ====== Configuração da página ======
st.set_page_config(
    page_title="EPI Detection - CCTV",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
    }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 12px; text-align: center; margin: 0.5rem; border-left: 4px solid #1E88E5; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .alert-box { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 1.2rem; border-radius: 12px; margin: 1rem 0; font-weight: bold; }
    .success-box { background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); color: white; padding: 1.2rem; border-radius: 12px; margin: 1rem 0; font-weight: bold; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.6rem 1rem; border-radius: 10px; font-weight: bold; width: 100%; margin-top: 1rem; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)

# ====== Detector class ======
class EPIDetector:
    def __init__(self, model_path):
        self.model = None
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None

        # classes (mapear conforme seu modelo)
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

            results = self.model(pil_image, verbose=False, conf=confidence)
            return results[0] if results else None
        except Exception as e:
            st.error(f"Erro na detecção: {e}")
            return None

# util (não estritamente usado no fluxo atual, mantido)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_detections_pil(pil_image, results, required_epis, confidence, epi_classes):
    """Desenha boxes/labels em PIL image. Retorna (image, detected_epis, missing_epis, people_without_epi)"""
    if results is None:
        return pil_image, [], list(required_epis or []), []

    boxes = getattr(results, 'boxes', None)
    if boxes is None:
        return pil_image, [], list(required_epis or []), []

    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []

    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Primeiro passe: identificar EPIs detectados
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

    # Segundo passe: desenhar
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

# load model com cache_resource
@st.cache_resource
def load_model():
    model_paths = [
        "models/best.pt",
        "runs/detect/epi_correction_training/weights/best.pt",
        "best.pt",
        "yolov8n.pt"
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                detector = EPIDetector(path)
                if detector.model is not None:
                    return detector
            except Exception:
                continue
    # fallback: tenta carregar yolov8n (pode falhar se não existir)
    return EPIDetector("yolov8n.pt")

# constantes
EPI_CLASSES = {
    8: "glasses", 9: "gloves", 10: "helmet", 14: "boots",
    15: "safety-suit", 16: "safety-vest"
}

def process_image_file(pil_image, detector, required_epis, confidence):
    """Processa uma PIL image e exibe resultados"""
    results = detector.detect_epis(pil_image, confidence)
    processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
        pil_image, results, required_epis, confidence, EPI_CLASSES
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Imagem Original")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.subheader("🎯 Imagem Processada")
        st.image(processed_image, use_container_width=True)

    st.subheader("📊 Estatísticas de Detecção")
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

    if missing_epis:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo_img = f"imagens/sem_epi_{timestamp}.jpg"
        processed_image.save(nome_arquivo_img)
        epi_id_map = {
            "boots": 4, "helmet": 1, "safety-suit": 5,
            "gloves": 3, "safety-vest": 6, "glasses": 2
        }
        for epi in missing_epis:
            tipos_epi_id = epi_id_map.get(epi, 1)
            inserir_ocorrencia_arquivo(
                conforme=0,
                setores_id=1,
                cameras_id=1,
                tipos_epi_id=tipos_epi_id,
                nome_arquivo=nome_arquivo_img,
                tipo_arquivo='imagem'
            )
        st.error(f"🚨 ALERTA: {len(missing_epis)} EPI(s) obrigatório(s) não detectado(s)!")
    else:
        st.success("✅ Todos os EPIs obrigatórios foram detectados!")

def process_webcam_camera_input(detector, required_epis, confidence):
    """Usa st.camera_input() — captura snapshots e processa"""
    st.header("🔴 WEBCAM (captura)")
    st.write("Clique em 'Capturar' para tirar uma foto com a câmera e processar.")

    # mostrar o componente de câmera
    img_file_buffer = st.camera_input("Use a webcam para capturar uma imagem")
    if img_file_buffer is not None:
        # converter para PIL
        try:
            pil_image = Image.open(img_file_buffer)
            if st.button("🎯 PROCESSAR CAPTURA"):
                process_image_file(pil_image, detector, required_epis, confidence)
        except Exception as e:
            st.error(f"Erro ao ler imagem da câmera: {e}")

# Envio de e-mail (mantive a lógica, ajuste credenciais fora do código em produção)
def send_email_alert(image_pil, subject, body, missing_epis=None):
    try:
        sender_email = "seu_email@gmail.com"
        sender_password = "sua_senha"
        receiver_email = "destinatario@gmail.com"
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        full_body = f"Data e Hora do Alerta: {now}\n\n"
        if missing_epis:
            full_body += ("Equipamentos Faltantes:\n- " + "\n- ".join(missing_epis) + "\n\n")
        full_body += body
        text = MIMEText(full_body)
        msg.attach(text)

        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image = MIMEImage(image_data, name="alerta_epi.jpg")
        msg.attach(image)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        # Feedback ao usuário
        try:
            st.success("📧 Alerta de e-mail enviado com sucesso!")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}. Verifique suas configurações de email.")

def main():
    st.markdown('<h1 class="main-header">🛡️ SISTEMA DE DETECÇÃO DE EPI</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURAÇÕES")
        st.subheader("🎯 EPIs Obrigatórios")
        required_epis = st.multiselect(
            "Selecione os EPIs obrigatórios:",
            ["helmet", "gloves", "safety-vest", "safety-suit", "glasses", "boots"],
            default=["helmet", "gloves"]
        )

        st.subheader("🔧 Configurações de Detecção")
        confidence = st.slider("Confiança mínima:", 0.1, 0.9, 0.3, 0.05)

        st.subheader("📷 Fonte de Imagem")
        image_source = st.radio("Selecione a fonte:", ["Imagem", "Webcam"], index=0)

        st.subheader("ℹ️ Informações")
        st.info("""
        **Classes detectáveis:**
        - 👷 Capacete (helmet)
        - 🧤 Luvas (gloves)
        - 🦺 Colete (safety-vest)
        - 🛡️ Macacão (safety-suit)
        - 👓 Óculos (glasses)
        - 👢 Botas (boots)
        """)

        st.subheader("📧 Teste de E-mail")
        if st.button("Enviar E-mail de Teste"):
            image_to_send = Image.new("RGB", (400, 80), color="white")
            draw = ImageDraw.Draw(image_to_send)
            draw.text((10, 10), "Imagem de teste do sistema EPI", fill="black")
            body = "Este é um e-mail de teste do sistema de detecção de EPI."
            subject = "E-mail de Teste do Sistema de Detecção de EPI"
            send_email_alert(image_to_send, subject, body)

    # Carregar modelo
    detector = load_model()
    if detector.model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique as configurações e os weights.")
        return

    st.header("🖼️ PROCESSAMENTO")

    if image_source == "Imagem":
        uploaded_file = st.file_uploader(
            "📤 Faça upload de uma imagem",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Formatos suportados: JPG, JPEG, PNG, BMP"
        )
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                if st.button("🎯 PROCESSAR IMAGEM"):
                    process_image_file(pil_image, detector, required_epis, confidence)
            except Exception as e:
                st.error(f"❌ Erro ao processar a imagem: {e}")

    elif image_source == "Webcam":
        # Usamos st.camera_input() para compatibilidade com Streamlit Cloud
        process_webcam_camera_input(detector, required_epis, confidence)

if __name__ == "__main__":
    main()
