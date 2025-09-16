import streamlit as st
import numpy as np
import tempfile
import time
import os
from PIL import Image, ImageDraw
import io
import cv2
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Garantir pasta de imagens
if not os.path.exists("imagens"):
    os.makedirs("imagens")

# Evitar memory leak
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ------------------ Funções de banco ------------------ #
def inserir_ocorrencia_arquivo(conforme, setores_id, cameras_id, tipos_epi_id, nome_arquivo, tipo_arquivo):
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
            sql_ocorrencia = "INSERT INTO ocorrencias (conforme, setores_id, cameras_id, tipos_epi_id) VALUES (%s,%s,%s,%s)"
            cursor.execute(sql_ocorrencia, (conforme, setores_id, cameras_id, tipos_epi_id))
            ocorrencia_id = cursor.lastrowid

            sql_arquivo = "INSERT INTO arquivos (nome_arquivo, tipo_arquivo, ocorrencias_id) VALUES (%s,%s,%s)"
            cursor.execute(sql_arquivo, (nome_arquivo, tipo_arquivo, ocorrencia_id))

            conn.commit()
            print(f"✅ Ocorrência e arquivo inseridos. ID: {ocorrencia_id}")

    except Error as e:
        print(f"❌ Erro MySQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# ------------------ Detector de EPI ------------------ #
class EPIDetector:
    def __init__(self, model_path):
        try:
            # Carregar YOLO apenas offline (evita Hub)
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None

        self.epi_classes = {
            8: "glasses", 9: "gloves", 10: "helmet", 
            14: "boots", 15: "safety-suit", 16: "safety-vest"
        }

    def detect_epis(self, frame, confidence=0.5):
        if self.model is None:
            return None
        try:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            results = self.model(frame, verbose=False, conf=confidence)
            return results[0] if results else None
        except Exception as e:
            st.error(f"Erro detecção: {e}")
            return None

# ------------------ Funções auxiliares ------------------ #
def draw_detections_pil(pil_image, results, required_epis, confidence, epi_classes):
    if results is None or results.boxes is None:
        return pil_image, [], [], []

    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []

    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)

    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        if conf < confidence:
            continue
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            detected_epis.add(epi_name)
            missing_epis.discard(epi_name)

    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        if conf < confidence:
            continue
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1-25), f"{epi_name} {conf:.2f}", fill="green")
        elif cls_id == 0:
            person_missing_epis = list(missing_epis)
            color = "red" if person_missing_epis else "blue"
            label = f"MISSING: {', '.join(person_missing_epis)}" if person_missing_epis else "PESSOA"
            people_without_epi.append(person_missing_epis)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1-25), label, fill=color)

    return draw_image, list(detected_epis), list(missing_epis), people_without_epi

# ------------------ Carregar modelo com Streamlit Cache ------------------ #
@st.cache_resource
def load_model():
    model_paths = ["models/best.pt", "best.pt", "yolov8n.pt"]
    for path in model_paths:
        if os.path.exists(path):
            detector = EPIDetector(path)
            if detector.model:
                return detector
    return EPIDetector("yolov8n.pt")

EPI_CLASSES = {
    8: "glasses", 9: "gloves", 10: "helmet", 
    14: "boots", 15: "safety-suit", 16: "safety-vest"
}

# ------------------ E-mail ------------------ #
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
        full_body = f"Data e Hora: {now}\n\n"
        if missing_epis:
            full_body += "Equipamentos Faltantes:\n- " + "\n- ".join(missing_epis) + "\n\n"
        full_body += body
        msg.attach(MIMEText(full_body))

        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG")
        msg.attach(MIMEImage(buffer.getvalue(), name="alerta_epi.jpg"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.toast("📧 Alerta de e-mail enviado!")
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}")

# ------------------ Processamento ------------------ #
def process_webcam(detector, required_epis, confidence):
    st.header("🔴 WEBCAM AO VIVO")
    if "cap" not in st.session_state:
        st.session_state.cap = None
        st.session_state.webcam_image_saved = False

    col1, col2 = st.columns(2)
    start_pressed = col1.button("Iniciar")
    stop_pressed = col2.button("Parar")

    if start_pressed and st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.webcam_image_saved = False

    if stop_pressed and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    while st.session_state.cap is not None:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Não foi possível ler frame.")
            st.session_state.cap.release()
            st.session_state.cap = None
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        results = detector.detect_epis(pil_image, confidence)
        processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
            pil_image, results, required_epis, confidence, EPI_CLASSES
        )

        with frame_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, use_container_width=True)
            with col2:
                st.image(processed_image, use_container_width=True)

        with stats_placeholder.container():
            col3, col4, col5 = st.columns(3)
            col3.metric("EPIs Detectados", len(detected_epis))
            col4.metric("EPIs Faltantes", len(missing_epis))
            col5.metric("Pessoas sem EPI", len(people_without_epi))

        # Salvar apenas 1 vez
        if missing_epis and not st.session_state.webcam_image_saved:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo_img = f"imagens/sem_epi_webcam_{timestamp}.jpg"
            processed_image.save(nome_arquivo_img)

            epi_id_map = {
                "boots": 4, "helmet": 1, "safety-suit": 5,
                "gloves": 3, "safety-vest": 6, "glasses": 2
            }
            for epi in missing_epis:
                tipos_epi_id = epi_id_map.get(epi, 1)
                inserir_ocorrencia_arquivo(0, 1, 1, tipos_epi_id, nome_arquivo_img, "webcam")
            st.session_state.webcam_image_saved = True

        time.sleep(0.03)

# ------------------ App Principal ------------------ #
def main():
    st.set_page_config(page_title="EPI Detection", page_icon="🛡️", layout="wide")

    st.markdown("""
    <h1 style='text-align:center;color:white;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:15px;padding:1rem'>
    🛡️ SISTEMA DE DETECÇÃO DE EPI</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ CONFIGURAÇÕES")
        required_epis = st.multiselect("EPIs Obrigatórios",
                                       ["helmet","gloves","safety-vest","safety-suit","glasses","boots"],
                                       default=["helmet","gloves"])
        confidence = st.slider("Confiança mínima", 0.1, 0.9, 0.3, 0.05)
        image_source = st.radio("Fonte de Imagem", ["Imagem","Webcam"], index=0)

        st.subheader("📧 Teste de E-mail")
        if st.button("Enviar E-mail de Teste"):
            img = Image.new("RGB",(200,50),"white")
            draw = ImageDraw.Draw(img)
            draw.text((10,10),"Teste EPI",fill="black")
            send_email_alert(img,"Teste E-mail","Mensagem de teste")

    detector = load_model()
    if detector.model is None:
        st.error("❌ Não foi possível carregar o modelo.")
        return

    if image_source=="Imagem":
        uploaded_file = st.file_uploader("📤 Faça upload de uma imagem", type=["jpg","jpeg","png","bmp"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            if st.button("🎯 PROCESSAR IMAGEM"):
                results = detector.detect_epis(pil_image, confidence)
                processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
                    pil_image, results, required_epis, confidence, EPI_CLASSES
                )
                col1, col2 = st.columns(2)
                with col1: st.image(pil_image, use_container_width=True)
                with col2: st.image(processed_image, use_container_width=True)

                col3, col4, col5 = st.columns(3)
                col3.metric("EPIs Detectados", len(detected_epis))
                col4.metric("EPIs Faltantes", len(missing_epis))
                col5.metric("Pessoas sem EPI", len(people_without_epi))

                if missing_epis:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_arquivo_img = f"imagens/sem_epi_{timestamp}.jpg"
                    processed_image.save(nome_arquivo_img)
                    epi_id_map = {"boots":4,"helmet":1,"safety-suit":5,"gloves":3,"safety-vest":6,"glasses":2}
                    for epi in missing_epis:
                        tipos_epi_id = epi_id_map.get(epi,1)
                        inserir_ocorrencia_arquivo(0,1,1,tipos_epi_id,nome_arquivo_img,"imagem")

    elif image_source=="Webcam":
        process_webcam(detector, required_epis, confidence)

if __name__=="__main__":
    main()
