import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
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
import threading

# Pasta para salvar imagens
if not os.path.exists("imagens"):
    os.makedirs("imagens")

# ------------------ FUNÇÕES DE BANCO DE DADOS ------------------ #
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

            # Inserir ocorrência
            sql_ocorrencia = """
                INSERT INTO ocorrencias (conforme, setores_id, cameras_id, tipos_epi_id)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql_ocorrencia, (conforme, setores_id, cameras_id, tipos_epi_id))
            ocorrencia_id = cursor.lastrowid

            # Inserir arquivo relacionado
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
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# ------------------ CONFIGURAÇÃO ------------------ #
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(
    page_title="EPI Detection - CCTV",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header { font-size:2.5rem; color:#fff; text-align:center; margin-bottom:1rem; padding:1rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:15px; }
    .metric-card { background-color:#f8f9fa; padding:1rem; border-radius:15px; text-align:center; margin:0.5rem; border-left:4px solid #1E88E5; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
    .alert-box { background:linear-gradient(135deg,#ff6b6b 0%,#ee5a52 100%); color:white; padding:1.5rem; border-radius:15px; margin:1rem 0; font-weight:bold;}
    .success-box { background:linear-gradient(135deg,#4ecdc4 0%,#44a08d 100%); color:white; padding:1.5rem; border-radius:15px; margin:1rem 0; font-weight:bold;}
    .stButton>button { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border:none; padding:0.75rem 1.5rem; border-radius:10px; font-weight:bold; width:100%; margin-top:1rem; transition: all 0.3s ease;}
    .stButton>button:hover { transform:translateY(-2px); box-shadow:0 4px 8px rgba(0,0,0,0.2);}
    .sidebar .sidebar-content { background:linear-gradient(180deg,#f8f9fa 0%,#e9ecef 100%);}
</style>
""", unsafe_allow_html=True)

# ------------------ CLASSES ------------------ #
class EPIDetector:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
            
        self.epi_classes = {8:"glasses", 9:"gloves", 10:"helmet", 14:"boots", 15:"safety-suit", 16:"safety-vest"}
        
    def detect_epis(self, frame, confidence=0.5):
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
        if conf < confidence: continue
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            detected_epis.add(epi_name)
            if epi_name in missing_epis:
                missing_epis.remove(epi_name)
    
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        if conf < confidence: continue
        
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            color = "green"
            label = f"{epi_name} {conf:.2f}"
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.text((x1,y1-25), label, fill=color)
        elif cls_id == 0:
            person_missing_epis = missing_epis.copy()
            color = "red" if person_missing_epis else "blue"
            label = "PESSOA"
            if person_missing_epis:
                label = f"MISSING: {', '.join(person_missing_epis)}"
                people_without_epi.append(person_missing_epis)
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.text((x1,y1-25), label, fill=color)
    
    return draw_image, list(detected_epis), list(missing_epis), people_without_epi

@st.cache_resource
def load_model():
    model_paths = ["models/best.pt", "runs/detect/epi_correction_training/weights/best.pt", "best.pt", "yolov8n.pt"]
    for path in model_paths:
        if os.path.exists(path):
            try:
                detector = EPIDetector(path)
                if detector.model is not None:
                    return detector
            except:
                continue
    return EPIDetector("yolov8n.pt")

# ------------------ ENVIO DE EMAIL ------------------ #
def send_email_alert(image_pil, subject, body, missing_epis=None):
    def send():
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
                full_body += "Equipamentos Faltantes:\n- " + "\n- ".join(missing_epis) + "\n\n"
            full_body += body
            msg.attach(MIMEText(full_body))

            buffer = io.BytesIO()
            image_pil.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            msg.attach(MIMEImage(image_data, name="alerta_epi.jpg"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            st.session_state.email_sent = True
            st.toast("📧 Alerta de e-mail enviado com sucesso!")

        except Exception as e:
            st.error(f"❌ Erro ao enviar e-mail: {e}")

    if "email_sent" not in st.session_state:
        st.session_state.email_sent = False

    if not st.session_state.email_sent:
        threading.Thread(target=send).start()

# ------------------ PROCESSAMENTO DE WEBCAM ------------------ #
def process_webcam(detector, required_epis, confidence):
    st.header("🔴 WEBCAM AO VIVO")
    col1, col2 = st.columns(2)
    start_pressed = col1.button("Iniciar", key="start_webcam")
    stop_pressed = col2.button("Parar", key="stop_webcam")

    if "cap" not in st.session_state:
        st.session_state.cap = None
        st.session_state.webcam_image_saved = False

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
            st.error("Não foi possível ler o frame da webcam.")
            st.session_state.cap.release()
            st.session_state.cap = None
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        results = detector.detect_epis(pil_image, confidence)
        processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
            pil_image, results, required_epis, confidence, EPIDetector("").epi_classes
        )

        with frame_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Imagem Original")
                st.image(pil_image, use_container_width=True)
            with col2:
                st.subheader("🎯 Imagem Processada")
                st.image(processed_image, use_container_width=True)

        with stats_placeholder.container():
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("EPIs Detectados", len(detected_epis))
                if detected_epis: st.write("✅ " + ", ".join(detected_epis))
            with col4:
                st.metric("EPIs Faltantes", len(missing_epis))
                if missing_epis: st.write("❌ " + ", ".join(missing_epis))
            with col5:
                st.metric("Pessoas sem EPI", len(people_without_epi))

            if missing_epis and not st.session_state.webcam_image_saved:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo_img = f"imagens/sem_epi_webcam_{timestamp}.jpg"
                processed_image.save(nome_arquivo_img)

                epi_id_map = {"boots":4,"helmet":1,"safety-suit":5,"gloves":3,"safety-vest":6,"glasses":2}
                for epi in missing_epis:
                    tipos_epi_id = epi_id_map.get(epi,1)
                    inserir_ocorrencia_arquivo(
                        conforme=0,
                        setores_id=1,
                        cameras_id=1,
                        tipos_epi_id=tipos_epi_id,
                        nome_arquivo=nome_arquivo_img,
                        tipo_arquivo='webcam'
                    )
                send_email_alert(processed_image, "Alerta EPI Webcam", "Alerta gerado automaticamente.", missing_epis)
                st.session_state.webcam_image_saved = True

        time.sleep(0.03)

# ------------------ MAIN ------------------ #
def main():
    st.markdown('<h1 class="main-header">🛡️ SISTEMA DE DETECÇÃO DE EPI</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ CONFIGURAÇÕES")
        required_epis = st.multiselect(
            "Selecione os EPIs obrigatórios:",
            ["helmet", "gloves", "safety-vest", "safety-suit", "glasses", "boots"],
            default=["helmet","gloves"]
        )
        confidence = st.slider("Confiança mínima:",0.1,0.9,0.3,0.05)
        image_source = st.radio("Selecione a fonte:", ["Imagem","Webcam"], index=0)

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
            image_to_send = Image.new("RGB",(200,50),color="white")
            draw = ImageDraw.Draw(image_to_send)
            draw.text((10,10),"Imagem de teste do sistema EPI", fill="black")
            send_email_alert(image_to_send, "E-mail de Teste do Sistema de Detecção de EPI", "Este é um e-mail de teste do sistema de detecção de EPI.")

    detector = load_model()
    if detector.model is None:
        st.error("❌ Não foi possível carregar o modelo.")
        return

    st.header("🖼️ PROCESSAMENTO")
    if image_source=="Imagem":
        uploaded_file = st.file_uploader("📤 Faça upload de uma imagem", type=["jpg","jpeg","png","bmp"])
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            if st.button("🎯 PROCESSAR IMAGEM", type="primary", use_container_width=True):
                results = detector.detect_epis(pil_image, confidence)
                processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
                    pil_image, results, required_epis, confidence, detector.epi_classes
                )
                col1, col2 = st.columns(2)
                with col1: st.subheader("📸 Imagem Original"); st.image(pil_image, use_container_width=True)
                with col2: st.subheader("🎯 Imagem Processada"); st.image(processed_image, use_container_width=True)

                col3, col4, col5 = st.columns(3)
                with col3: st.metric("EPIs Detectados", len(detected_epis)); 
                if detected_epis: st.write("✅ "+", ".join(detected_epis))
                with col4: st.metric("EPIs Faltantes", len(missing_epis))
                if missing_epis: st.write("❌ "+", ".join(missing_epis))
                with col5: st.metric("Pessoas sem EPI", len(people_without_epi))

                if missing_epis:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_arquivo_img = f"imagens/sem_epi_{timestamp}.jpg"
                    processed_image.save(nome_arquivo_img)

                    epi_id_map = {"boots":4,"helmet":1,"safety-suit":5,"gloves":3,"safety-vest":6,"glasses":2}
                    for epi in missing_epis:
                        tipos_epi_id = epi_id_map.get(epi,1)
                        inserir_ocorrencia_arquivo(0,1,1,tipos_epi_id,nome_arquivo_img,'imagem')

                    send_email_alert(processed_image,"Alerta EPI Imagem","Alerta gerado automaticamente.",missing_epis)
                else:
                    st.success("✅ Todos os EPIs obrigatórios foram detectados!")
    else:
        process_webcam(detector, required_epis, confidence)

if __name__=="__main__":
    main()
