import streamlit as st
import numpy as np
import tempfile
import time
import os
from PIL import Image, ImageDraw, ImageFont
import io
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Configurações para evitar problemas no Streamlit Cloud
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Importação segura do OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV não disponível: {e}")
    CV2_AVAILABLE = False

# Importação segura do Ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"YOLO não disponível: {e}")
    YOLO_AVAILABLE = False

if not os.path.exists("imagens"):
    os.makedirs("imagens")
    
def inserir_ocorrencia_arquivo(conforme, setores_id, cameras_id, tipos_epi_id, nome_arquivo, tipo_arquivo):
    conn = None
    cursor = None
    try:
        # Conexão com o banco de dados
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
            return True

    except Error as e:
        print(f"❌ Erro ao conectar ou inserir: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False
    finally:
        # Fechar conexões apenas se foram abertas
        try:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()
        except Exception as e:
            print(f"❌ Erro ao fechar conexões: {e}")

# Configuração da página
st.set_page_config(
    page_title="EPI Detection - CCTV",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .success-box {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)


class EPIDetector:
    def __init__(self, model_path):
        if not YOLO_AVAILABLE:
            st.sidebar.error("❌ YOLO não está disponível")
            self.model = None
            return
            
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
            
        # Classes de EPI do seu modelo treinado
        self.epi_classes = {
            8: "glasses",     # Óculos - classe 8
            9: "gloves",      # Luvas - classe 9  
            10: "helmet",     # Capacete - classe 10
            14: "boots",      # Botas - classe 14
            15: "safety-suit",# Macacão - classe 15
            16: "safety-vest" # Colete - classe 16
        }
        
    def detect_epis(self, frame, confidence=0.5):
        """Detecta EPIs no frame com confiança ajustável"""
        if self.model is None:
            return None
        try:
            # Converte numpy array para PIL Image se necessário
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
    """Desenha detecções usando PIL em vez de OpenCV"""
    if results is None or results.boxes is None:
        return pil_image, [], [], []
    
    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []
    
    # Cria uma cópia da imagem para desenhar
    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Primeiro passada: detectar todos os EPIs
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        
        if conf < confidence:
            continue
            
        if cls_id in epi_classes:
            epi_name = epi_classes[cls_id]
            detected_epis.add(epi_name)
            if epi_name in missing_epis:
                missing_epis.remove(epi_name)
    
    # Segunda passada: desenhar as bounding boxes
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        if conf < confidence:
            continue
            
        if cls_id in epi_classes:  # É um EPI
            epi_name = epi_classes[cls_id]
            color = "green"  # Verde para EPI detectado
            label = f"{epi_name} {conf:.2f}"
            
            # Desenha retângulo
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Desenha texto
            draw.text((x1, y1-25), label, fill=color)
            
        elif cls_id == 0:  # É uma pessoa
            # Verifica se está sem EPIs obrigatórios
            person_missing_epis = missing_epis.copy()
            
            color = "red" if person_missing_epis else "blue"
            label = "PESSOA"
            
            if person_missing_epis:
                label = f"MISSING: {', '.join(person_missing_epis)}"
                people_without_epi.append(person_missing_epis)
            
            # Desenha retângulo
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Desenha texto
            draw.text((x1, y1-25), label, fill=color)
    
    return draw_image, list(detected_epis), list(missing_epis), people_without_epi

@st.cache_resource
def load_model():
    """Carrega o modelo com fallback"""
    if not YOLO_AVAILABLE:
        return None
        
    model_paths = [
        "models/best.pt",
        "runs/detect/epi_correction_training/weights/best.pt", 
        "best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                detector = EPIDetector(path)
                if detector.model is not None:
                    return detector
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erro ao carregar {path}: {e}")
                continue
    
    st.sidebar.error("❌ Nenhum modelo válido encontrado.")
    return None

# Constantes globais 
EPI_CLASSES = {
    8: "glasses", 9: "gloves", 10: "helmet", 14: "boots",
    15: "safety-suit", 16: "safety-vest"
}

def process_video(video_path, detector, required_epis, confidence):
    """Processa vídeo com detecção de EPI usando apenas PIL"""
    try:
        # Tenta abrir o vídeo com PIL (para frames individuais)
        pil_image = Image.open(video_path)
        
        # Processar frame
        results = detector.detect_epis(pil_image, confidence)
        processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
            pil_image, results, required_epis, confidence, EPI_CLASSES
        )
        
        # Exibir resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Imagem Original")
            st.image(pil_image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Imagem Processada")
            st.image(processed_image, use_container_width=True)
        
        # Estatísticas
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
        
        # Alertas
        if missing_epis:
            st.error(f"🚨 ALERTA: {len(missing_epis)} EPI(s) obrigatório(s) não detectado(s)!")
        else:
            st.success("✅ Todos os EPIs obrigatórios foram detectados!")
            
    except Exception as e:
        st.error(f"❌ Erro ao processar a imagem: {e}")

def process_webcam(detector, required_epis, confidence):
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV não está disponível. Funcionalidade de webcam desativada.")
        st.info("🚫 Esta funcionalidade só está disponível localmente")
        return
        
    if detector is None or detector.model is None:
        st.error("❌ Modelo não carregado. Verifique as configurações.")
        return
        
    st.header("🔴 WEBCAM AO VIVO")
    st.write("Pressione 'Iniciar' para começar a detecção e 'Parar' para terminar.")

    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "webcam_image_saved" not in st.session_state:
        st.session_state.webcam_image_saved = False

    col1, col2 = st.columns(2)
    start_pressed = col1.button("Iniciar", key="start_webcam")
    stop_pressed = col2.button("Parar", key="stop_webcam")

    if start_pressed and st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("❌ Webcam não disponível. Verifique a conexão.")
            st.session_state.cap = None
        else:
            st.session_state.webcam_image_saved = False

    if stop_pressed and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()  

    while st.session_state.cap is not None and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Não foi possível ler o frame da webcam.")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
            st.session_state.cap = None
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        results = detector.detect_epis(pil_image, confidence)
        processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
            pil_image, results, required_epis, confidence, EPI_CLASSES
        )
        
        # Exibir resultados
        with frame_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Imagem Original")
                st.image(pil_image, use_container_width=True)
            with col2:
                st.subheader("🎯 Imagem Processada")
                st.image(processed_image, use_container_width=True)

        with stats_placeholder.container():
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

            # Alertas
            if missing_epis:
                st.error(f"🚨 ALERTA: {len(missing_epis)} EPI(s) obrigatório(s) não detectado(s)!")
            else:
                st.success("✅ Todos os EPIs obrigatórios foram detectados!")

        # Salvar imagem e ocorrência apenas uma vez
        if missing_epis and not st.session_state.webcam_image_saved:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo_img = f"imagens/sem_epi_webcam_{timestamp}.jpg"
            processed_image.save(nome_arquivo_img)
            epi_id_map = {
                "boots": 4,
                "helmet": 1,
                "safety-suit": 5,
                "gloves": 3,
                "safety-vest": 6,
                "glasses": 2
            }
            for epi in missing_epis:
                tipos_epi_id = epi_id_map.get(epi, 1)
                success = inserir_ocorrencia_arquivo(
                    conforme=0,
                    setores_id=1,
                    cameras_id=1,
                    tipos_epi_id=tipos_epi_id,
                    nome_arquivo=nome_arquivo_img,
                    tipo_arquivo='webcam'
                )
                if success:
                    st.success(f"✅ Ocorrência registrada para {epi}")
                else:
                    st.error(f"❌ Erro ao registrar ocorrência para {epi}")
            
            st.session_state.webcam_image_saved = True

        time.sleep(0.03)

def send_email_alert(image_pil, subject, body, missing_epis=None):
    try:
        # Configurações de email (ajuste conforme necessário)
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
            full_body += (
                "Equipamentos Faltantes:\n- " + "\n- ".join(missing_epis) + "\n\n"
            )

        full_body += body
        text = MIMEText(full_body)
        msg.attach(text)

        # Anexar imagem
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image = MIMEImage(image_data, name="alerta_epi.jpg")
        msg.attach(image)

        # Enviar email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.toast("📧 Alerta de e-mail enviado com sucesso!")

    except Exception as e:
        st.error(
            f"Erro ao enviar e-mail: {e}. Verifique suas configurações de email."
        )

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ SISTEMA DE DETECÇÃO DE EPI</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURAÇÕES")
        
        # Seleção de EPIs obrigatórios
        st.subheader("🎯 EPIs Obrigatórios")
        required_epis = st.multiselect(
            "Selecione os EPIs obrigatórios:",
            ["helmet", "gloves", "safety-vest", "safety-suit", "glasses", "boots"],
            default=["helmet", "gloves"]
        )
        
        # Configurações de confiança
        st.subheader("🔧 Configurações de Detecção")
        confidence = st.slider("Confiança mínima:", 0.1, 0.9, 0.3, 0.05)
        
        # Seleção de fonte
        st.subheader("📷 Fonte de Imagem")
        image_source = st.radio(
            "Selecione a fonte:", ["Imagem", "Webcam"], index=0
        )
        
        # Informações do sistema
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
            # Fallback para uma imagem de teste simples
            image_to_send = Image.new("RGB", (200, 50), color="white")
            draw = ImageDraw.Draw(image_to_send)
            draw.text((10, 10), "Imagem de teste do sistema EPI", fill="black")
            body = "Este é um e-mail de teste do sistema de detecção de EPI."
            
            subject = "E-mail de Teste do Sistema de Detecção de EPI"
            send_email_alert(image_to_send, subject, body)

    # Carregar modelo
    detector = load_model()
    if detector is None or detector.model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique as configurações.")
        st.info("💡 Certifique-se de que o arquivo do modelo está na pasta correta")
        return
    
    # Main content
    st.header("🖼️ PROCESSAMENTO")

    if image_source == "Imagem":
        uploaded_file = st.file_uploader(
            "📤 Faça upload de uma imagem", 
            type=["jpg", "jpeg", "png", "bmp"],
            help="Formatos suportados: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Processar imagem
            try:
                pil_image = Image.open(uploaded_file)
                
                if st.button("🎯 PROCESSAR IMAGEM", type="primary", use_container_width=True):
                    # Processar frame
                    results = detector.detect_epis(pil_image, confidence)
                    processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
                        pil_image, results, required_epis, confidence, EPI_CLASSES
                    )
                    
                    # Exibir resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📸 Imagem Original")
                        st.image(pil_image, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Imagem Processada")
                        st.image(processed_image, use_container_width=True)
                    
                    # Estatísticas
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
                    
                    # Alertas
                    if missing_epis:
                        # Gera nome único usando data/hora
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arquivo_img = f"imagens/sem_epi_{timestamp}.jpg"
                        processed_image.save(nome_arquivo_img)

                        epi_id_map = {
                            "boots": 4,
                            "helmet": 1,
                            "safety-suit": 5,
                            "gloves": 3,
                            "safety-vest": 6,
                            "glasses": 2
                        }
                        for epi in missing_epis:
                            tipos_epi_id = epi_id_map.get(epi, 1)
                            success = inserir_ocorrencia_arquivo(
                                conforme=0,
                                setores_id=1,
                                cameras_id=1,
                                tipos_epi_id=tipos_epi_id,
                                nome_arquivo=nome_arquivo_img,
                                tipo_arquivo='imagem'
                            )
                            if success:
                                st.success(f"✅ Ocorrência registrada para {epi}")
                            else:
                                st.error(f"❌ Erro ao registrar ocorrência para {epi}")
                        
                    else:
                        st.success("✅ Todos os EPIs obrigatórios foram detectados!")
                        
            except Exception as e:
                st.error(f"❌ Erro ao processar a imagem: {e}")
    
    elif image_source == "Webcam":
        process_webcam(detector, required_epis, confidence)

if __name__ == "__main__":
    main()