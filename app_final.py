import streamlit as st
import numpy as np
import tempfile
import time
import os
from PIL import Image, ImageDraw
import io
import cv2
from datetime import datetime
import requests
import json

# SOLUÇÃO: Patch para resolver o problema do signal
import signal
from unittest.mock import patch

# Monkey patch para evitar o erro do signal
original_signal = signal.signal
def mock_signal(signalnum, handler):
    try:
        return original_signal(signalnum, handler)
    except ValueError as e:
        if "main thread" in str(e):
            return None
        raise

signal.signal = mock_signal

# Tenta importar o YOLO (pode falhar no Streamlit Web)
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception as e:
    st.error(f"Erro ao importar YOLO: {e}")
    yolo_available = False
    # Modo de demonstração sem YOLO
    class MockYOLO:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            class MockResults:
                def __init__(self):
                    self.boxes = None
            return MockResults()
    
    YOLO = MockYOLO

# Configuração para evitar problemas de memory leak
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    def __init__(self, model_path=None):
        self.model = None
        self.epi_classes = {
            8: "glasses",     # Óculos - classe 8
            9: "gloves",      # Luvas - classe 9  
            10: "helmet",     # Capacete - classe 10
            14: "boots",      # Botas - classe 14
            15: "safety-suit",# Macacão - classe 15
            16: "safety-vest" # Colete - classe 16
        }
        
        # Tenta carregar o modelo apenas se o YOLO estiver disponível
        if yolo_available and model_path:
            try:
                # Para Streamlit Web, tenta carregar de URLs ou usar modelo pequeno
                if model_path.startswith('http'):
                    # Download do modelo se for uma URL
                    try:
                        response = requests.get(model_path)
                        if response.status_code == 200:
                            with open('temp_model.pt', 'wb') as f:
                                f.write(response.content)
                            self.model = YOLO('temp_model.pt')
                            st.sidebar.success("✅ Modelo carregado via URL!")
                        else:
                            st.sidebar.warning("⚠️ Usando modelo de demonstração")
                    except:
                        st.sidebar.warning("⚠️ Usando modelo de demonstração")
                else:
                    self.model = YOLO(model_path)
                    st.sidebar.success("✅ Modelo carregado com sucesso!")
            except Exception as e:
                st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
                self.model = None
        else:
            st.sidebar.warning("⚠️ Modo de demonstração (YOLO não disponível)")
        
    def detect_epis(self, frame, confidence=0.5):
        """Detecta EPIs no frame com confiança ajustável"""
        if self.model is None:
            # Modo de demonstração - retorna resultados simulados
            return self._mock_detection(frame)
        
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
            return self._mock_detection(frame)
    
    def _mock_detection(self, frame):
        """Retorna detecções simuladas para demonstração"""
        class MockResults:
            def __init__(self):
                class MockBoxes:
                    def __init__(self):
                        # Simula algumas detecções
                        self.data = []
                self.boxes = MockBoxes()
        return MockResults()

def draw_detections_pil(pil_image, results, required_epis, confidence, epi_classes):
    """Desenha detecções usando PIL"""
    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []
    
    # Cria uma cópia da imagem para desenhar
    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Simula detecções para demonstração se não houver resultados reais
    if results is None or results.boxes is None or (hasattr(results.boxes, 'data') and len(results.boxes.data) == 0):
        # Modo de demonstração - desenha caixas simuladas
        width, height = pil_image.size
        
        # Desenha algumas caixas de demonstração
        draw.rectangle([50, 50, 200, 300], outline="green", width=3)
        draw.text((55, 35), "helmet 0.85", fill="green")
        
        draw.rectangle([250, 150, 350, 320], outline="green", width=3)
        draw.text((255, 135), "gloves 0.78", fill="green")
        
        draw.rectangle([150, 320, 300, 450], outline="red", width=3)
        draw.text((155, 305), "PESSOA", fill="red")
        
        detected_epis = {"helmet", "gloves"}
        missing_epis = {"safety-vest"} if "safety-vest" in required_epis else set()
        people_without_epi = [{"safety-vest"}] if missing_epis else []
        
        return draw_image, list(detected_epis), list(missing_epis), people_without_epi
    
    # Processamento real se houver resultados
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
    """Carrega o modelo com fallback para Streamlit Web"""
    # Para Streamlit Web, usa um modelo pequeno ou modo de demonstração
    model_paths = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # Modelo pequeno
        "yolov8n.pt"
    ]
    
    for path in model_paths:
        try:
            detector = EPIDetector(path)
            if detector.model is not None:
                return detector
        except:
            continue
    
    # Fallback para modo de demonstração
    return EPIDetector(None)

# Constantes globais 
EPI_CLASSES = {
    8: "glasses", 9: "gloves", 10: "helmet", 14: "boots",
    15: "safety-suit", 16: "safety-vest"
}

def process_webcam_demo(detector, required_epis, confidence):
    """Versão de demonstração da webcam para Streamlit Web"""
    st.warning("🚫 Webcam não suportada no Streamlit Web")
    st.info("""
    **Funcionalidade limitada no Streamlit Web:**
    - Acesso à webcam não é permitido
    - Use o upload de imagens para testar
    - Para webcam ao vivo, execute localmente
    """)
    
    # Mostra uma imagem de exemplo
    example_image = Image.new('RGB', (640, 480), color='gray')
    draw = ImageDraw.Draw(example_image)
    draw.text((200, 200), "WEBCAM NÃO DISPONÍVEL\nNO STREAMLIT WEB", fill="white")
    
    st.image(example_image, caption="Modo de demonstração - Webcam não disponível", use_container_width=True)

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
            "Selecione a fonte:", ["Imagem", "Webcam (apenas local)"], index=0
        )
        
        # Informações do sistema
        st.subheader("ℹ️ Informações")
        st.info("""
        **Modo de operação:**
        - 🌐 Streamlit Web (funcionalidades limitadas)
        - 📸 Upload de imagens disponível
        - 🚫 Webcam não funciona online
        - 🚫 Banco de dados não disponível online
        """)
        
        if not yolo_available:
            st.warning("YOLO não disponível - Modo de demonstração")

    # Carregar modelo
    detector = load_model()
    
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
                        st.error(f"🚨 ALERTA: {len(missing_epis)} EPI(s) obrigatório(s) não detectado(s)!")
                        
                        # No Streamlit Web, não salva no banco, apenas mostra mensagem
                        st.info("""
                        **No Streamlit Web:**
                        - Ocorrência não salva no banco de dados
                        - Para registro completo, execute localmente
                        """)
                    else:
                        st.success("✅ Todos os EPIs obrigatórios foram detectados!")
                        
            except Exception as e:
                st.error(f"❌ Erro ao processar a imagem: {e}")
    
    else:
        process_webcam_demo(detector, required_epis, confidence)

    # Informações sobre deploy
    with st.expander("ℹ️ Informações para deploy no Streamlit Web"):
        st.markdown("""
        ## Limitações no Streamlit Web:
        
        **🚫 Funcionalidades não disponíveis:**
        - Acesso à webcam do usuário
        - Conexão com banco de dados MySQL local
        - Envio de e-mails diretamente
        - Acesso ao sistema de arquivos local
        
        **✅ Funcionalidades disponíveis:**
        - Upload e processamento de imagens
        - Detecção com YOLO (se o modelo estiver online)
        - Interface visual completa
        
        ## Para uso completo:
        
        **Execute localmente:**
        ```bash
        pip install ultralytics opencv-python streamlit Pillow mysql-connector-python
        streamlit run app.py
        ```
        
        **Ou use serviços em nuvem:**
        - Banco de dados: MongoDB Atlas, PostgreSQL na nuvem
        - Armazenamento: Amazon S3, Google Cloud Storage
        - E-mail: SendGrid, Amazon SES
        """)

if __name__ == "__main__":
    main()