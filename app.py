import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
from PIL import Image, ImageDraw, ImageFont
import io

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
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            st.sidebar.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
            
        # Classes de EPI do seu modelo treinado (baseado no sh17.yaml)
        self.epi_classes = {
            8: "glasses",     # Óculos - classe 8
            9: "gloves",      # Luvas - classe 9  
            10: "helmet",     # Capacete - classe 10
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

def draw_detections_pil(pil_image, results, required_epis, confidence):
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
            
        if cls_id in EPI_CLASSES:
            epi_name = EPI_CLASSES[cls_id]
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
            
        if cls_id in EPI_CLASSES:  # É um EPI
            epi_name = EPI_CLASSES[cls_id]
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
    model_paths = [
        "models/best.pt",
        "runs/detect/epi_correction_training/weights/best.pt", 
        "best.pt",
        "yolov8n.pt"  # Fallback
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                detector = EPIDetector(path)
                if detector.model is not None:
                    return detector
            except:
                continue
    return EPIDetector("yolov8n.pt")

# Constantes globais
EPI_CLASSES = {
    8: "glasses", 9: "gloves", 10: "helmet", 
    15: "safety-suit", 16: "safety-vest"
}

def process_video(video_path, detector, required_epis, confidence):
    """Processa vídeo com detecção de EPI usando apenas PIL"""
    try:
        # Tenta abrir o vídeo com PIL (para frames individuais)
        # Para processamento de vídeo completo, precisaríamos de uma abordagem diferente
        # Vamos processar apenas o primeiro frame como demonstração
        pil_image = Image.open(video_path)
        
        # Processar frame
        results = detector.detect_epis(pil_image, confidence)
        processed_image, detected_epis, missing_epis, people_without_epi = draw_detections_pil(
            pil_image, results, required_epis, confidence
        )
        
        # Exibir resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Imagem Original")
            st.image(pil_image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Imagem Processada")
            st.image(processed_image, use_column_width=True)
        
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
            ["helmet", "gloves", "safety-vest", "safety-suit", "glasses"],
            default=["helmet", "gloves"]
        )
        
        # Configurações de confiança
        st.subheader("🔧 Configurações de Detecção")
        confidence = st.slider("Confiança mínima:", 0.1, 0.9, 0.5, 0.05)
        
        # Seleção de fonte
        st.subheader("📷 Fonte de Imagem")
        image_source = st.radio(
            "Selecione a fonte:",
            ["Upload de imagem", "Exemplo"],
            index=0
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
        """)
    
    # Carregar modelo
    detector = load_model()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🖼️ PROCESSAMENTO")
        
        if image_source == "Upload de imagem":
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
                            pil_image, results, required_epis, confidence
                        )
                        
                        # Exibir resultados
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📸 Imagem Original")
                            st.image(pil_image, use_column_width=True)
                        
                        with col2:
                            st.subheader("🎯 Imagem Processada")
                            st.image(processed_image, use_column_width=True)
                        
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
        
        elif image_source == "Exemplo":
            st.info("📝 Modo de exemplo ativado. Use upload de imagem para processar suas próprias imagens.")
    
    with col2:
        st.header("📊 ESTATÍSTICAS")
        
        # Cartões de métricas
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Performance</h3>
                <p>Modelo: YOLOv8</p>
                <p>Resolução: 640px</p>
                <p>Confiança: {}</p>
            </div>
            """.format(confidence), unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Detecção</h3>
                <p>EPIs: 5 classes</p>
                <p>Pessoas: 1 classe</p>
                <p>Total: 17 classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Legenda de cores
        st.subheader("🎨 LEGENDA")
        st.markdown("""
        - <span style='color: green; font-weight: bold'>🟢 VERDE</span>: EPI detectado
        - <span style='color: red; font-weight: bold'>🔴 VERMELHO</span>: EPI faltante
        - <span style='color: blue; font-weight: bold'>🔵 AZUL</span>: Pessoa com EPI
        """, unsafe_allow_html=True)
        
        # Status do sistema
        st.subheader("📈 STATUS")
        if detector.model is not None:
            st.success("✅ Sistema carregado e pronto")
        else:
            st.error("❌ Erro ao carregar modelo")

if __name__ == "__main__":
    main()
