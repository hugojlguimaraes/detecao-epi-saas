import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
from PIL import Image

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
            results = self.model(frame, verbose=False, conf=confidence)
            return results[0] if results else None
        except Exception as e:
            st.error(f"Erro na detecção: {e}")
            return None

def draw_detections(frame, results, required_epis, confidence):
    """Desenha detecções com cores baseadas no uso de EPI"""
    if results is None or results.boxes is None:
        return frame, [], []
    
    detected_epis = set()
    missing_epis = set(required_epis) if required_epis else set()
    people_without_epi = []
    
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
            color = (0, 255, 0)  # Verde para EPI detectado
            label = f"{epi_name} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        elif cls_id == 0:  # É uma pessoa
            # Verifica se está sem EPIs obrigatórios
            person_missing_epis = missing_epis.copy()
            
            color = (0, 0, 255) if person_missing_epis else (255, 0, 0)
            label = "PESSOA"
            
            if person_missing_epis:
                label = f"MISSING: {', '.join(person_missing_epis)}"
                people_without_epi.append(person_missing_epis)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame, list(detected_epis), list(missing_epis), people_without_epi

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
        
        # Seleção de fonte de vídeo
        st.subheader("📹 Fonte de Vídeo")
        video_source = st.radio(
            "Selecione a fonte:",
            ["Upload de vídeo", "Webcam (local)"],
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
        st.header("🎥 VISUALIZAÇÃO")
        
        if video_source == "Webcam (local)":
            st.warning("""
            ⚠️ **Webcam não funciona no Streamlit Cloud**
            
            Para usar webcam, execute localmente:
            ```bash
            streamlit run app.py
            ```
            """)
            
        elif video_source == "Upload de vídeo":
            uploaded_file = st.file_uploader(
                "📤 Faça upload de um vídeo", 
                type=["mp4", "avi", "mov", "mkv"],
                help="Formatos suportados: MP4, AVI, MOV, MKV"
            )
            
            if uploaded_file is not None:
                # Salvar arquivo temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                if st.button("🎯 PROCESSAR VÍDEO", type="primary", use_container_width=True):
                    process_video(video_path, detector, required_epis, confidence)
    
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
        - <span style='color: orange; font-weight: bold'>🟠 LARANJA</span>: Análise
        """, unsafe_allow_html=True)
        
        # Status do sistema
        st.subheader("📈 STATUS")
        if detector.model is not None:
            st.success("✅ Sistema carregado e pronto")
        else:
            st.error("❌ Erro ao carregar modelo")

def process_video(video_path, detector, required_epis, confidence):
    """Processa vídeo com detecção de EPI"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo")
        return
    
    # Obter informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    st.info(f"""
    **📊 Informações do vídeo:**
    - Frames: {total_frames}
    - FPS: {fps:.1f}
    - Duração: {duration:.1f}s
    """)
    
    # Elementos da interface
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar frame
        results = detector.detect_epis(frame, confidence)
        processed_frame, detected_epis, missing_epis, people_without_epi = draw_detections(
            frame.copy(), results, required_epis, confidence
        )
        
        # Converter para RGB (Streamlit)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Exibir frame
        stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
        
        # Atualizar métricas
        with metrics_col1:
            st.metric("EPIs Detectados", len(detected_epis))
        with metrics_col2:
            st.metric("EPIs Faltantes", len(missing_epis))
        with metrics_col3:
            st.metric("Pessoas", len(people_without_epi))
        
        # Atualizar status
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        status_text.text(f"""
        ⚡ Processando: {frame_count}/{total_frames} frames
        🎯 FPS: {fps_actual:.1f}
        ⏱️ Tempo: {elapsed_time:.1f}s
        """)
        
        # Atualizar barra de progresso
        if total_frames > 0:
            progress_bar.progress(frame_count / total_frames)
        
        # Controlar velocidade de processamento
        time.sleep(0.03)  # ~30 FPS
    
    # Finalização
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Estatísticas finais
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    st.success(f"""
    ✅ **Processamento concluído!**
    
    **📊 Estatísticas:**
    - Frames processados: {frame_count}
    - Tempo total: {total_time:.1f}s
    - FPS médio: {avg_fps:.1f}
    - Velocidade: {avg_fps/fps:.1f}x tempo real
    """)

if __name__ == "__main__":
    main()