import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
from PIL import Image
from collections import defaultdict
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# === CONFIGURAÇÕES DO ROBOFLOW ===
ROBOFLOW_API_KEY = "PIBBw04VbkhOIoOLFq42"
ROBOFLOW_MODEL = "tcc-visao-computacional-epi-v2-isyfu/2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"

# === FUNÇÃO DE DETECÇÃO ===
def detectar_epi_em_frame(frame, contador=None):
    if contador is None:
        contador = defaultdict(int)
    
    # Flag para verificar se algum EPI foi detectado
    epi_detectado = False

    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        ROBOFLOW_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "frame.jpg"},
    )

    if response.status_code != 200:
        # Desenha borda vermelha quando não há conexão com a API
        frame = cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 10)
        return frame, contador

    preds = response.json().get("Predictions", [])  # Note: "Predictions" com P maiúsculo
    if not preds:
        preds = response.json().get("predictions", [])  # Fallback para minúsculo

    # Verifica se há detecções
    if len(preds) > 0:
        epi_detectado = True

    for pred in preds:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        conf = pred["confidence"]
        contador[class_name] += 1

        pt1 = (x - w // 2, y - h // 2)
        pt2 = (x + w // 2, y + h // 2)
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({conf:.2f})", (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Adiciona borda colorida baseada na detecção (APÓS processar todas as predições)
    border_color = (0, 255, 0) if epi_detectado else (0, 0, 255)  # Verde se detectado, Vermelho se não
    border_thickness = 10
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), border_color, border_thickness)

    return frame, contador


# === PROCESSA IMAGEM ===
def processar_imagem(pil_image):
    contador = defaultdict(int)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    result, contador = detectar_epi_em_frame(image, contador)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, contador


# === PROCESSA VÍDEO ===
def processar_video_em_tempo_real(uploaded_file):
    contador = defaultdict(int)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, contador = detectar_epi_em_frame(frame, contador)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, caption="Detectando EPIs...", use_container_width=True)

    cap.release()
    return contador


# === CLASSE PARA STREAMING WEBCAM ===
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.contador = defaultdict(int)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, self.contador = detectar_epi_em_frame(img, self.contador)
        return img


# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="Detecção de EPIs", layout="wide")

st.title("🦺 Sistema de Detecção de EPIs")
st.sidebar.title("🔧 Menu")
aba = st.sidebar.radio("Escolha uma opção", ["Imagem", "Vídeo", "Webcam ao vivo"])

# Adicionar instruções de cores
st.sidebar.info("""
**Legenda de Cores:**
- 🟢 **Borda Verde**: EPI detectado
- 🔴 **Borda Vermelha**: Nenhum EPI detectado
""")

if aba == "Imagem":
    st.header("📸 Detecção de EPI em Imagem")
    uploaded_image = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_img, caption="Imagem Original", use_container_width=True)

        with st.spinner("🔍 Processando imagem..."):
            resultado, contador = processar_imagem(pil_img)
            
        with col2:
            st.image(resultado, caption="Resultado com Detecção", use_container_width=True)

        if contador:
            st.success("✅ EPIs detectados!")
            st.subheader("📄 Relatório de EPIs Detectados")
            for epi, qtd in contador.items():
                st.markdown(f"- **{epi}**: {qtd}")
        else:
            st.error("❌ Nenhum EPI detectado na imagem!")

elif aba == "Vídeo":
    st.header("🎥 Detecção de EPI em Vídeo")
    uploaded_video = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)
        
        if st.button("Iniciar Processamento do Vídeo"):
            st.write("🔍 Processando vídeo...")
            
            with st.spinner("Processando em tempo real..."):
                contador = processar_video_em_tempo_real(uploaded_video)

            if contador:
                st.success("✅ EPIs detectados!")
                st.subheader("📄 Relatório de EPIs Detectados")
                for epi, qtd in contador.items():
                    st.markdown(f"- **{epi}**: {qtd}")
            else:
                st.error("❌ Nenhum EPI detectado no vídeo!")

elif aba == "Webcam ao vivo":
    st.header("📷 Detecção de EPI pela Webcam")
    st.info("🔴 Borda vermelha = Nenhum EPI detectado | 🟢 Borda verde = EPI detectado")
    
    # Adicionar um placeholder para status
    status_placeholder = st.empty()
    
    webrtc_ctx = webrtc_streamer(
        key="epi-detection", 
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if webrtc_ctx.video_transformer:
        # Atualizar status baseado no contador
        if webrtc_ctx.video_transformer.contador:
            status_placeholder.success("✅ EPI detectado!")
        else:
            status_placeholder.error("❌ Nenhum EPI detectado")