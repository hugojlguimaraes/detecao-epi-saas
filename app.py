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

# === CONFIGURAÇÕES AJUSTÁVEIS ===
CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "🔧 Threshold de Confiança",
    min_value=0.1,
    max_value=0.9,
    value=0.7,
    step=0.05,
    help="Aumente para reduzir falsos positivos. Valores mais altos = menos detecções, mas mais precisas"
)

MIN_AREA = st.sidebar.slider(
    "🔧 Área Mínima (pixels)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Ignora detecções muito pequenas para reduzir falsos positivos"
)

# === FUNÇÃO DE DETECÇÃO ===
def detectar_epi_em_frame(frame, contador=None):
    if contador is None:
        contador = defaultdict(int)
    
    epi_detectado = False
    frame_copy = frame.copy()
    
    # Codifica e envia para a API
    _, img_encoded = cv2.imencode(".jpg", frame_copy)
    response = requests.post(
        ROBOFLOW_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "frame.jpg"},
    )

    if response.status_code != 200:
        # Borda vermelha em caso de erro na API
        frame = cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 10)
        cv2.putText(frame, "ERRO NA API", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, contador

    # Processa a resposta da API
    response_data = response.json()
    preds = []
    if "predictions" in response_data:
        preds = response_data["predictions"]
    elif "Predictions" in response_data:
        preds = response_data["Predictions"]
    
    # Filtra detecções por confiança e área mínima
    preds_filtradas = []
    for pred in preds:
        conf = pred["confidence"]
        w = int(pred["width"])
        h = int(pred["height"])
        area = w * h
        
        if conf >= CONFIDENCE_THRESHOLD and area >= MIN_AREA:
            preds_filtradas.append(pred)
    
    # Verifica se há detecções válidas
    if len(preds_filtradas) > 0:
        epi_detectado = True

    # Processa cada detecção válida
    for pred in preds_filtradas:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])
        class_name = pred["class"]
        conf = pred["confidence"]
        contador[class_name] += 1

        # Desenha bounding box
        pt1 = (x - w // 2, y - h // 2)
        pt2 = (x + w // 2, y + h // 2)
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
        
        # Desenha label com confiança
        label = f"{class_name} ({conf:.2f})"
        cv2.putText(frame, label, (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Adiciona borda colorida baseada na detecção
    border_color = (0, 255, 0) if epi_detectado else (0, 0, 255)
    border_thickness = 12
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), border_color, border_thickness)
    
    # Adiciona texto de status
    status_text = "✅ EPI DETECTADO" if epi_detectado else "❌ SEM EPI"
    text_color = (0, 255, 0) if epi_detectado else (0, 0, 255)
    cv2.putText(frame, status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
    
    # Adiciona informações de configuração
    config_text = f"Threshold: {CONFIDENCE_THRESHOLD} | Área min: {MIN_AREA}px"
    cv2.putText(frame, config_text, (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
    status_display = st.empty()
    info_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, contador = detectar_epi_em_frame(frame, contador)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, caption="Detectando EPIs...", use_container_width=True)
        
        # Atualiza status
        if contador:
            status_display.success(f"✅ EPIs detectados: {sum(contador.values())} objetos")
            info_text = "**EPIs encontrados:**\n"
            for epi, qtd in contador.items():
                info_text += f"- {epi}: {qtd}\n"
            info_display.info(info_text)
        else:
            status_display.error("❌ Nenhum EPI detectado")
            info_display.info("Ajuste o threshold ou área mínima se estiver havendo falsos positivos")

    cap.release()
    return contador

# === CLASSE PARA STREAMING WEBCAM ===
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.contador = defaultdict(int)
        self.epi_detectado = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, self.contador = detectar_epi_em_frame(img, self.contador)
        self.epi_detectado = len(self.contador) > 0
        return img

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="Detecção de EPIs", layout="wide")

st.title("🦺 Sistema de Detecção de EPIs com Filtros Avançados")
st.sidebar.title("🔧 Configurações")

# Informações e instruções
st.sidebar.info("""
**🎯 Como usar:**
1. Ajuste o **Threshold** para controlar a sensibilidade
2. Ajuste a **Área Mínima** para ignorar objetos pequenos
3. Teste com diferentes imagens/vídeos
4. Valores recomendados:
   - Threshold: 0.7-0.8
   - Área Mínima: 1000-2000px
""")

st.sidebar.warning("""
**⚠️ Problemas comuns:**
- **Falsos positivos**: Aumente o threshold
- **Objetos pequenos**: Aumente a área mínima
- **Falsos negativos**: Diminua o threshold
""")

aba = st.sidebar.radio("Escolha uma opção", ["Imagem", "Vídeo", "Webcam ao vivo"])

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

        # Relatório de resultados
        st.subheader("📊 Relatório de Detecção")
        if contador:
            st.success(f"✅ **EPIs detectados:** {sum(contador.values())} objetos")
            for epi, qtd in contador.items():
                st.markdown(f"- **{epi}**: {qtd} detecções")
        else:
            st.error("❌ **Nenhum EPI detectado**")
            st.info("Tente diminuir o threshold ou a área mínima nas configurações")

elif aba == "Vídeo":
    st.header("🎥 Detecção de EPI em Vídeo")
    uploaded_video = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        st.video(uploaded_video)
        
        if st.button("🎬 Iniciar Processamento do Vídeo", type="primary"):
            st.write("🔍 Processando vídeo...")
            
            with st.spinner("Processando em tempo real..."):
                contador = processar_video_em_tempo_real(uploaded_video)

            # Relatório final do vídeo
            st.subheader("📊 Relatório Final do Vídeo")
            if contador:
                st.success(f"✅ **EPIs detectados no vídeo:** {sum(contador.values())} objetos no total")
                for epi, qtd in contador.items():
                    st.markdown(f"- **{epi}**: {qtd} detecções")
            else:
                st.error("❌ **Nenhum EPI detectado no vídeo inteiro**")
                st.info("Considere ajustar as configurações para melhorar a detecção")

elif aba == "Webcam ao vivo":
    st.header("📷 Detecção de EPI pela Webcam")
    st.info("""
    **🎥 Modo Webcam Ativo:**
    - Borda 🟢 VERDE = EPI detectado
    - Borda 🔴 VERMELHA = Nenhum EPI
    - Configurações ao lado são aplicadas em tempo real
    """)
    
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    webrtc_ctx = webrtc_streamer(
        key="epi-detection", 
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
    
    if webrtc_ctx.video_transformer:
        # Atualizar status baseado no contador
        if webrtc_ctx.video_transformer.contador:
            status_placeholder.success("✅ EPI detectado em tempo real!")
            stats_text = "**EPIs detectados:**\n"
            for epi, qtd in webrtc_ctx.video_transformer.contador.items():
                stats_text += f"- {epi}: {qtd}\n"
            stats_placeholder.info(stats_text)
        else:
            status_placeholder.error("❌ Nenhum EPI detectado")
            stats_placeholder.info("Aponte a câmera para objetos com EPI")

# Rodapé com informações
st.sidebar.markdown("---")
st.sidebar.caption("""
**🔍 Dicas para melhorar a precisão:**
1. Use boa iluminação
2. Posicione os EPIs claramente visíveis
3. Teste diferentes ângulos
4. Ajuste as configurações conforme necessário
""")