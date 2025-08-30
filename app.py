import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
from PIL import Image
from collections import defaultdict
import threading
import os

# === CONFIGURAÇÕES DO ROBOFLOW ===
ROBOFLOW_API_KEY = "PIBBw04VbkhOIoOLFq42"
ROBOFLOW_MODEL = "tcc-visao-computacional-epi-v2-isyfu/2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"

# === DETECÇÃO ===
def detectar_epi_em_frame(frame, contador=None):
    if contador is None:
        contador = defaultdict(int)

    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        ROBOFLOW_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "frame.jpg"},
    )

    if response.status_code != 200:
        print("Erro API:", response.text)
        return frame, contador

    preds = response.json().get("predictions", [])

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

    frame_display = st.empty()  # Elemento que será atualizado com cada frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, contador = detectar_epi_em_frame(frame, contador)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, caption="Detectando EPIs...", use_container_width=True)

    cap.release()
    return contador


# === INICIAR WEBCAM ===
def iniciar_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erro ao acessar a webcam.")
        return

    st.info("Pressione Q para encerrar a janela da webcam.")
    contador = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, contador = detectar_epi_em_frame(frame, contador)
        cv2.imshow("🦺 Detecção de EPIs (pressione Q para sair)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nRelatório da Webcam:")
    for epi, qtd in contador.items():
        print(f"{epi}: {qtd}")

# === STREAMLIT ===
st.set_page_config(page_title="Detecção de EPIs", layout="wide")


st.title("🦺 Sistema de Detecção de EPIs")

st.sidebar.title("🔧 Menu")
aba = st.sidebar.radio("Escolha uma opção", ["Imagem", "Vídeo", "Webcam ao vivo"])

if aba == "Imagem":
    st.header("📸 Detecção de EPI em Imagem")
    uploaded_image = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image)
        st.image(pil_img, caption="Imagem Original", width=400)

        with st.spinner("🔍 Processando imagem..."):
            resultado, contador = processar_imagem(pil_img)
            st.image(resultado, caption="Resultado com EPIs", width=400)

        if contador:
            st.subheader("📄 Relatório de EPIs Detectados")
            for epi, qtd in contador.items():
                st.markdown(f"- **{epi}**: {qtd}")

elif aba == "Vídeo":
    st.header("🎥 Detecção de EPI em Vídeo")
    uploaded_video = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)  # Mostra o vídeo original
        st.write("🔍 Processando vídeo...")

        with st.spinner("Processando em tempo real..."):
            contador = processar_video_em_tempo_real(uploaded_video)

        if contador:
            st.subheader("📄 Relatório de EPIs Detectados")
            for epi, qtd in contador.items():
                st.markdown(f"- **{epi}**: {qtd}")

elif aba == "Webcam ao vivo":
    st.header("📷 Detecção de EPI pela Webcam")
    if st.button("Iniciar Webcam"):
        threading.Thread(target=iniciar_webcam).start()
        
