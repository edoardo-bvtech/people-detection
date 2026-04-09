import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from inference_sdk import InferenceHTTPClient

# ===============================
# CONFIGURAZIONE
# ===============================
st.set_page_config(page_title="People Detection", layout="wide")
st.title("♿ People Detection - Rilevamento Persone con Mobilità Ridotta")

# 🔐 API KEY da st.secrets
try:
    API_KEY = st.secrets["roboflow"]["api_key"]
except KeyError:
    st.error("❌ API Key non trovata! Configura st.secrets.")
    st.stop()

# Inizializza client Roboflow
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ===============================
# INPUT VIDEO
# ===============================
video_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

stframe = st.empty()
info_box = st.empty()

# ===============================
# AVVIO
# ===============================
if st.button("Avvia rilevamento"):

    if video_file is None:
        st.warning("Carica prima un video!")
        st.stop()

    # Salva file temporaneo in directory temp del sistema
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        temp_path = tmp.name

    try:
        cap = cv2.VideoCapture(temp_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ridimensiona per velocità
            frame = cv2.resize(frame, (640, 360))

            # Encode immagine
            _, buffer = cv2.imencode(".jpg", frame)
            img_bytes = buffer.tobytes()

            # 🔥 CHIAMATA ROBOFLOW
            result = client.infer(
                img_bytes,
                model_id="road-users-disabilities/5"
            )

            detections_text = ""

            # Disegno bounding box
            for pred in result.get("predictions", []):
                x = int(pred["x"])
                y = int(pred["y"])
                w = int(pred["width"])
                h = int(pred["height"])
                label = pred["class"]
                conf = pred["confidence"]

                # Box
                cv2.rectangle(
                    frame,
                    (x - w // 2, y - h // 2),
                    (x + w // 2, y + h // 2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Logica base
                if label in ["wheelchair", "cane", "stroller"]:
                    detections_text += f"{label.upper()} rilevato ({conf:.2f})\n"

            # Converti BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mostra video
            stframe.image(frame, channels="RGB")

            # Mostra info
            if detections_text:
                info_box.text(detections_text)

        cap.release()

    finally:
        # Pulisci il file temporaneo
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.success("✅ Rilevamento completato!")
