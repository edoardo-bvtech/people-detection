import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# ===============================
# CONFIGURAZIONE
# ===============================
st.set_page_config(page_title="Disability Detection", layout="wide")
st.title("♿ Rilevamento persone con mobilità ridotta")

# 🔐 API KEY (meglio usare st.secrets in produzione)
API_KEY = "INSERISCI_LA_TUA_API_KEY"

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

    # Salva file temporaneo
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

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
