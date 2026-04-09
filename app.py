import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="People Detection", layout="wide")

st.title("👥 People Detection - Roboflow (Python 3.14)")

# 🔑 Inserisci la tua API KEY
API_KEY = "INSERISCI_LA_TUA_API_KEY"

MODEL_URL = "https://detect.roboflow.com/people-detection-o4rdr/7"

uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

def draw_boxes(image, predictions):
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        label = f"{pred['class']} {pred['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("📷 Immagine originale")
    st.image(image, use_column_width=True)

    with st.spinner("🔍 Analisi in corso..."):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")

        response = requests.post(
            MODEL_URL,
            params={"api_key": API_KEY},
            files={"file": buffered.getvalue()}
        )

        if response.status_code != 200:
            st.error(f"Errore API: {response.text}")
        else:
            data = response.json()
            predictions = data.get("predictions", [])

            st.write(f"👀 Persone rilevate: {len(predictions)}")

            img_with_boxes = draw_boxes(img_np.copy(), predictions)

            st.subheader("📊 Risultato")
            st.image(img_with_boxes, use_column_width=True)
