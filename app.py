import streamlit as st
import numpy as np
import cv2
from PIL import Image
# 1. Installa con: pip install inference-sdk
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title="People Detection", layout="wide")
st.title("👥 People Detection - Roboflow")

# Configurazione API
API_KEY = "7IvJ8E5kwCJd2MAsZFE5"
MODEL_ID = "people-detection-o4rdr/7"

# Inizializza il client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

def draw_boxes(image_rgb, predictions):
    # Converti da RGB (Streamlit) a BGR (OpenCV)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    for pred in predictions:
        # Roboflow restituisce il centro (x, y) e dimensioni (w, h)
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Riconverti in RGB per Streamlit
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    with st.spinner("🔍 Analisi in corso..."):
        # L'SDK accetta direttamente l'array numpy o il file
        result = client.infer(img_np, model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        st.write(f"👀 Persone rilevate: {len(predictions)}")
        
        # Disegna e mostra
        img_final = draw_boxes(img_np.copy(), predictions)
        st.image(img_final, caption="Risultato Analisi", use_container_width=True)
