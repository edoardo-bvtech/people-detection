import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Configurazione Pagina
st.set_page_config(page_title="People Detection", layout="wide")
st.title("👥 People Detection - Roboflow")

# 🔑 Configurazione API (Usa st.secrets per sicurezza su Streamlit Cloud)
# Se non hai ancora impostato i Secrets, puoi scrivere temporaneamente la stringa qui
try:
    API_KEY = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("Errore: Chiave 'ROBOFLOW_API_KEY' non trovata nei Secrets di Streamlit.")
    st.stop() # Ferma l'app se manca la chiave
MODEL_ID = "people-detection-o4rdr/7"

# Inizializza il client Roboflow
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

def draw_boxes(image_rgb, predictions):
    """Disegna le box usando OpenCV gestendo correttamente i colori RGB/BGR"""
    # Streamlit usa RGB, OpenCV usa BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    for pred in predictions:
        # Roboflow restituisce il centro (x, y) e dimensioni (w, h)
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        
        # Calcolo coordinate rettangolo (top-left e bottom-right)
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        label = f"{pred['class']} {pred['confidence']:.2f}"
        
        # Disegno Box e Testo
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Riconverti in RGB per la visualizzazione in Streamlit
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Interfaccia di caricamento
uploaded_file = st.file_uploader("Carica un'immagine per rilevare persone", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Carica e converti l'immagine
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Immagine Originale")
        st.image(image, use_container_width=True)

    with st.spinner("🔍 Analisi Roboflow in corso..."):
        # Chiamata API tramite SDK
        try:
            result = client.infer(img_np, model_id=MODEL_ID)
            predictions = result.get("predictions", [])
            
            with col2:
                st.subheader("📊 Risultato Rilevamento")
                if predictions:
                    img_final = draw_boxes(img_np.copy(), predictions)
                    st.image(img_final, use_container_width=True)
                    st.success(f"Trovate {len(predictions)} persone!")
                else:
                    st.warning("Nessuna persona rilevata nell'immagine.")
                    st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
