import streamlit as st
import cv2
import tempfile
import numpy as np
from inference_sdk import InferenceHTTPClient

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Video Analysis - Roboflow Workflows", layout="wide")
st.title("📹 Video Analysis con Roboflow Workflows")

# Recupera la chiave dai Secrets di Streamlit
if "ROBOFLOW_API_KEY" not in st.secrets:
    st.error("⚠️ Configura 'ROBOFLOW_API_KEY' nei Secrets di Streamlit Cloud!")
    st.stop()

API_KEY = st.secrets["ROBOFLOW_API_KEY"]

# Inizializza il client per i Workflows
# L'endpoint corretto per i Workflows è detect.roboflow.com
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Parametri del tuo Workflow su Roboflow
WORKSPACE_ID = "trikxonns-workspace"
WORKFLOW_ID = "custom-workflow"

# --- INTERFACCIA ---
uploaded_video = st.file_uploader("Carica un video per l'analisi (mp4)", type=["mp4"])

if uploaded_video:
    # 1. Salvataggio temporaneo del file caricato per leggerlo con OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    st.info("🚀 Analisi in corso frame per frame. Attendi il caricamento dei risultati...")
    
    # Placeholder per mostrare il video che scorre
    frame_placeholder = st.empty()
    
    # 2. Apertura del video con OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Ottieni informazioni sul video (opzionale, utile per il debug)
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.sidebar.write(f"🎞️ Video FPS originale: {fps:.2f}")

    # 3. Ciclo di elaborazione frame per frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # --- CHIAMATA WORKFLOW ---
            # Passiamo [frame] come lista per evitare errori di decodifica nell'SDK
            # 'image' deve corrispondere al nome del blocco Input nel Workflow
            results = client.run_workflow(
                workspace_name=WORKSPACE_ID,
                workflow_id=WORKFLOW_ID,
                images={"image": [frame]}
            )
            
            # 4. Gestione Risultati
            if results and isinstance(results, list) and len(results) > 0:
                output_data = results[0]
                
                # 'annotated_video' deve corrispondere al nome del Workflow Output
                annotated_frame = output_data.get("annotated_video")
                
                if annotated_frame is not None:
                    # Converti BGR (OpenCV) -> RGB (Streamlit)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Mostra il frame elaborato
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                else:
                    st.error("❌ Output 'annotated_video' non trovato. Controlla i nomi nel Workflow!")
                    break
            else:
                st.warning("⚠️ Nessun dato ricevuto dal Workflow per questo frame.")
                
        except Exception as e:
            st.error(f"❌ Errore durante l'esecuzione del Workflow: {e}")
            break

    # 5. Pulizia
    cap.release()
    st.success("✅ Analisi video completata!")
