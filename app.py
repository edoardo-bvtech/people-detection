import streamlit as st
import cv2
import tempfile
import numpy as np
from inference_sdk import InferenceHTTPClient

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Video Analysis - Roboflow", layout="wide")
st.title("📹 Video Analysis con Roboflow Workflows")

# Recupera la chiave dai Secrets
if "ROBOFLOW_API_KEY" not in st.secrets:
    st.error("⚠️ Configura 'ROBOFLOW_API_KEY' nei Secrets di Streamlit Cloud!")
    st.stop()

API_KEY = st.secrets["ROBOFLOW_API_KEY"]

# Inizializza il client per i Workflows
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

WORKSPACE_ID = "trikxonns-workspace"
WORKFLOW_ID = "custom-workflow"

# --- INTERFACCIA ---
uploaded_video = st.file_uploader("Carica un video (mp4)", type=["mp4"])

if uploaded_video:
    # Salvataggio temporaneo
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    st.info("🚀 Analisi in corso... I risultati appariranno qui sotto.")
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Eseguiamo il Workflow inviando il frame come lista [frame]
            # Assicurati che l'input nel Workflow si chiami 'image'
            results = client.run_workflow(
                workspace_name=WORKSPACE_ID,
                workflow_id=WORKFLOW_ID,
                images={"image": [frame]}
            )
            
            # Gestione del risultato (evitiamo errori di tipo lista/dizionario)
            if isinstance(results, list) and len(results) > 0:
                output_data = results[0]
                
                # 'annotated_video' deve corrispondere all'output nel tuo Workflow
                annotated_frame = output_data.get("annotated_video")
                
                if annotated_frame is not None:
                    # Converti BGR (OpenCV) -> RGB (Streamlit)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                else:
                    st.error(f"❌ Output 'annotated_video' non trovato. Disponibili: {list(output_data.keys())}")
                    break
            else:
                st.warning("⚠️ Nessun dato ricevuto dal Workflow.")
                
        except Exception as e:
            st.error(f"❌ Errore durante l'esecuzione: {e}")
            break

    cap.release()
    st.success("✅ Analisi completata!")
