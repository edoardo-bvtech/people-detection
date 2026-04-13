
import streamlit as st
import cv2
import tempfile
import supervision as sv
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title="Video Workflow Analysis", layout="wide")
st.title("📹 Video Analysis con Roboflow Workflows")

# Configurazione API dai Secrets
API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKFLOW_ID = "custom-workflow" # Inserisci il nome del tuo workflow

# Inizializza il client per i Workflows
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

uploaded_video = st.file_uploader("Carica un video (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Salvataggio temporaneo del video caricato
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    st.info("Elaborazione video in corso... Attendi il completamento.")
    
    # Setup per la visualizzazione
    frame_placeholder = st.empty()
    
    # Apertura del video con OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Esegui il Workflow sul frame attuale
        # Nota: 'image' deve corrispondere al nome dell'input nel tuo Workflow
        result = client.run_workflow(
            workspace_name="trikxonns-workspace",
            workflow_id=WORKFLOW_ID,
            images={"image": frame}
        )
        
        # 2. Recupera l'immagine annotata dal Workflow
        # 'annotated_video' deve corrispondere al nome dell'output nel tuo Workflow
        annotated_frame = result[0]["annotated_video"]
        
        # 3. Mostra il frame elaborato in Streamlit
        # Convertiamo BGR a RGB per Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.success("Elaborazione completata!")
