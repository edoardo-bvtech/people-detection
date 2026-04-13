import streamlit as st
import cv2
import tempfile
from inference_sdk import InferenceHTTPClient

# Setup API
API_KEY = st.secrets["ROBOFLOW_API_KEY"]
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

uploaded_video = st.file_uploader("Carica Video", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # Esegui Workflow
        results = client.run_workflow(
            workspace_name="trikxonns-workspace",
            workflow_id="custom-workflow",
            images={"image": frame}
        )
        
        # Prendi l'output 'annotated_video' (o come lo hai chiamato nel Workflow)
        # result è una lista, prendiamo il primo elemento [0]
        output_frame = results[0]["annotated_video"]
        
        # Converti BGR -> RGB per Streamlit
        frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
    
    cap.release()
