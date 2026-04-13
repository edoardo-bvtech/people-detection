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
        try:
            results = client.run_workflow(
                workspace_name="trikxonns-workspace",
                workflow_id="custom-workflow",
                images={"image": frame}
            )
            
            # Debug: se vuoi vedere cosa restituisce l'API (apparirà in piccolo nell'app)
            # st.write(results) 

            # Verifica che results sia una lista e contenga dati
            if isinstance(results, list) and len(results) > 0:
                # Prendi l'output 'annotated_video'
                output_frame = results[0].get("annotated_video")
                
                if output_frame is not None:
                    # Converti BGR -> RGB per Streamlit
                    frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                else:
                    st.error("L'output 'annotated_video' non è stato trovato nel Workflow.")
                    st.stop()
            else:
                st.error("Il Workflow non ha restituito risultati validi.")
                st.stop()
                
        except Exception as e:
            st.error(f"Errore durante l'esecuzione del Workflow: {e}")
            st.stop()
    
    cap.release()
