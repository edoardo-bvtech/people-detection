# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import base64

# ===============================
# CONFIGURAZIONE INIZIALE
# ===============================
st.set_page_config(page_title="Disability Detection Stream", layout="wide")
st.title("♿ Rilevamento persone con mobilità ridotta")

# API Key e workspace Roboflow
API_KEY = "TUA_API_KEY"
WORKFLOW_NAME = "custom-workflow"
WORKSPACE = "trikxonns-workspace"

# ===============================
# SELEZIONE INPUT VIDEO
# ===============================
input_option = st.radio("Scegli il video source", ["Webcam", "Video File"])

video_file = None
if input_option == "Video File":
    video_file = st.file_uploader("Carica il video", type=["mp4", "avi", "mov"])

# ===============================
# INIZIALIZZA CLIENT ROBOTFLOW
# ===============================
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

VIDEO_OUTPUT = "annotated_video"
DATA_OUTPUTS = ["analytics_data"]

config = StreamConfig(
    stream_output=[],
    data_output=["annotated_video","analytics_data"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

# ===============================
# STREAM VIDEO
# ===============================
stframe = st.empty()
analytics_frame = st.empty()

if st.button("Avvia rilevamento"):

    if input_option == "Webcam":
        source = VideoFileSource("0", realtime_processing=True)
    elif input_option == "Video File":
        if video_file is None:
            st.warning("Carica prima un video!")
            st.stop()
        # Salva temporaneamente
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        source = VideoFileSource("temp_video.mp4", realtime_processing=False)

    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW_NAME,
        workspace=WORKSPACE,
        image_input="image",
        config=config
    )

    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        # ===============================
        # VIDEO ANNOTATO
        # ===============================
        if VIDEO_OUTPUT in data:
            frame_bytes = base64.b64decode(data[VIDEO_OUTPUT]["value"])
            img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Mostra in Streamlit
            stframe.image(img, channels="RGB", use_column_width=True)

        # ===============================
        # ANALYTICS DATA
        # ===============================
        if DATA_OUTPUTS[0] in data:
            detections = data[DATA_OUTPUTS[0]]["value"]
            output_text = ""
            for det in detections:
                label = det.get("class", "")
                confidence = det.get("confidence", 0)
                if label in ["wheelchair", "cane", "stroller"]:
                    output_text += f"{label.upper()} rilevato - conf: {confidence:.2f}\n"
            analytics_frame.text(output_text)

    session.run()
