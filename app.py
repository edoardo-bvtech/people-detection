# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import base64

from inference_sdk import InferenceHTTPClient
from inference_sdk.stream import StreamConfig
from inference_sdk.stream import VideoFileSource
from inference_sdk.stream import VideoMetadata

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
    api_key=7IvJ8E5kwCJd2MAsZFE5
)

VIDEO_OUTPUT = "annotated_video"
DATA_OUTPUTS = ["analytics_data"]

config = StreamConfig(
    stream_output=[],
    data_output=["annotated_video","analytics_data"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

if st.button("Avvia rilevamento"):

    if input_option == "Video File":
        if video_file is None:
            st.warning("Carica prima un video!")
            st.stop()

        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()

            # CHIAMATA API ROBOFLOW
            result = client.infer(
                img_bytes,
                model_id="road-users-disabilities/5"
            )

            # DISEGNA BOX
            for pred in result["predictions"]:
                x = int(pred["x"])
                y = int(pred["y"])
                w = int(pred["width"])
                h = int(pred["height"])

                label = pred["class"]

                cv2.rectangle(frame,
                              (x - w//2, y - h//2),
                              (x + w//2, y + h//2),
                              (0, 255, 0), 2)

                cv2.putText(frame, label,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame)

        cap.release()


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
