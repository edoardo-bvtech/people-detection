import streamlit as st
import cv2
import tempfile
import pandas as pd
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title="Video Report - Roboflow", layout="wide")
st.title("📊 Analisi Video e Report Individui")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

uploaded_video = st.file_uploader("Carica Video per Analisi", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Dizionario per accumulare i dati di ogni individuo (chiave = tracker_id)
    report_data = {}

    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    
    st.info("⌛ Elaborazione in corso... Il report apparirà al termine.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        results = client.run_workflow(
            workspace_name="trikxonns-workspace",
            workflow_id="custom-workflow",
            images={"image": [frame]}
        )
        
        if results and len(results) > 0:
            output = results[0]
            
            # 1. Visualizzazione Video
            if "annotated_video" in output:
                frame_rgb = cv2.cvtColor(output["annotated_video"], cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # 2. Accumulo Dati per il Report (da analytics_data)
            # 'analytics_data' contiene la lista delle persone rilevate in QUESTO frame
            detections = output.get("analytics_data", [])
            for det in detections:
                tid = det.get("tracker_id")
                if tid is not None:
                    # Aggiorna o crea l'entrata per questo individuo
                    if tid not in report_data:
                        report_data[tid] = {"tempo_max": 0, "velocita_max": 0, "classe": det.get("class_name")}
                    
                    # Aggiorna il tempo di permanenza e la velocità massima registrata
                    report_data[tid]["tempo_max"] = max(report_data[tid]["tempo_max"], det.get("time_in_zone", 0))
                    report_data[tid]["velocita_max"] = max(report_data[tid]["velocita_max"], det.get("velocity", 0))

    cap.release()

    # --- GENERAZIONE REPORT FINALE ---
    st.divider()
    st.subheader("📋 Report Analisi Individui")
    
    if report_data:
        # Trasformiamo il dizionario in una lista per creare un DataFrame Pandas
        final_list = []
        for tid, values in report_data.items():
            final_list.append({
                "ID Individuo": tid,
                "Tipo": values["classe"],
                "Tempo in Scena (sec)": round(values["tempo_max"], 2),
                "Velocità Max Registrata": round(values["velocita_max"], 2)
            })
        
        df = pd.DataFrame(final_list)
        st.dataframe(df, use_container_width=True)
        
        # Opzione per scaricare il report in CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Report CSV", csv, "report_video.csv", "text/csv")
    else:
        st.warning("Nessun individuo rilevato nel video.")
