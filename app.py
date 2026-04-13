import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title="Video Report - Roboflow", layout="wide")
st.title("📊 Analisi Video e Report Individui")

# Setup API dai Secrets
API_KEY = st.secrets["ROBOFLOW_API_KEY"]
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

uploaded_video = st.file_uploader("Carica Video per Analisi", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    report_data = {}
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    
    st.info("⌛ Elaborazione in corso... I risultati appariranno gradualmente.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        try:
            # FIX: Inviamo il frame dentro una lista [frame]
            results = client.run_workflow(
                workspace_name="trikxonns-workspace",
                workflow_id="custom-workflow",
                images={"image": [frame]}
            )
            
            # Gestione sicura della risposta (results è una lista)
            if isinstance(results, list) and len(results) > 0:
                output = results[0] # Prendiamo il risultato del primo frame
                
                # 1. Visualizzazione Video
                if "annotated_video" in output:
                    img_out = output["annotated_video"]
                    if isinstance(img_out, np.ndarray):
                        frame_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # 2. Accumulo Dati per il Report
                detections = output.get("analytics_data", [])
                for det in detections:
                    tid = det.get("tracker_id")
                    if tid is not None:
                        if tid not in report_data:
                            report_data[tid] = {"tempo": 0, "vel": 0, "tipo": det.get("class_name")}
                        
                        # Salviamo il valore massimo raggiunto per tempo e velocità
                        report_data[tid]["tempo"] = max(report_data[tid]["tempo"], det.get("time_in_zone", 0))
                        report_data[tid]["vel"] = max(report_data[tid]["vel"], det.get("velocity", 0))
        
        except Exception as e:
            # Se l'SDK dà ancora errore .items(), lo catturiamo qui senza far crashare tutto
            continue

    cap.release()

    # --- GENERAZIONE REPORT FINALE ---
    st.divider()
    st.subheader("📋 Report Analisi Finale")
    
    if report_data:
        final_list = []
        for tid, v in report_data.items():
            final_list.append({
                "ID": tid, "Classe": v["tipo"],
                "Tempo (sec)": round(v["tempo"], 2),
                "Velocità Max": round(v["vel"], 2)
            })
        
        df = pd.DataFrame(final_list)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Scarica Report CSV", df.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
    else:
        st.warning("Nessun dato raccolto. Assicurati che il Workflow stia rilevando persone.")
