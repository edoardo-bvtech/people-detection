# ♿ People Detection - Rilevamento Persone con Mobilità Ridotta

Applicazione Streamlit che utilizza il modello Roboflow per rilevare persone con mobilità ridotta nei video.

## 🚀 Funzionalità

- ✅ Upload video (mp4, avi, mov)
- ✅ Rilevamento real-time di persone con disabilità motorie
- ✅ Bounding box e confidence score
- ✅ API Roboflow integrata

## 📋 Prerequisiti

- Python 3.11+
- Account Roboflow con API key

## 🛠️ Installazione Locale

1. **Clona il repository**
   ```bash
   git clone https://github.com/edoardo-bvtech/people-detection.git
   cd people-detection
   ```

2. **Crea un ambiente virtuale**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

3. **Installa le dipendenze**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura l'API Key**
   - Copia `.streamlit/secrets.toml.example` in `.streamlit/secrets.toml`
   - Sostituisci `YOUR_ROBOFLOW_API_KEY_HERE` con la tua API key Roboflow

5. **Avvia l'app**
   ```bash
   streamlit run app.py
   ```

## 🌐 Deploy su Streamlit Cloud

1. **Pushare il codice su GitHub**
   ```bash
   git add .
   git commit -m "Deploy ready"
   git push
   ```

2. **Vai su [Streamlit Cloud](https://streamlit.io/cloud)**
   - Clicca "New app"
   - Seleziona il repository e il branch
   - Scegli `app.py` come file principale
   - Clicca "Deploy"

3. **Configura i secrets su Streamlit Cloud**
   - Vai alle impostazioni dell'app
   - Sezione "Secrets"
   - Aggiungi:
     ```toml
     [roboflow]
     api_key = "YOUR_ROBOFLOW_API_KEY"
     ```

## 📁 Struttura del Progetto

```
people-detection/
├── app.py                          # App principale Streamlit
├── requirements.txt                # Dipendenze Python
├── runtime.txt                     # Versione Python
├── README.md                       # Questo file
├── .gitignore                      # File da escludere da git
└── .streamlit/
    ├── config.toml                 # Configurazione Streamlit
    └── secrets.toml.example        # Template per secrets
```

## ⚙️ Configurazione

### Environment Variables (locale)

Crea `.streamlit/secrets.toml`:
```toml
[roboflow]
api_key = "YOUR_API_KEY"
```

### Streamlit Cloud

Configura i secrets direttamente nel dashboard di Streamlit Cloud.

## 📦 Dipendenze

- `streamlit==1.27.0` - Framework web
- `opencv-python-headless==4.8.1.78` - Elaborazione video
- `numpy==1.26.4` - Calcoli numerici
- `pillow` - Manipolazione immagini
- `inference-sdk==0.45.0` - API Roboflow

## 🔐 Sicurezza

- ⚠️ **NON committare mai** `.streamlit/secrets.toml`
- L'API key è gestita via `st.secrets`
- `.gitignore` esclude automaticamente i file sensibili

## 🐛 Troubleshooting

### "API Key non trovata!"
- Verifica che `.streamlit/secrets.toml` esista e contenga l'API key corretta
- Su Streamlit Cloud, controlla che i secrets siano configurati

### "ModuleNotFoundError"
- Reinstalla le dipendenze: `pip install -r requirements.txt --upgrade`

### Video non caricato
- Verifica il formato (mp4, avi, mov)
- Prova con un file più piccolo

## 📝 Licenza

MIT

## 👨‍💻 Autore

[Edoardo BV Tech](https://github.com/edoardo-bvtech)</content>
<parameter name="filePath">C:\Users\rifez\Desktop\pearson_detect\people-detection\README.md