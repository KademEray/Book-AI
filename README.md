# Book-AI

## Ausführung
1. CPU
```bash
cd AI_CPU
docker compose up -d
```

1. GPU
```bash
cd AI_GPU
docker compose up -d
```

2.
```bash
cd Gradio
docker-compose up
```
oder auf die gradio/app.py Datei und dann run drücken

Hinweis: 
Bei Versionen Probleme erst einmal pip install Gradio, dann gradio starten.
Danach pip install pydantic==1.10.19 und dann den Backend starten

3. 
```bash
cd backend
python backend.py
```
oder auf die Backend Datei und dann run drücken