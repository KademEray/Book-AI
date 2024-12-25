# Basis-Image
FROM python:3.9-slim

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY . .

# Port für Gradio-Frontend öffnen
EXPOSE 7860

# Startbefehl
CMD ["python", "app.py"]
