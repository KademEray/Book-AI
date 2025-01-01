# Basis-Image
FROM python:3.10-slim

# Arbeitsverzeichnis im Container setzen
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY . .

# Port für Gradio öffnen
EXPOSE 3000

# Startbefehl
CMD ["python", "app.py"]
