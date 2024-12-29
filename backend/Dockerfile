# Basis-Image
FROM python:3.10-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten hinzufügen
COPY requirements.txt requirements.txt

# Python-Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# Applikationscode hinzufügen
COPY . .

# Port 5000 freigeben
EXPOSE 5000

# Startbefehl
CMD ["python", "backend.py"]
