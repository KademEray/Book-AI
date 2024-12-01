# Basis-Image
FROM ubuntu:latest

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten installieren
RUN apt-get update && \
    apt-get install -y curl python3 python3-pip && \
    apt-get clean

# Ollama installieren
RUN curl -fsSL https://ollama.com/install.sh | sh

# Umgebungsvariable setzen, damit der Server auf alle IP-Adressen hört
ENV OLLAMA_HOST=http://0.0.0.0:11434

# Port 11434 exposen
EXPOSE 11434

# Container ausführen und Ollama starten
CMD ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b && \
    wait && \
    pkill -f "ollama serve" && \
    sleep 2 && \
    ollama serve