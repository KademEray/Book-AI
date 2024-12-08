# Basis-Image: NVIDIA CUDA mit breiter Unterstützung
FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Arbeitsverzeichnis setzen
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    python3 \
    python3-pip \
    wget \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ollama installieren
RUN curl -fsSL https://ollama.com/install.sh | sh

# Umgebungsvariable setzen
ENV OLLAMA_HOST=0.0.0.0:11434
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Port 11434 exposen
EXPOSE 11434

# GPU- oder CPU-Auswahl basierend auf Verfügbarkeit
COPY select_runtime.sh /app/select_runtime.sh
RUN chmod +x /app/select_runtime.sh

# Container ausführen
ENTRYPOINT ["/app/select_runtime.sh"]
