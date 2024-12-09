#!/bin/bash

# Funktion zum Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Funktion zum Überprüfen der GPU-Verfügbarkeit
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            log "NVIDIA GPU detected"
            return 0
        fi
    fi
    log "No NVIDIA GPU detected or drivers not properly installed"
    return 1
}

# Funktion zum Starten von Ollama
start_ollama() {
    local gpu_mode=$1
    local cmd="ollama serve"
    
    if [ "$gpu_mode" = "cpu" ]; then
        cmd="$cmd --cpu"
        log "Starting Ollama in CPU mode"
    else
        log "Starting Ollama in GPU mode"
    fi
    
    # Starte Ollama Server
    log "Starting Ollama server..."
    $cmd &
    local ollama_pid=$!
    
    # Warte bis Ollama bereit ist
    local max_attempts=30
    local attempt=0
    while ! curl -s http://localhost:11434/api/version &>/dev/null; do
        sleep 1
        ((attempt++))
        if [ $attempt -ge $max_attempts ]; then
            log "Error: Ollama failed to start after $max_attempts seconds"
            exit 1
        fi
    done
    
    log "Ollama server is ready. Pulling model llama3.2:3b..."
    ollama pull llama3.2:3b
    
    if [ $? -eq 0 ]; then
        log "Model llama3.2:3b successfully pulled"
        log "Ollama is now ready to use"
    else
        log "Error pulling model llama3.2:3b"
        exit 1
    fi
    
    # Behalte den Server-Prozess im Vordergrund
    wait $ollama_pid
}

# Hauptlogik
if check_gpu; then
    start_ollama "gpu"
else
    start_ollama "cpu"
fi
