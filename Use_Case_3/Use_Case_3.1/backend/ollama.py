import requests


class OllamaLLM:
    """Wrapper für Ollama LLM."""
    def _call(self, prompt):
        try:
            # POST-Anfrage an den LM Studio Chat Completions Endpunkt
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions",  # API-Endpunkt
                json={
                    "model": "llama-3.2-3b-instruct",  # Modellname
                    "messages": [
                        { "role": "system", "content": "Gebe eine Antwort zu dem Prompt von dem User ohne weitere Hinweise, Informationen, Kontext oder sonstiges sondern nur zu dem Prompt antworten" },  # Systemrolle mit Anweisung
                        { "role": "user", "content": prompt }  # Benutzerrolle mit Eingabe
                    ],
                    "max_tokens": -1,  # Keine Begrenzung der Tokens
                    "stream": False  # Setze auf False, wenn die gesamte Antwort abgewartet werden soll
                },
                timeout=900  # Timeout in Sekunden
            )
            response.raise_for_status()  # Überprüft auf HTTP-Fehler
            # Extrahiere die Antwort aus der JSON-Antwort
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Fehler: Keine Antwort erhalten.")
        except requests.exceptions.RequestException as e:
            return f"Fehler bei der Verbindung zu LM Studio: {str(e)}"
