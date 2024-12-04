from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

@app.route("/api/generate", methods=["POST"])
def generate():
    print("Anfrage erhalten")
    data = request.json
    user_input = data.get("user_input", "")

    if not user_input:
        print("Keine Eingabe erhalten")
        return jsonify({"error": "Keine Eingabe erhalten"}), 400

    try:
        print("Sende Anfrage an Ollama...")
        print(f"Payload: {{'model': 'llama3.2:3b', 'prompt': {user_input}}}")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": user_input
            },
            stream=True
        )
        print("Status Code:", response.status_code)
        print("Antworttext:", response.text)

        response.raise_for_status()

        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    json_chunk = json.loads(chunk.decode("utf-8"))
                    if "response" in json_chunk:
                        full_response += json_chunk["response"]
                except json.JSONDecodeError as e:
                    print("JSONDecodeError:", e)

        if not full_response:
            return jsonify({"error": "Keine gültige Antwort erhalten"}), 500

        return jsonify({"response": full_response})

    except requests.exceptions.RequestException as e:
        print(f"Fehler bei der Anfrage: {e}")
        return jsonify({"error": f"Fehler bei der Kommunikation mit der KI: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
