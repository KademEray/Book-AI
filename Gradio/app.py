import gradio as gr
import requests
import uuid

# Generiere eine eindeutige Chat-ID
def generate_chat_id():
    return str(uuid.uuid4())

# Chat-Session-Management
chat_sessions = {}

# Backend-Aufruf
def handle_message(user_input, uploaded_file, session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = generate_chat_id()

    chat_id = chat_sessions[session_id]

    try:
        # Dateiinhalt lesen, falls vorhanden
        file_content = None
        if uploaded_file:
            with open(uploaded_file.name, "r", encoding="utf-8") as file:
                file_content = file.read()

        # Backend-Anfrage vorbereiten
        payload = {"user_input": user_input, "chat_id": chat_id}
        if file_content:
            payload["file_content"] = file_content

        # Anfrage an das Backend
        response = requests.post("http://backend:5000/api/generate", json=payload)
        response.raise_for_status()

        # Verarbeite Backend-Antwort
        data = response.json()
        steps = "\n\n".join([
            f"**Agent**: {step.get('agent', 'Unbekannt')}\n**Status**: {step.get('status', 'Unbekannt')}\n**Output**: {step.get('output', 'Keine Ausgabe verfügbar')}"
            for step in data.get("steps", [])
        ])
        final_response = data.get("final_response", "Keine finale Antwort verfügbar")

        return steps, final_response

    except requests.exceptions.RequestException as e:
        return f"Fehler bei der Verbindung zum Backend: {str(e)}", ""
    except Exception as e:
        return f"Unerwarteter Fehler: {str(e)}", ""

# Gradio-Interface
with gr.Blocks() as demo:
    gr.Markdown("## Interaktives Chat- und Dateisystem mit KI")
    with gr.Row():
        with gr.Column():
            chatbox = gr.Chatbot(label="Chatverlauf", type='messages')
            file_upload = gr.File(label="Datei hochladen (optional)", file_types=[".txt", ".pdf"])
            user_input = gr.Textbox(label="Ihre Nachricht eingeben...", placeholder="Ihre Nachricht hier eingeben...")
            submit_button = gr.Button("Senden")
            session_id = gr.State(value=str(uuid.uuid4()))

        with gr.Column():
            steps_output = gr.Textbox(label="Agenten-Schritte", lines=15, interactive=False)
            final_output = gr.Textbox(label="Finale Antwort", lines=5, interactive=False)

    def chat_interaction(user_input, uploaded_file, history, session_id):
        steps, final_response = handle_message(user_input, uploaded_file, session_id)
        # Aktualisiere die Historie im korrekten Format
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_response})
        return history, steps, final_response, None

    submit_button.click(
        chat_interaction,
        inputs=[user_input, file_upload, chatbox, session_id],
        outputs=[chatbox, steps_output, final_output, user_input]
    )

# Starte die Gradio-Anwendung
demo.launch(server_name="0.0.0.0", server_port=3000, debug=True)
