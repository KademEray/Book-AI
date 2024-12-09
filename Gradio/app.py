import gradio as gr
import requests
import uuid

# Generiere eine eindeutige Chat-ID
def generate_chat_id():
    return str(uuid.uuid4())

# Chat-Session-Management
chat_sessions = {}

# Backend-Aufruf
def handle_message(user_input, session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = generate_chat_id()

    chat_id = chat_sessions[session_id]

    try:
        # Backend-Anfrage
        response = requests.post(
            "http://localhost:5000/api/generate",
            json={"user_input": user_input, "chat_id": chat_id}
        )
        response.raise_for_status()

        # Verarbeite Backend-Antwort
        data = response.json()
        steps = "\n\n".join([
            f"**Agent**: {step.get('agent', 'Unbekannt')}\n**Status**: {step.get('status', 'Unbekannt')}\n**Output**: {step.get('output', 'Keine Ausgabe verfügbar')}"
            for step in data.get("steps", [])
        ])
        final_response = data.get("final_response", "Keine finale Antwort verfügbar")

        # Kombiniere Schritte und finale Antwort
        return f"**Schritte:**\n\n{steps}\n\n**Finale Antwort:**\n\n{final_response}"

    except requests.exceptions.RequestException as e:
        return f"Fehler bei der Verbindung zum Backend: {str(e)}"
    except Exception as e:
        return f"Unerwarteter Fehler: {str(e)}"

# Gradio-Interface
with gr.Blocks() as demo:
    gr.Markdown("## Chat mit dem KI-System")
    with gr.Row():
        with gr.Column():
            chatbox = gr.Chatbot(label="Chat")
            user_input = gr.Textbox(placeholder="Ihre Nachricht eingeben...")
            submit_button = gr.Button("Senden")
            session_id = gr.State(value=str(uuid.uuid4()))

    def chat_interaction(user_input, history, session_id):
        response = handle_message(user_input, session_id)
        history.append((user_input, response))
        return history, ""

    submit_button.click(
        chat_interaction,
        inputs=[user_input, chatbox, session_id],
        outputs=[chatbox, user_input]
    )

# Starte die Gradio-Anwendung
demo.launch()
