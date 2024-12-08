import chainlit as cl
import requests
import uuid

# Generiere eine eindeutige Chat-ID
def generate_chat_id():
    return str(uuid.uuid4())

# Wird bei Chat-Start ausgeführt
@cl.on_chat_start
async def on_chat_start():
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        cl.user_session.set("chat_id", generate_chat_id())

# Wird bei jeder eingehenden Nachricht ausgeführt
@cl.on_message
async def main(message: cl.Message):
    try:
        print(f"Neue Nachricht erhalten: {message.content}")

        # Statusnachricht senden
        await cl.Message(content=" Anfrage wird verarbeitet...").send()

        # Backend-Anfrage
        response = requests.post(
            "http://localhost:5000/api/generate",
            json={"user_input": message.content, "chat_id": cl.user_session.get("chat_id")}
        )
        print(f"Backend-Antwort: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        print(f"Backend-Daten: {data}")

                # Zeige jeden Schritt im Validierungsprozess an
        for step in data.get("steps", []):
            agent = step.get("agent", "Unbekannt")
            status = step.get("status", "Unbekannt")
            output = step.get("output", "Keine Ausgabe verfügbar")
            
            # Nachricht formatieren
            step_message = f"""
            **Agent**: {agent}
            **Status**: {status}
            **Output**: {output}
            """
            await cl.Message(content=step_message.strip()).send()


        # Finale Antwort anzeigen
        final_response = data.get("final_response", "Keine finale Antwort verfügbar")
        await cl.Message(content="**Finale Antwort**:\n\n" + final_response).send()

    except requests.exceptions.RequestException as e:
        await cl.Message(content=" Fehler bei der Verbindung zum Backend: " + str(e)).send()
    except Exception as e:
        await cl.Message(content=" Unerwarteter Fehler: " + str(e)).send()