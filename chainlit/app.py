import chainlit as cl
import httpx  # Python HTTP-Client
import uuid  # Für die Generierung einer eindeutigen Chat-ID

# Generiere eine eindeutige Chat-ID
def generate_chat_id():
    return str(uuid.uuid4())

# Wird bei Chat-Start ausgeführt, um die Chat-ID zu initialisieren
@cl.on_chat_start
async def on_chat_start():
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        chat_id = generate_chat_id()
        cl.user_session.set("chat_id", chat_id)
    
# Wird bei jeder eingehenden Nachricht ausgeführt
@cl.on_message
async def main(message):
    user_input = message.content  # Inhalt der Nachricht
    chat_id = cl.user_session.get("chat_id")  # Hole die gespeicherte Chat-ID
    if not chat_id:
        # Sicherstellen, dass eine Chat-ID vorhanden ist
        chat_id = generate_chat_id()
        cl.user_session.set("chat_id", chat_id)

    print(f"User Input: {user_input}, Chat ID: {chat_id}")  # Debugging

    # Konfiguriere den Timeout auf "unendlich"
    timeout_config = httpx.Timeout(None)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Sende die Anfrage an das Flask-Backend
            response = await client.post(
                url="http://localhost:5000/api/generate",
                json={
                    "user_input": user_input,
                    "chat_id": chat_id  # Verwende die gespeicherte Chat-ID
                }
            )
            response.raise_for_status()  # Fehler bei HTTP-Status >= 400

            # Verarbeite die Antwort
            response_data = response.json()
            if "response" in response_data:
                await cl.Message(content=response_data["response"]).send()
            else:
                await cl.Message(content="Fehler: Keine gültige Antwort erhalten.").send()

        except httpx.RequestError as e:
            print(f"HTTP-Anfrage fehlgeschlagen: {e}")
            await cl.Message(content="Fehler bei der Verbindung zum Backend.").send()
        except Exception as e:
            print(f"Allgemeiner Fehler: {e}")
            await cl.Message(content="Ein unerwarteter Fehler ist aufgetreten.").send()
