import chainlit as cl
import httpx
import uuid

# Generiere eine eindeutige Chat-ID
def generate_chat_id():
    return str(uuid.uuid4())

# Wird bei Chat-Start ausgeführt
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

    timeout_config = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Sende die Anfrage an das Backend
            response = await client.post(
                url="http://localhost:5000/api/generate",
                json={
                    "user_input": user_input,
                    "chat_id": chat_id
                }
            )
            response.raise_for_status()

            # Verarbeite die Antwort
            response_data = response.json()

            # Zeige Fortschritt der Agenten
            for step in response_data.get("steps", []):
                agent_name = step["agent"]
                status = step["status"]
                output = step.get("output", "Keine Ausgabe verfügbar.") if agent_name != "ValidationAgent" else ""
                validation_output = f"Validierung: {step['output']}" if agent_name == "ValidationAgent" else ""
                await cl.Message(content=f"Agent: {agent_name}\nStatus: {status}\n{validation_output}{output}").send()

            # Zeige finale Antwort
            final_response = response_data.get("final_response", "Keine Antwort.")
            await cl.Message(content=f"Finale Antwort:\n{final_response}").send()

        except httpx.RequestError as e:
            await cl.Message(content=f"Fehler bei der Verbindung zum Backend: {e}").send()
        except Exception as e:
            await cl.Message(content=f"Ein unerwarteter Fehler ist aufgetreten: {e}").send()
