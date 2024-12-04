import chainlit as cl
import httpx  # Python HTTP-Client

@cl.on_message
async def main(message):
    user_input = message.content  # Greife auf das Attribut 'content' zu
    print(f"User Input: {user_input}")  # Debugging
    
    # Konfiguriere den Timeout auf "unendlich"
    timeout_config = httpx.Timeout(None)  # Kein Timeout

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Sende die Anfrage an das Flask-Backend
            response = await client.post(
                url="http://localhost:5000/api/generate",
                json={"user_input": user_input}
            )
            response.raise_for_status()  # Wirft einen Fehler bei HTTP-Statuscodes >= 400
            
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
