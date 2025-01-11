import os
from flask import Flask, request, jsonify, send_file, Response
from chromadb import PersistentClient
import logging
from agent import AgentSystem
from chatAgent import ChatAgent
from duckduckgo import DuckDuckGoSearch
import zipfile


logging.basicConfig(
    filename='Use_Case_1/Use_Case_1.1/backend/backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Initialisiere Chroma mit persistentem Speicher
client = PersistentClient(path="./Use_Case_1/Use_Case_1.1/backend/chroma_storage")

# Erstelle oder erhalte eine Sammlung (Collection)
vectorstore = client.get_or_create_collection(
    name="conversation_context"
)
logger = logging.getLogger(__name__)

# Flask-Setup
app = Flask(__name__)
RESULTS_DIR = "Use_Case_1/Use_Case_1.1/Ergebnisse"


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        min_chapter = data.get("min_chapter", 0)  # min_chapter erfassen
        min_subchapter = data.get("min_subchapter", 0)  # min_subchapter erfassen
        logger.info(f"Received request: user_input={user_input}, min_chapter={min_chapter}")
        
        # min_chapter an run_agents übergeben
        agent_system = AgentSystem()
        result = agent_system.run_agents(user_input, min_chapter=min_chapter, min_subchapter=min_subchapter)
        if not result:
            raise ValueError("Die Antwortstruktur ist unvollständig.")
        
        logger.info(f"Result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during request processing: {e}")
        return jsonify({"error": f"Fehler: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        logger.info(f"Received chat request: user_input={user_input}")
        
        # Chat-Agent erstellen
        chat_agent = ChatAgent(vectorstore)
        result = chat_agent.chat(user_input)

        if not result or "final_response" not in result:
            raise ValueError("Die Antwortstruktur ist unvollständig.")
        
        logger.info(f"Result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during chat request processing: {e}")
        return jsonify({"error": f"Fehler: {str(e)}"}), 500
    
@app.route('/api/search', methods=['POST'])
def search():
    try:
        # Anfrage-Daten verarbeiten
        data = request.get_json()
        request_input = data.get("request_input", "").strip()
        if not request_input:
            raise ValueError("Die Suchanfrage ist leer. Bitte einen gültigen Suchbegriff angeben.")

        request_input = request_input.replace('"', '').replace("'", "").strip()

        logger.info(f"Received search request: request_input={request_input}")

        # DuckDuckGo-Suche durchführen
        search_agent = DuckDuckGoSearch()
        result = search_agent.perform_search(query=request_input)

        # Überprüfen, ob Ergebnisse vorhanden sind
        if not result or not result.get("results"):
            logger.warning("Keine relevanten Ergebnisse gefunden.")
            return jsonify({"results": [], "message": "Keine relevanten Ergebnisse gefunden."})

        logger.info(f"Search Result: {result}")
        return jsonify(result)

    except ValueError as e:
        logger.error(f"Validation Error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during search request processing: {e}")
        return jsonify({"error": f"Fehler: {str(e)}"}), 500
    

# CORS aktivieren
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
