from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_chroma import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
import requests
from datetime import datetime

# Flask-Setup
app = Flask(__name__)

# Custom LLM, das Ollama als Backend verwendet
class OllamaLLM(LLM):
    def _call(self, prompt, stop=None):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False  # Deaktiviere Streaming
                },
                timeout=30,
            )
            response.raise_for_status()
            
            # Parse die JSON-Antwort
            response_json = response.json()
            if "response" in response_json:
                return response_json["response"]
            else:
                print("Unerwartetes Antwortformat:", response_json)
                return "Fehler: Unerwartetes Antwortformat von Ollama"
                
        except requests.exceptions.RequestException as e:
            print(f"Fehler bei der Kommunikation mit der KI: {e}")
            return "Fehler bei der Kommunikation mit der KI."
        except json.JSONDecodeError as e:
            print(f"Fehler beim Dekodieren der JSON-Antwort: {e}")
            return "Fehler beim Verarbeiten der Antwort."

    @property
    def _identifying_params(self):
        return {"model": "llama3.2:3b"}

    @property
    def _llm_type(self):
        return "ollama_llm"

# Benutzerdefinierte Embedding-Funktion, die das Embeddings-Interface implementiert
class CustomEmbeddingFunction(Embeddings):
    def __init__(self, n_features=128):
        print(f"Initializing HashingVectorizer with {n_features} features.")
        self.vectorizer = HashingVectorizer(n_features=n_features, norm=None, alternate_sign=False)
        self.dimension = n_features

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.vectorizer.transform(texts).toarray()
            return embeddings.tolist()
        except Exception as e:
            print(f"Fehler bei der Batch-Embedding-Berechnung: {e}")
            return [[0.0] * self.dimension] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        try:
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Fehler bei der Query-Embedding-Berechnung: {e}")
            return [0.0] * self.dimension

# Instanziiere die Embedding-Funktion
custom_embedding_function = CustomEmbeddingFunction(n_features=128)

# Integration mit Chroma
vectorstore = Chroma(
    collection_name="conversation_context",
    embedding_function=custom_embedding_function
)

# Funktion zum Speichern einer neuen Konversation
def add_conversation(user_input: str, response: str, chat_id: str):
    try:
        context = f"User: {user_input}\nAssistant: {response}"
        vectorstore.add_texts(
            texts=[context],
            metadatas=[{
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat()
            }]
        )
        print(f"Konversation zur ChromaDB hinzugefügt für Chat {chat_id}")
    except Exception as e:
        print(f"Fehler beim Hinzufügen zur ChromaDB: {e}")

# Funktion zum Abrufen des Chat-Verlaufs
def get_chat_history(chat_id: str):
    try:
        # Hole alle Dokumente aus der Collection
        results = vectorstore.get()
        
        # Filtere nach der chat_id
        if results and results['ids']:
            filtered_documents = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata.get('chat_id') == chat_id:
                    filtered_documents.append(results['documents'][i])
            
            # Sortiere nach Zeitstempel, falls vorhanden
            return filtered_documents
        return []
    except Exception as e:
        print(f"Fehler beim Abrufen des Chat-Verlaufs: {e}")
        return []

# Konfiguriere Ollama als LLM
llm = OllamaLLM()

@app.route("/api/generate", methods=["POST"])
def generate():
    print("Anfrage erhalten")
    data = request.json
    user_input = data.get("user_input", "")
    chat_id = data.get("chat_id", "default")  # Chat-ID aus der Anfrage

    if not user_input:
        print("Keine Eingabe erhalten")
        return jsonify({"error": "Keine Eingabe erhalten"}), 400

    try:
        print(f"User Input: {user_input}")

        # Hole den aktuellen Chat-Verlauf
        chat_history = get_chat_history(chat_id)
        conversation_context = "\n\n".join(chat_history) if chat_history else ""
        
        # Erstelle den Prompt mit dem Chat-Verlauf
        prompt = f"""Bisherige Konversation:
{conversation_context}

User: {user_input}

Bitte antworte basierend auf der bisherigen Konversation."""

        # Generiere Antwort mit Ollama
        response = llm._call(prompt)
        
        # Speichere die neue Konversation
        add_conversation(user_input, response, chat_id)
        
        return jsonify({
            "response": response,
            "chat_id": chat_id
        })

    except Exception as e:
        print(f"Fehler bei der Anfrage: {e}")
        return jsonify({"error": f"Fehler bei der Kommunikation mit der KI: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
