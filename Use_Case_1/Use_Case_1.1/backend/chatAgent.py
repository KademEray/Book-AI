from datetime import datetime
import logging

from chromadb import PersistentClient

from ollama import OllamaLLM


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

class ChatAgent:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def chat(self, user_input):
        log = {"agent": "ChatAgent", "status": "running", "details": []}
        try:
            # Kontext abrufen
            logger.debug("Rufe den vollständigen Kontext ab...")
            context = self.get_context_all()
            logger.debug(f"Erhaltener Kontext:\n{context}")

            # Aktualisierter Prompt für die KI
            prompt = f"""
            Dies ist ein fortlaufender Chat. Der Kontext enthält alle vorherigen Interaktionen zwischen dem Benutzer und der KI.

            Kontext:
            {context}

            Neue Benutzeranfrage:
            {user_input}

            Basierend auf dem Kontext und der neuen Anfrage, gib bitte eine passende Antwort:
            """
            logger.debug(f"Prompt für ChatAgent:\n{prompt}")
            
            # KI anfragen
            llm = OllamaLLM()
            response = llm._call(prompt).strip()
            logger.debug(f"Antwort von LLM:\n{response}")

            # Kontext speichern
            self.store_context("User Input", user_input)
            self.store_context("AI Response", response)

            log.update({
                "status": "completed",
                "output": response,
                "details": ["Chat erfolgreich abgeschlossen."]
            })
            return {"log": log, "final_response": response}
        
        except Exception as e:
            log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
            logger.error(f"Fehler im ChatAgent: {e}")
            return {"log": log, "final_response": f"Fehler: {str(e)}"}

    def get_next_document_id(self):
        """Ermittelt die nächste ID basierend auf der Anzahl der gespeicherten Dokumente."""
        try:
            all_data = self.vectorstore.get()  # Alle Daten abrufen
            existing_ids = all_data.get("ids", [])
            return str(len(existing_ids) + 1)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der nächsten ID: {e}")
            return "1"  # Fallback auf ID "1", falls ein Fehler auftritt

    def store_context(self, label, data):
        """Speichert den Kontext in ChromaDB mit einer fortlaufenden ID."""
        try:
            doc_id = self.get_next_document_id()  # Zugriff auf die Instanzmethode
            metadata = {"timestamp": datetime.now().isoformat()}
            logger.info(f"Speichere Kontext: {label} -> {data} (ID: {doc_id})")  # Nur die ersten 100 Zeichen loggen

            # Dokument speichern
            self.vectorstore.upsert(
                documents=[f"{label}: {data}"],
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.debug("Speichern erfolgreich.")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Kontexts: {e}")

    def get_context(self):
        """Ruft den gespeicherten Kontext aus ChromaDB ab."""
        try:
            logger.debug("Abfrage aller Dokumente aus der Collection.")
            results = self.vectorstore.get()  # Alle Dokumente abrufen
            documents = results.get("documents", [])
            logger.info(f"Kontext erfolgreich abgerufen: {len(documents)} Dokument(e) gefunden.")
            return "\n".join(documents)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Kontexts: {e}")
            return "Standardkontext: Keine vorherigen Daten gefunden."

    def get_context_all(self):
        """Ruft alle gespeicherten Dokumente ab."""
        try:
            results = self.vectorstore.get()  # Alle Daten aus der Collection abrufen
            documents = results.get("documents", [])
            if not documents:
                logger.warning("Keine gespeicherten Dokumente gefunden.")
                return "Keine gespeicherten Dokumente verfügbar."
            logger.info(f"{len(documents)} Dokument(e) erfolgreich abgerufen.")
            return "\n".join(documents)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen aller Dokumente: {e}")
            return "Standardkontext: Keine Dokumente gefunden."
