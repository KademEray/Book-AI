from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import requests
from datetime import datetime
import re
import logging
import uuid

logging.basicConfig(
    filename='backend/backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

# Flask-Setup
app = Flask(__name__)

class OllamaLLM:
    """Wrapper für Ollama LLM."""
    def _call(self, prompt):
        try:
            # POST-Anfrage an den LM Studio Chat Completions Endpunkt
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions",  # API-Endpunkt
                json={
                    "model": "llama-3.2-3b-instruct:2",  # Modellname
                    "messages": [
                        { "role": "system", "content": "Gebe eine Antwort zu dem Prompt von dem User ohne weitere Hinweise, Informationen, Kontext oder sonstiges sondern nur zu dem Prompt antworten" },  # Systemrolle mit Anweisung
                        { "role": "user", "content": prompt }  # Benutzerrolle mit Eingabe
                    ],
                    "temperature": 0.7,  # Einstellbare Temperatur
                    "max_tokens": -1,  # Keine Begrenzung der Tokens
                    "stream": False  # Setze auf False, wenn die gesamte Antwort abgewartet werden soll
                },
                timeout=3000  # Timeout in Sekunden
            )
            response.raise_for_status()  # Überprüft auf HTTP-Fehler
            # Extrahiere die Antwort aus der JSON-Antwort
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Fehler: Keine Antwort erhalten.")
        except requests.exceptions.RequestException as e:
            return f"Fehler bei der Verbindung zu LM Studio: {str(e)}"
        
# Benutzerdefinierte Embedding-Funktion
class CustomEmbeddingFunction(Embeddings):
    def __init__(self, n_features=128):
        logger.info(f"Initializing HashingVectorizer with {n_features} features.")
        self.vectorizer = HashingVectorizer(n_features=n_features, norm=None, alternate_sign=False)
        self.dimension = n_features

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.vectorizer.transform(texts).toarray()
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Fehler bei der Batch-Embedding-Berechnung: {e}")
            return [[0.0] * self.dimension] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Fehler bei der Query-Embedding-Berechnung: {e}")
            return [0.0] * self.dimension

# Instanziiere die Embedding-Funktion und Chroma
custom_embedding_function = CustomEmbeddingFunction(n_features=128)
vectorstore = Chroma(
    collection_name="conversation_context",
    embedding_function=custom_embedding_function
)

# Funktion zur Extraktion des Dictionaries
def extract_dictionary(output, variable_name="X_Dict"):
    try:
        # Sucht nach der genauen Struktur des Dictionaries
        pattern = rf"{variable_name}\s*=\s*\{{.*?\}}"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            extracted_dict = match.group(0)
            try:
                # Evaluiere die extrahierte Dictionary-Struktur
                parsed_dict = eval(extracted_dict.split('=', 1)[1].strip())
                return parsed_dict
            except SyntaxError as parse_error:
                raise ValueError(f"Fehler beim Parsen des Dictionaries: {parse_error}")
        else:
            # Debug-Ausgabe, falls kein Treffer
            raise ValueError(f"'{variable_name}' konnte im Output nicht gefunden werden. Ausgabe: {output}")
    except Exception as e:
        raise ValueError(f"Fehler bei der Extraktion des Dictionaries: {e}")

class AgentSystem:
    def __init__(self):
        self.agents = []
        self.session_id = str(uuid.uuid4())  # Generiere eine eindeutige Session-ID

    def add_agent(self, agent_function, kontrolliert=False):
        self.agents.append({"function": agent_function, "kontrolliert": kontrolliert})

    def run_agents(self, user_input):
        logger.debug(f"run_agents called with user_input: {user_input}")
        response_data = {"steps": [], "final_response": ""}
        context = self.get_context()
        logger.debug(f"Abgerufener Kontext: {context}")

        # Schritt 0: Benutzerinput speichern
        self.store_context("User Input", user_input)

        # Schritt 1: Synopsis-Agent ausführen
        validated_synopsis = None
        while not validated_synopsis:
            logger.debug("Starting synopsis_agent...")
            synopsis_result = synopsis_agent(user_input, context)
            logger.debug(f"synopsis_agent result: {synopsis_result}")
            response_data["steps"].append(synopsis_result["log"])
            
            if synopsis_result["log"].get("status") == "completed":
                validated_synopsis = synopsis_result["output"]
                logger.debug(f"synopsis_agent completed with: {validated_synopsis}")
                # Kontext speichern
                self.store_context("Synopsis", validated_synopsis)
            else:
                logger.error("Fehler beim Erstellen der Synopsis. Wiederhole...")

        # Schritt 2: Kapitelstruktur-Agent ausführen und validieren
        validated_chapters = None
        while not validated_chapters:
            logger.debug("Starting chapter_agent...")
            chapter_result = chapter_agent(user_input, validated_synopsis)
            logger.debug(f"chapter_agent result: {chapter_result}")
            response_data["steps"].append(chapter_result["log"])
            
            if chapter_result["log"].get("status") == "completed":
                logger.debug("chapter_agent completed, now validating chapters...")
                chapter_validation_result = chapter_validation_agent(chapter_result["output"])
                logger.debug(f"chapter_validation_agent result: {chapter_validation_result}")
                response_data["steps"].append(chapter_validation_result["log"])
                
                if chapter_validation_result["log"].get("status") == "completed":
                    validated_chapters = chapter_result["output"]  # Verwende Originalausgabe von chapter_agent
                    logger.debug(f"Chapters validated: {validated_chapters}")
                    # Kontext speichern
                    self.store_context("Chapters", validated_chapters)
                else:
                    logger.error("Validation failed, retrying...")
            else:
                logger.error("chapter_agent did not complete, retrying...")

        # Schritt 3: Schreiben der Kapitel mit writing_agent
        final_text = None
        while not final_text:
            logger.debug("Starting writing_agent...")
            writing_result = writing_agent(user_input, validated_chapters)
            logger.debug(f"writing_agent result: {writing_result}")
            response_data["steps"].append(writing_result["log"])
            
            if writing_result["log"].get("status") == "completed":
                final_text = writing_result["output"]
                logger.debug(f"writing_agent completed with: {final_text}")
                # Kontext speichern
                self.store_context("Final Text", final_text)
            else:
                logger.error("Writing failed, retrying...")

        # Finales Ergebnis speichern
        response_data["final_response"] = final_text
        logger.debug(f"All agents completed. Returning response_data: {response_data}")
        return response_data

    def get_context(self):
        try:
            # Kontext aus vectorstore abrufen
            results = vectorstore.get(where={"metadata.session_id": {"$eq": self.session_id}})
            logger.debug(f"Vectorstore get results: {results}")  # Debugging

            # Kontextdaten extrahieren
            context_data = results.get('documents', [])
            if not context_data:
                logger.warning("Kein gespeicherter Kontext gefunden.")
                return "Standardkontext: Keine vorherigen Daten gefunden."
            
            # Kontext zurückgeben
            return "\n".join(context_data)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Kontexts: {e}")
            return "Standardkontext: Keine vorherigen Daten gefunden."

    def store_context(self, user_input, final_output):
        try:
            logger.info(f"Speichere Kontext: {user_input} -> {final_output}")
            vectorstore.add_texts(
                texts=[f"User: {user_input}\nAssistant: {final_output}"],
                metadatas=[{"session_id": self.session_id, "timestamp": datetime.now().isoformat()}]
            )
            logger.debug(f"Stored texts in vectorstore: {user_input} -> {final_output}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Kontexts: {e}")

# Synopsis-Agent
def synopsis_agent(user_input, context):
    log = {"agent": "SynopsisAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Kontext:
        {context}

        Aufgabe: Erstelle eine prägnante Synopsis basierend auf dem folgenden Benutzerinput:
        {user_input}
        """
        logger.debug(f"synopsis_agent prompt:\n{prompt}")
        llm = OllamaLLM()
        response = llm._call(prompt)

        log.update({"status": "completed", "output": response})
        return {"log": log, "output": response}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": f"Fehler: {str(e)}"}

# Validierungs-Agent
def validation_agent(user_input, output):
    log = {"agent": "ValidationAgent", "status": "processing"}

    try:
        logger.debug("[DEBUG] ValidationAgent gestartet")
        logger.debug(f"[DEBUG] Benutzerinput: {user_input}")
        logger.debug(f"[DEBUG] Ausgabe zum Validieren: {output}")

        context = agent_system.get_context()  # Kontext abrufen
        logger.debug(f"[DEBUG] Abgerufener Kontext: {context}")

        prompt = f"""
        Überprüfe die folgende Ausgabe darauf, ob sie inhaltlich mit der Benutzeranfrage und dem Kontext übereinstimmt.

        Benutzeranfrage:
        {user_input}

        Kontext:
        {context}

        Ausgabe:
        {output}

        Antworte mit:
        1. "Ja" oder "Nein", ob die Ausgabe inhaltlich korrekt ist.
        2. Begründung, warum die Ausgabe korrekt oder falsch ist.
        """
        llm = OllamaLLM()
        validation_result = llm._call(prompt)
        logger.debug(f"[DEBUG] Validierungsergebnis von LLM: {validation_result}")

        if "Ja" in validation_result:
            log.update({
                "status": "completed",
                "output": "Validierung erfolgreich. Ausgabe ist inhaltlich korrekt."
            })
        else:
            reason = validation_result.split("Begründung:")[1].strip() if "Begründung:" in validation_result else "Unzureichende Begründung erhalten."
            log.update({
                "status": "failed",
                "output": f"Validierung fehlgeschlagen: {reason}"
            })

    except Exception as e:
        log.update({
            "status": "failed",
            "output": f"Fehler bei der Validierung: {str(e)}"
        })
        logger.error(f"[DEBUG] Fehler in ValidationAgent: {e}")

    return {"log": log}

def chapter_agent(user_input, validated_synopsis):
    log = {"agent": "ChapterAgent", "status": "running", "details": []}
    try:
        # Präziser Prompt zur Generierung der Kapitelliste
        prompt = f"""
        Basierend auf der folgenden validierten Synopsis, erstelle eine strukturierte Liste von Kapiteln.

        Validierte Synopsis:
        {validated_synopsis}

        Anforderungen:
        Erstelle eine Liste in Form eines Python-Lists mit so vielen Kapiteln wie möglich basierend auf der validierten Synopsis.
        Jedes Kapitel sollte mit "Kapitel" beginnen, gefolgt von einer Nummer und einem Titel. Beispiel: "Kapitel 1: Einleitung".
        Die Liste soll eine logische Reihenfolge haben und keine Wiederholungen enthalten. Erstelle nur die Python-Liste und gib sie als Text zurück.
        Keine weiteren Informationen oder Erklärungen.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die Kapitel aus der Antwort
        chapter_list = extract_chapter_list(response)
        logger.debug(f"Generated chapter list: {chapter_list}")

        # Validierung der Kapitelstruktur
        if not validate_chapter_structure(chapter_list):
            raise ValueError("Kapitelstruktur ist ungültig oder unvollständig.")

        # Erstelle das Kapitel-Dictionary
        chapter_dict = build_chapter_dictionary(chapter_list)

        # Validierung der Kapitelstruktur durch den Agenten
        validation_result = chapter_validation_agent(chapter_dict)
        log["details"].append(validation_result["log"])

        if validation_result["log"]["status"] == "completed":
            log.update({"status": "completed", "output": chapter_dict})
            return {"log": log, "output": chapter_dict}
        else:
            raise ValueError("Validation of chapter structure failed.")

    except Exception as e:
        log.update({"status": "failed", "output": f"Error: {str(e)}"})
        logger.error(f"Error in chapter_agent: {e}")
        return {"log": log, "output": f"Error: {str(e)}"}

def extract_chapter_list(response):
    """
    Extrahiert die Kapitel aus der KI-Antwort und validiert das Format.
    """
    try:
        # Bereinigen der Antwort und Extrahieren der Kapitel
        response = response.replace('"', '').replace("'", "").strip()
        chapters = re.findall(r"Kapitel \d+: .+", response)
        
        if not chapters:
            logger.warning("Kein gültiges Kapitel-Format gefunden. Versuche alternative Extraktion.")
            # Alternative Extraktion
            chapters = [line.strip() for line in response.split("\n") if line.startswith("Kapitel")]
        
        if not chapters:
            raise ValueError("Kein gültiges Kapitel-Format gefunden.")
        
        # Bereinigung der Kapitel-Liste
        chapters = [chapter.strip() for chapter in chapters]
        return chapters
    except Exception as e:
        raise ValueError(f"Fehler bei der Extraktion der Kapitel-Liste: {str(e)}")


def validate_chapter_structure(chapter_list):
    """
    Validiert die Struktur der Kapitel-Liste.
    """
    try:
        if not chapter_list or len(chapter_list) < 2:
            logger.error("Kapitel-Liste ist leer oder enthält zu wenige Kapitel.")
            return False

        seen_titles = set()
        for i, chapter in enumerate(chapter_list, start=1):
            expected_prefix = f"Kapitel {i}:"
            if not chapter.startswith(expected_prefix):
                logger.error(f"Kapitelnummerierung stimmt nicht: {chapter}. Erwartet: {expected_prefix}")
                return False

            title = chapter.split(": ", 1)[-1]
            if title in seen_titles:
                logger.error(f"Doppelter Titel erkannt: {title}")
                return False
            seen_titles.add(title)

        return True
    except Exception as e:
        logger.error(f"Fehler bei der Validierung der Kapitelstruktur: {e}")
        return False

def build_chapter_dictionary(chapter_list):
    """
    Baut ein Kapitel-Dictionary basierend auf der Liste der Kapitel auf.
    """
    try:
        chapter_dict = {"Chapters": []}
        for i, chapter in enumerate(chapter_list, start=1):
            # Extrahiere den Titel, indem der Präfix "Kapitel X: " entfernt wird
            title = chapter.split(": ", 1)[-1]
            chapter_dict["Chapters"].append({
                "Number": i,
                "Title": title,
                "Subchapters": []  # Subchapters können später hinzugefügt werden
            })
        return chapter_dict
    except Exception as e:
        raise ValueError(f"Fehler beim Erstellen des Kapitel-Dictionaries: {str(e)}")


def chapter_validation_agent(chapter_dict, **kwargs):
    log = {"agent": "ChapterValidationAgent", "status": "running", "details": []}
    try:
        # Kontext abrufen
        context = agent_system.get_context()
        if not context:
            logger.debug("Warnung: Kein Kontext verfügbar. Verwende Standardkontext.")
            context = "Standardkontext: Keine vorherigen Daten gefunden."

        # Kapitel in ein textbasiertes Format umwandeln
        chapters_text = "\n".join(
            [f"Kapitel {chapter['Number']}: {chapter['Title']}" for chapter in chapter_dict.get("Chapters", [])]
        )

        # Prompt für die Validierung der Inhalte
        prompt = f"""
        Kontext:
        {context}

        Überprüfen Sie die folgende Kapitelstruktur auf inhaltliche Korrektheit und Konsistenz.

        Kapitelstruktur:
        {chapters_text}

        Anforderungen:
        1. Alle Kapitel müssen sinnvolle und logische Titel haben.
        2. Die Inhalte müssen inhaltlich konsistent sein und keine Widersprüche enthalten.
        3. Die Kapitel sollten logisch aufeinander aufbauen und thematisch kohärent sein.

        Antworten Sie am Anfang des Satzes mit:
        - "Ja", wenn die Kapitelstruktur inhaltlich korrekt und konsistent ist.
        - "Nein", gefolgt von einer detaillierten Begründung und Korrekturvorschlägen, falls die Inhalte angepasst werden müssen.
        """
        # LLM-Aufruf
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Prüfung der Antwort
        if response.startswith("Ja"):
            log.update({
                "status": "completed",
                "output": chapter_dict,  # Originaldaten zurückgeben
                "details": ["Kapitelstruktur ist inhaltlich korrekt."]
            })
            agent_system.store_context(chapters_text, "Kapitelstruktur inhaltlich validiert.")
            return {"log": log}

        elif response.startswith("Nein"):
            reason = response.split("Begründung:", 1)[-1].strip() if "Begründung:" in response else "Keine Begründung erhalten."
            corrections = response.split("Korrekturvorschläge:", 1)[-1].strip() if "Korrekturvorschläge:" in response else "Keine Korrekturvorschläge erhalten."
            log.update({
                "status": "failed",
                "output": {
                    "reason": reason,
                    "corrections": corrections
                },
                "details": ["Kapitelstruktur ist inhaltlich fehlerhaft."]
            })
            return {"log": log}

        else:
            raise ValueError(f"Unbekannte Antwort vom LLM: {response}")

    except ValueError as e:
        error_message = f"Fehler bei der Validierung: {e}"
        logger.error(error_message)
        log.update({
            "status": "failed",
            "output": error_message,
            "details": ["Validierungsprozess abgebrochen."]
        })
        return {"log": log}

    except Exception as e:
        error_message = f"Ein unerwarteter Fehler ist aufgetreten: {e}"
        logger.error(error_message)
        log.update({
            "status": "failed",
            "output": error_message,
            "details": ["Unbekannter Fehler im Agenten."]
        })
        return {"log": log}

def writing_agent(user_input, validated_chapters):
    log = {"agent": "WritingAgent", "status": "running", "details": []}
    final_text = {"Chapters": []}

    try:
        logger.debug("WritingAgent gestartet")
        logger.debug(f"[DEBUG] validated_chapters: {validated_chapters}")

        for chapter in validated_chapters.get("Chapters", []):
            logger.debug(f"[DEBUG] Verarbeite Kapitel: {chapter}")
            validated = False

            while not validated:
                try:
                    prompt = f"""
                    Kapitel {chapter['Number']} - {chapter['Title']}

                    Schreibe den vollständigen Text für dieses Kapitel. Konzentriere dich ausschließlich auf den Inhalt des Kapitels und vermeide jegliche Hinweise oder Erklärungen zum Benutzerinput oder zum Schreibprozess. Gib nur den reinen Text des Kapitels zurück.
                    """
                    llm = OllamaLLM()
                    chapter_content = llm._call(prompt)
                    logger.debug(f"[DEBUG] Generierter Kapitelinhalt: {chapter_content}")

                    # Kapiteltext validieren
                    validation_result = validation_agent(user_input, chapter_content)
                    logger.debug(f"[DEBUG] Validierungsergebnis: {validation_result}")

                    log["details"].append(validation_result["log"])

                    if validation_result["log"].get("status") == "completed":
                        validated = True
                        final_text["Chapters"].append({
                            "Number": chapter["Number"],
                            "Title": chapter["Title"],
                            "Content": chapter_content
                        })
                        agent_system.store_context(chapter["Title"], chapter_content)
                        logger.debug(f"[DEBUG] Kapitel erfolgreich gespeichert: {chapter['Title']}")
                    else:
                        logger.error(f"[DEBUG] Kapitel {chapter['Number']} nicht validiert. Wiederhole...")

                except Exception as e:
                    logger.error(f"[DEBUG] Fehler beim Verarbeiten des Kapitels {chapter['Number']}: {e}")
                    log["details"].append({
                        "status": "failed",
                        "error": str(e),
                        "chapter": chapter["Title"]
                    })
                    break  # Bricht die Schleife für dieses Kapitel ab, um nicht in eine Endlosschleife zu geraten

        if final_text["Chapters"]:
            log.update({"status": "completed", "output": final_text})
            logger.debug(f"[DEBUG] WritingAgent erfolgreich abgeschlossen: {final_text}")
            return {"log": log, "output": final_text}
        else:
            raise ValueError("Kein Kapitel konnte erfolgreich verarbeitet werden.")

    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        logger.error(f"[DEBUG] Fehler in WritingAgent: {e}")
        return {"log": log, "output": {}}


# Initialisiere das Agentensystem
agent_system = AgentSystem()
agent_system.add_agent(synopsis_agent, kontrolliert=True)
agent_system.add_agent(chapter_agent, kontrolliert=True)
agent_system.add_agent(chapter_validation_agent, kontrolliert=False)
agent_system.add_agent(writing_agent, kontrolliert=True)


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        logger.info(f"Received request: user_input={user_input}")
        
        result = agent_system.run_agents(user_input)
        if not result or "final_response" not in result:
            raise ValueError("Die Antwortstruktur ist unvollständig.")
        
        logger.info(f"Result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during request processing: {e}")
        return jsonify({"error": f"Fehler: {str(e)}"}), 500


# CORS aktivieren
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
