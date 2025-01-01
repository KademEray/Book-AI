from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import HashingVectorizer
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
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

# Initialisiere Chroma mit persistentem Speicher
client = PersistentClient(path="./chroma_storage")

# Erstelle oder erhalte eine Sammlung (Collection)
vectorstore = client.get_or_create_collection(
    name="conversation_context"
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
        
class AgentSystem:
    def __init__(self):
        self.agents = []
        self.session_id = str(uuid.uuid4())  # Eindeutige ID für die Sitzung

    def add_agent(self, agent_function, kontrolliert=False):
        self.agents.append({"function": agent_function, "kontrolliert": kontrolliert})

    def run_agents(self, user_input):
        logger.debug(f"run_agents called with user_input: {user_input}")
        response_data = {"steps": [], "final_response": ""}
        context = self.get_context()

        # Schritt 1: Speichere den initialen Benutzerinput
        user_input_id = self.store_context("User Input", user_input)

        # Schritt 2: Synopsis-Agent
        validated_synopsis = None
        while not validated_synopsis:
            logger.debug("Starting synopsis_agent...")
            synopsis_result = synopsis_agent(user_input, context)
            logger.debug(f"synopsis_agent result: {synopsis_result}")
            response_data["steps"].append(synopsis_result["log"])

            if synopsis_result["log"]["status"] == "completed":
                validated_synopsis = synopsis_result["output"]
                # Speichere die validierte Synopsis
                synopsis_id = self.store_context("Synopsis", validated_synopsis)
            else:
                logger.error("Fehler beim Erstellen der Synopsis. Wiederhole...")

        # Schritt 3: Kapitelstruktur-Agent
        validated_chapters = None
        while not validated_chapters:
            logger.debug("Starting chapter_agent...")
            chapter_result = chapter_agent()  # Keine Argumente übergeben
            logger.debug(f"chapter_agent result: {chapter_result}")
            response_data["steps"].append(chapter_result["log"])

            if chapter_result["log"]["status"] == "completed":
                validation_result = chapter_validation_agent(chapter_result["output"])
                response_data["steps"].append(validation_result["log"])

                if validation_result["log"]["status"] == "completed":
                    validated_chapters = chapter_result["output"]
                    # Speichere die validierten Kapitel im Kontext (kein Rückgabewert erforderlich)
                    self.store_context("Chapters", validated_chapters)
                else:
                    logger.error("Kapitelvalidierung fehlgeschlagen. Wiederhole...")
            else:
                logger.error("chapter_agent did not complete, retrying...")

        # Schritt 4: Schreiben der Kapitel
        final_text = None
        while not final_text:
            logger.debug("Starting writing_agent...")
            writing_result = writing_agent(user_input, validated_chapters)
            logger.debug(f"writing_agent result: {writing_result}")
            response_data["steps"].append(writing_result["log"])

            if writing_result["log"]["status"] == "completed":
                final_text = writing_result["output"]
                # Speichere die Ergebnisse des Schreib-Agenten
                final_text_id = self.store_context("Final Text", final_text)
            else:
                logger.error("Writing failed, retrying...")

        # Finales Ergebnis zurückgeben
        response_data["final_response"] = final_text
        logger.debug(f"All agents completed. Returning response_data: {response_data}")
        return response_data


    def get_next_document_id(self):
        """Ermittelt die nächste ID basierend auf der Anzahl der gespeicherten Dokumente."""
        try:
            all_data = vectorstore.get()  # Alle Daten abrufen
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
            vectorstore.upsert(
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
            results = vectorstore.get()  # Alle Dokumente abrufen
            documents = results.get("documents", [])
            logger.info(f"Kontext erfolgreich abgerufen: {len(documents)} Dokument(e) gefunden.")
            return "\n".join(documents)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Kontexts: {e}")
            return "Standardkontext: Keine vorherigen Daten gefunden."

    def get_context_all(self):
        try:
            results = vectorstore.get()  # Alle Daten aus der Collection abrufen
            documents = results.get("documents", [])
            if not documents:
                logger.warning("Keine gespeicherten Dokumente gefunden.")
                return "Keine gespeicherten Dokumente verfügbar."
            logger.info(f"{len(documents)} Dokument(e) erfolgreich abgerufen.")
            return "\n".join(documents)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen aller Dokumente: {e}")
            return "Standardkontext: Keine Dokumente gefunden."

    def validate_saved_data(self):
        """Validiert, ob gespeicherte Daten direkt abrufbar sind."""
        try:
            logger.debug("Validiere gespeicherte Daten...")
            results = vectorstore.get(where={"metadata.session_id": {"$eq": self.session_id}})
            logger.debug(f"Validierungsergebnis: {results}")
            if not results.get("documents", []):
                logger.error("Gespeicherte Daten konnten nicht abgerufen werden.")
            return results.get("documents", [])
        except Exception as e:
            logger.error(f"Fehler bei der Validierung gespeicherter Daten: {e}")
            return []

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

def chapter_agent():
    log = {"agent": "ChapterAgent", "status": "running", "details": []}
    try:
        # Kontext aus allen Dokumenten abrufen
        context = agent_system.get_context_all()
        if not context:
            raise ValueError("Kein Kontext verfügbar.")

        # Präziser Prompt zur Generierung der Kapitelliste
        prompt = f"""
        Basierend auf den folgenden gespeicherten Dokumenten, erstelle eine Liste von Kapiteln für den Dokument.

        Anforderungen:
        1. Jede Kapitelüberschrift muss mit "Kapitel X: " beginnen, wobei X die fortlaufende Nummer des Kapitels ist.
        2. Die Kapitelüberschriften müssen klar und prägnant sein, das Thema präzise beschreiben und dürfen keine weiteren Inhalte enthalten.
        3. Die Reihenfolge der Kapitel muss logisch strukturiert sein, sodass die Themen nahtlos ineinandergreifen.
        4. Vermeide inhaltliche Widersprüche oder Redundanzen zwischen den Kapiteln.
        5. Gib nur die Kapitelüberschriften zurück, ohne zusätzliche Erklärungen, Kommentare oder Formatierungen.

        Format der Ausgabe:
        Kapitel 1: Titel des ersten Kapitels
        Kapitel 2: Titel des zweiten Kapitels
        ...

        Gespeicherte Dokumente:
        {context}
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
            # Wenn die Validierung erfolgreich war, gib die validierten Kapitel zurück
            log.update({"status": "completed", "output": chapter_dict})
            return {"log": log, "output": chapter_dict}
        else:
            # Wenn die Validierung fehlschlägt, erneut versuchen
            corrections = validation_result["log"]["output"].get("corrections", None)
            invalid_chapter_dict = validation_result["log"]["output"].get("reason", chapter_dict)

            if corrections:
                logger.warning(f"Validation failed. Reattempting with corrections: {corrections}")
                corrected_prompt = f"""
                Hier sind die ursprünglichen Kapitel, die überarbeitet werden sollen:
                {invalid_chapter_dict}

                Bitte überarbeite sie basierend auf den folgenden Vorschlägen:
                {corrections}
                """
                corrected_response = llm._call(corrected_prompt).strip()
                corrected_chapter_list = extract_chapter_list(corrected_response)

                if not validate_chapter_structure(corrected_chapter_list):
                    raise ValueError("Die korrigierte Kapitelstruktur ist ungültig.")

                corrected_chapter_dict = build_chapter_dictionary(corrected_chapter_list)
                log.update({"status": "completed", "output": corrected_chapter_dict})
                return {"log": log, "output": corrected_chapter_dict}

            else:
                log.update({"status": "failed", "output": invalid_chapter_dict})
                return {"log": log, "output": f"Validierung fehlgeschlagen: {invalid_chapter_dict}"}

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

# Validierungslogik für Kapitelinhalte
def validate_chapter_content(chapters):
    issues = []
    for chapter in chapters:
        title = chapter["Title"]
        content = chapter["Content"]

        if not content.startswith(title.split(": ")[1]):
            issues.append({
                "Chapter": chapter["Number"],
                "Issue": "Content does not clearly align with the title",
                "Suggested Action": "Revise content to align with the chapter title"
            })
        if "allgemein" in content or "Beispiel" in content:
            issues.append({
                "Chapter": chapter["Number"],
                "Issue": "Content appears too general or incomplete",
                "Suggested Action": "Expand and specify content to align with the chapter theme"
            })
    return issues

def validate_chapter_structure(chapter_list):
    try:
        if not chapter_list or len(chapter_list) < 2:
            logger.error("Kapitel-Liste ist leer oder enthält zu wenige Kapitel.")
            return False

        for i, chapter in enumerate(chapter_list, start=1):
            if not chapter.startswith(f"Kapitel {i}:"):
                logger.error(f"Kapitel {i} fehlt oder ist falsch nummeriert: {chapter}")
                return False

        logger.debug("Kapitelstruktur erfolgreich validiert.")
        return True
    except Exception as e:
        logger.error(f"Fehler bei der Kapitelvalidierung: {e}")
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

        Überprüfen Sie die folgende Kapitelstruktur und Inhalte auf Korrektheit, Konsistenz und Präzision.

        Kapitelstruktur:
        {chapters_text}

        Anforderungen:
        1. Alle Kapitel müssen sinnvolle und logische Titel haben.
        2. Die Inhalte müssen inhaltlich konsistent sein und keine Widersprüche enthalten.
        3. Die Kapitel sollten logisch aufeinander aufbauen und thematisch kohärent sein.
        4. Kapitelinhalte müssen klar mit den Titeln übereinstimmen und dürfen nicht zu allgemein oder unvollständig sein.

        Antworten Sie mit:
        - "Ja" am Anfang, wenn die Kapitelstruktur und Inhalte korrekt sind, gefolgt von einer kurzen Begründung.
        - "Nein" am Anfang, gefolgt von einer detaillierten Begründung und Korrekturvorschlägen, falls die Inhalte angepasst werden müssen.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        logger.debug(f"LLM response: {response}")

        # Prüfung der Antwort
        if response.startswith("Ja"):
            log.update({
                "status": "completed",
                "output": chapter_dict,  # Originaldaten zurückgeben
                "details": ["Kapitelstruktur und Inhalte sind inhaltlich korrekt."]
            })
            agent_system.store_context(chapters_text, "Kapitelstruktur inhaltlich validiert.")
            return {"log": log}

        elif response.startswith("Nein"):
            reason = response.split("Begründung:", 1)[-1].strip() if "Begründung:" in response else "Keine Begründung erhalten."
            corrections = response.split("Korrekturvorschläge:", 1)[-1].strip() if "Korrekturvorschläge:" in response else "Keine Korrekturvorschläge erhalten."

            # Zusätzliche Validierung der Inhalte
            content_issues = validate_chapter_content(chapter_dict["Chapters"])
            if content_issues:
                corrections += f"\nZusätzliche Validierungsprobleme: {json.dumps(content_issues, indent=2)}"

            log.update({
                "status": "failed",
                "output": {
                    "reason": reason,
                    "corrections": corrections
                },
                "details": ["Kapitelstruktur oder Inhalte sind fehlerhaft."]
            })
            return {"log": log}

        else:
            # Unbekannte Antwort behandeln
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
if not agent_system.validate_saved_data():
    logger.warning("Es gibt ungültige oder nicht abrufbare gespeicherte Daten.")
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
