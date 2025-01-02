from flask import Flask, request, jsonify
import json
import chromadb
from chromadb import PersistentClient
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
                timeout=180  # Timeout in Sekunden
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
        logger.debug("Rufe alle gespeicherten Kontexte ab...")
        context = agent_system.get_context_all()
        if not context:
            raise ValueError("Kein Kontext verfügbar.")
        logger.debug(f"Erhaltener Kontext:\n{context}")

        validated_chapter_dict = None
        chapters = []  # Variable zum Speichern der validierten Kapitel
        iteration = 1  # Iterationszähler für die Kapitelgenerierung

        while not validated_chapter_dict:
            logger.info(f"Kapitelgenerierung - Iteration {iteration} gestartet.")
            prompt = f"""
            Basierend auf den folgenden gespeicherten Dokumenten, erstelle eine Liste von Kapiteln für das Dokument.

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
            logger.debug("Sende Prompt zur Kapitelgenerierung an LLM...")
            response = llm._call(prompt).strip()
            logger.debug(f"LLM-Antwort zur Kapitelgenerierung:\n{response}")

            chapter_list = extract_chapter_list(response)
            logger.debug(f"Extrahierte Kapitel-Liste:\n{chapter_list}")

            if not validate_chapter_structure(chapter_list):
                logger.error("Kapitelstruktur ungültig. Wiederhole Generierung...")
                iteration += 1
                continue
            
            print(f"Zeile 313:\n{chapter_list}")
            chapter_dict = build_chapter_dictionary(chapter_list)
            logger.debug(f"Generiertes Kapitel-Dictionary:\n{chapter_dict}")

            if not chapter_dict["Chapters"]:
                raise ValueError("Keine Kapitel generiert. Unterkapitel können nicht erstellt werden.")

            logger.info("Starte Kapitelvalidierung...")
            validation_result = chapter_validation_agent(chapter_dict)
            logger.debug(f"Ergebnis der Kapitelvalidierung:\n{validation_result}")

            if validation_result["log"]["status"] == "completed":
                validated_chapter_dict = chapter_dict
                chapters = validated_chapter_dict["Chapters"]
                logger.info("Kapitelvalidierung erfolgreich abgeschlossen.")
            else:
                logger.error("Kapitelvalidierung fehlgeschlagen. Wiederhole...")
                corrections = validation_result["log"]["output"].get("corrections", None)
                if corrections:
                    corrected_prompt = f"""
                    Hier sind die ursprünglichen Kapitel, die überarbeitet werden sollen:
                    {validation_result['log']['output']}

                    Anforderungen:
                    {corrections}
                    """
                    logger.debug("Korrekturvorschläge zur Kapitelvalidierung generieren...")
                    corrected_response = llm._call(corrected_prompt).strip()
                    logger.debug(f"LLM-Antwort zur Kapitelkorrektur:\n{corrected_response}")
                    chapter_list = extract_chapter_list(corrected_response)
                    print(f"Zeile 344:\n{chapter_list}")
                    chapter_dict = build_chapter_dictionary(chapter_list)

            iteration += 1

        for chapter in chapters:
            logger.info(f"Generiere Unterkapitel für Kapitel {chapter['Number']}: {chapter['Title']}")
            subchapter_result = subchapter_agent(chapter)
            logger.debug(f"Ergebnis der Unterkapitelgenerierung für Kapitel {chapter['Title']}:\n{subchapter_result}")

            if subchapter_result["log"]["status"] == "completed":
                # Verknüpfe Subchapters mit dem aktuellen Kapitel
                chapter["Subchapters"] = subchapter_result["chapter"].get("Subchapters", [])
                log["details"].append(subchapter_result["log"])
                logger.info(f"Unterkapitel erfolgreich für Kapitel {chapter['Number']} generiert.")
            else:
                logger.error(f"Fehler bei der Generierung der Unterkapitel für Kapitel {chapter['Number']}: {chapter['Title']}")
                raise ValueError(f"Fehler bei der Unterkapitelgenerierung für Kapitel: {chapter['Title']}")

        log.update({"status": "completed", "output": validated_chapter_dict})
        logger.info("Kapitel- und Unterkapitelgenerierung erfolgreich abgeschlossen.")
        return {"log": log, "output": validated_chapter_dict}

    except Exception as e:
        log.update({"status": "failed", "output": f"Error: {str(e)}"})
        logger.error(f"Fehler im chapter_agent: {e}")
        return {"log": log, "output": f"Error: {str(e)}"}

def subchapter_agent(current_chapter):
    log = {"agent": "SubchapterAgent", "status": "running", "details": []}
    try:
        subchapter_validated = False
        llm = OllamaLLM()
        chapter_number = current_chapter.get("Number", "X")
        chapter_title = current_chapter.get("Title", "Unbekanntes Kapitel")
        subchapter_list = []

        while not subchapter_validated:
            subchapter_prompt = f"""
            Erstelle eine Liste von Unterkapiteln für das Kapitel "{chapter_title}".
            
            Anforderungen:
            1. Jedes Unterkapitel muss mit "Subchapter {chapter_number}.Y: " beginnen, wobei Y die fortlaufende Nummer des Unterkapitels ist.
            2. Die Unterkapitel müssen prägnante Titel haben, die das Thema des Kapitels weiter gliedern.
            3. Vermeide inhaltliche Wiederholungen oder widersprüchliche Titel.
            4. Gib nur die Unterkapitelüberschriften zurück, ohne zusätzliche Erklärungen, Kommentare oder Formatierungen.

            Format der Ausgabe:
            Subchapter {chapter_number}.1: Titel des ersten Unterkapitels
            Subchapter {chapter_number}.2: Titel des zweiten Unterkapitels
            ...
            """
            logger.debug(f"Subchapter prompt for chapter '{chapter_title}':\n{subchapter_prompt}")

            try:
                subchapter_response = llm._call(subchapter_prompt).strip()
                logger.debug(f"Antwort von LLM für Unterkapitel von Kapitel '{chapter_title}':\n{subchapter_response}")
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der LLM-Antwort: {e}")
                raise

            try:
                subchapter_list = extract_chapter_list(
                    subchapter_response, 
                    is_subchapter=True, 
                    chapter_number=chapter_number
                )
                logger.debug(f"Extrahierte Unterkapitel für Kapitel '{chapter_title}': {subchapter_list}")
            except ValueError as ve:
                logger.error(f"Fehler bei der Extraktion der Unterkapitel für Kapitel '{chapter_title}': {ve}")
                raise

            if validate_chapter_structure(subchapter_list, is_subchapter=True):
                logger.info(f"Unterkapitel für Kapitel '{chapter_title}' erfolgreich validiert.")
                subchapter_validated = True
            else:
                logger.warning(f"Ungültige Struktur der Unterkapitel für Kapitel '{chapter_title}'. Starte Korrektur...")

                correction_prompt = f"""
                Die generierten Unterkapitel sind fehlerhaft. Überarbeite sie basierend auf den folgenden Anforderungen:
                {subchapter_response}

                Anforderungen:
                1. Jedes Unterkapitel muss mit "Subchapter {chapter_number}.Y: " beginnen.
                2. Titel müssen logisch, prägnant und konsistent sein.
                3. Entferne redundante oder widersprüchliche Titel.
                """
                try:
                    corrected_response = llm._call(correction_prompt).strip()
                    logger.debug(f"Korrigierte Antwort für Unterkapitel von Kapitel '{chapter_title}':\n{corrected_response}")
                    corrected_subchapter_list = extract_chapter_list(
                        corrected_response, 
                        is_subchapter=True, 
                        chapter_number=chapter_number
                    )

                    if validate_chapter_structure(corrected_subchapter_list, is_subchapter=True):
                        logger.info(f"Korrigierte Unterkapitel für Kapitel '{chapter_title}' validiert.")
                        subchapter_list = corrected_subchapter_list
                        subchapter_validated = True
                    else:
                        raise ValueError("Korrektur der Unterkapitelstruktur fehlgeschlagen.")
                except Exception as e:
                    logger.error(f"Fehler bei der Korrektur der Unterkapitel für Kapitel '{chapter_title}': {e}")
                    raise

        # Füge die validierten Subchapters zu current_chapter["Subchapters"] hinzu
        for subchapter in subchapter_list:
            subchapter_parts = subchapter.split(": ", 1)
            subchapter_number = subchapter_parts[0].split(" ")[1]
            subchapter_title = subchapter_parts[1]
            current_chapter["Subchapters"].append({
                "Number": subchapter_number,
                "Title": subchapter_title
            })

        log.update({"status": "completed", "chapter": current_chapter})
        return {"log": log, "chapter": current_chapter}

    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten der Unterkapitel für Kapitel '{current_chapter['Title']}': {e}")
        log.update({"status": "failed", "error": str(e)})
        return {"log": log, "chapter": current_chapter}

def extract_chapter_list(response, is_subchapter=False, chapter_number=None):
    try:
        logger.debug("Starte die Extraktion der Kapitel-/Subkapitelliste.")
        logger.debug(f"Eingehende Antwort:\n{response}")
        
        # Bereinigen der Antwort
        response = response.strip()
        logger.debug(f"Bereinigte Antwort:\n{response}")
        
        # Regex für Kapitel oder Subkapitel auswählen
        if is_subchapter and chapter_number is not None:
            pattern = rf"Subchapter {chapter_number}\.\d+: .+"
            logger.debug(f"Verwende Regex für Subkapitel: {pattern}")
        else:
            pattern = r"Kapitel \d+: .+"
            logger.debug(f"Verwende Regex für Kapitel: {pattern}")
        
        # Extrahiere Kapitel oder Subkapitel basierend auf dem Pattern
        items = re.findall(pattern, response)
        logger.debug(f"Gefundene Items mit Regex: {items}")
        
        # Alternative Extraktion, wenn regulärer Ausdruck fehlschlägt
        if not items:
            logger.warning("Regex-Extraktion fehlgeschlagen. Verwende alternative Methode.")
            prefix = f"Subchapter {chapter_number}." if is_subchapter and chapter_number else "Kapitel"
            items = [line.strip() for line in response.split("\n") if line.startswith(prefix)]
            logger.debug(f"Gefundene Items mit alternativer Methode: {items}")
        
        if not items:
            error_message = "Kein gültiges Format gefunden."
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Bereinigen und Duplikate entfernen
        items = list(dict.fromkeys(item.strip() for item in items))
        logger.info(f"Erfolgreich extrahierte Items: {items}")
        return items
    
    except Exception as e:
        logger.error(f"Fehler bei der Extraktion: {str(e)}")
        raise ValueError(f"Fehler bei der Extraktion: {str(e)}")

def validate_chapter_structure(item_list, is_subchapter=False):
    try:
        if not item_list:
            logger.error("Liste ist leer.")
            return False
        
        pattern = r"Subchapter \d+\.\d+: .+" if is_subchapter else r"Kapitel \d+: .+"
        for item in item_list:
            if not re.match(pattern, item):
                logger.error(f"Ungültiges Item: {item}")
                return False
        
        logger.info("Alle Items sind gültig.")
        return True
    except Exception as e:
        logger.error(f"Fehler bei der Validierung: {str(e)}")
        return False
    
def build_chapter_dictionary(chapter_list):
    try:
        if not chapter_list:
            raise ValueError("Die Kapitel-Liste ist leer.")

        chapter_dict = {"Chapters": []}
        current_chapter = None

        for entry in chapter_list:
            entry = entry.strip()
            if entry.startswith("Kapitel "):
                # Extrahiere Kapitelnummer und Titel
                try:
                    chapter_number = int(entry.split(" ")[1].split(":")[0])
                    title = entry.split(": ", 1)[-1].strip()
                except (IndexError, ValueError) as e:
                    logger.error(f"Ungültiges Kapitel-Format: {entry}")
                    continue

                # Erstelle neues Kapitel
                current_chapter = {
                    "Number": chapter_number,
                    "Title": title,
                    "Subchapters": []
                }
                chapter_dict["Chapters"].append(current_chapter)

            elif entry.startswith("Subchapter "):
                # Subkapitel kann nur zu einem Kapitel gehören
                if current_chapter is None:
                    logger.error(f"Subchapter gefunden, ohne Kapitel: {entry}")
                    continue

                try:
                    subchapter_number = entry.split(" ")[1].split(":")[0]
                    subchapter_title = entry.split(": ", 1)[-1].strip()
                except (IndexError, ValueError) as e:
                    logger.error(f"Ungültiges Subkapitel-Format: {entry}")
                    continue

                # Überprüfen, ob das Subkapitel zum aktuellen Kapitel gehört
                if not subchapter_number.startswith(f"{current_chapter['Number']}."):
                    logger.error(f"Ungültige Zuordnung: Subchapter {subchapter_number} gehört nicht zu Kapitel {current_chapter['Number']}.")
                    continue

                # Füge Subkapitel zum aktuellen Kapitel hinzu
                subchapter = {
                    "Number": subchapter_number,
                    "Title": subchapter_title
                }
                current_chapter["Subchapters"].append(subchapter)

        if not chapter_dict["Chapters"]:
            raise ValueError("Keine Kapitel gefunden.")
        
        return chapter_dict

    except Exception as e:
        logger.error(f"Fehler: {e}")
        raise

def chapter_validation_agent(chapter_dict, **kwargs):
    log = {"agent": "ChapterValidationAgent", "status": "running", "details": []}
    try:
        # Kontext abrufen
        logger.debug("Rufe den Kontext für die Kapitelvalidierung ab...")
        context = agent_system.get_context()
        if not context:
            logger.warning("Kein Kontext verfügbar. Verwende Standardkontext.")
            context = "Standardkontext: Keine vorherigen Daten gefunden."

        # Kapitelstruktur in Textformat umwandeln
        logger.debug("Erstelle Kapitelstruktur als Textformat für die Validierung...")
        chapters_text = "\n".join(
            [f"Kapitel {chapter['Number']}: {chapter['Title']}" for chapter in chapter_dict.get("Chapters", [])]
        )
        logger.debug(f"Kapitelstruktur:\n{chapters_text}")

        # Übergeordnete Validierung der Kapitel
        prompt = f"""
        Kontext:
        {context}

        Überprüfen Sie die folgende Kapitelstruktur und Inhalte auf Korrektheit, Konsistenz und Präzision.

        Kapitelstruktur:
        {chapters_text}

        Anforderungen:
        1. Alle Kapitel müssen sinnvolle und logische Titel haben.
        2. Die Kapitel sollten logisch aufeinander aufbauen und thematisch kohärent sein.
        3. Kapitelinhalte müssen klar mit den Titeln übereinstimmen.

        Antworten Sie mit:
        - "Ja" am Anfang, wenn die Kapitelstruktur korrekt ist, gefolgt von einer kurzen Begründung.
        - "Nein" am Anfang, gefolgt von einer detaillierten Begründung und Korrekturvorschlägen.
        """
        llm = OllamaLLM()
        logger.debug("Sende Anfrage zur Kapitelstrukturvalidierung an LLM...")
        response = llm._call(prompt).strip()
        logger.debug(f"LLM-Antwort zur Kapitelvalidierung:\n{response}")

        # Prüfung der Kapitelantwort
        if response.startswith("Nein"):
            reason = response.split("Begründung:", 1)[-1].strip() if "Begründung:" in response else "Keine Begründung erhalten."
            corrections = response.split("Korrekturvorschläge:", 1)[-1].strip() if "Korrekturvorschläge:" in response else "Keine Korrekturvorschläge erhalten."
            logger.error(f"Kapitelvalidierung fehlgeschlagen. Begründung: {reason}, Korrekturvorschläge: {corrections}")

            log.update({
                "status": "failed",
                "output": {
                    "reason": reason,
                    "corrections": corrections
                },
                "details": ["Kapitelstruktur ist fehlerhaft."]
            })
            return {"log": log}

        # Validierung der Unterkapitel
        for chapter in chapter_dict.get("Chapters", []):
            logger.debug(f"Verarbeite Kapitel {chapter['Number']}: {chapter['Title']}")

            subchapters_text = "\n".join(
                [f"Unterkapitel {sub['Number']}: {sub['Title']}" for sub in chapter.get("Subchapters", [])]
            )
            if not subchapters_text:
                logger.info(f"Kapitel {chapter['Number']} hat keine Unterkapitel. Überspringe Unterkapitelvalidierung.")
                continue

            logger.debug(f"Erstelle Validierungsprompt für Unterkapitel von Kapitel {chapter['Number']}:\n{subchapters_text}")
            subchapter_prompt = f"""
            Kontext:
            {context}

            Kapitel: {chapter['Title']}

            Überprüfen Sie die folgende Unterkapitelstruktur auf Korrektheit, Konsistenz und Präzision.

            Unterkapitelstruktur:
            {subchapters_text}

            Anforderungen:
            1. Alle Unterkapitel müssen sinnvolle und logische Titel haben.
            2. Die Unterkapitel sollten logisch aufeinander aufbauen und thematisch kohärent sein.
            3. Unterkapitelinhalte müssen klar mit den Titeln übereinstimmen.

            Antworten Sie mit:
            - "Ja" am Anfang, wenn die Unterkapitelstruktur korrekt ist, gefolgt von einer kurzen Begründung.
            - "Nein" am Anfang, gefolgt von einer detaillierten Begründung und Korrekturvorschlägen.
            """
            logger.debug("Sende Anfrage zur Unterkapitelvalidierung an LLM...")
            subchapter_response = llm._call(subchapter_prompt).strip()
            logger.debug(f"LLM-Antwort zur Unterkapitelvalidierung von Kapitel {chapter['Number']}:\n{subchapter_response}")

            if subchapter_response.startswith("Nein"):
                reason = subchapter_response.split("Begründung:", 1)[-1].strip() if "Begründung:" in subchapter_response else "Keine Begründung erhalten."
                corrections = subchapter_response.split("Korrekturvorschläge:", 1)[-1].strip() if "Korrekturvorschläge:" in subchapter_response else "Keine Korrekturvorschläge erhalten."
                logger.error(f"Unterkapitelvalidierung fehlgeschlagen für Kapitel {chapter['Number']}. Begründung: {reason}, Korrekturvorschläge: {corrections}")

                log.update({
                    "status": "failed",
                    "output": {
                        "reason": reason,
                        "corrections": corrections
                    },
                    "details": [f"Unterkapitel von Kapitel {chapter['Number']} sind fehlerhaft."]
                })
                return {"log": log}

        # Erfolgsmeldung, wenn alle Kapitel und Unterkapitel validiert wurden
        log.update({
            "status": "completed",
            "output": chapter_dict,
            "details": ["Kapitel und Unterkapitel sind inhaltlich korrekt."]
        })
        logger.info("Kapitel- und Unterkapitelvalidierung erfolgreich abgeschlossen.")
        agent_system.store_context(chapters_text, "Kapitel und Unterkapitel validiert.")
        return {"log": log}

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

            chapter_content = {"Number": chapter["Number"], "Title": chapter["Title"], "Subchapters": []}
            
            # Verarbeite Unterkapitel
            for subchapter in chapter.get("Subchapters", []):
                subchapter_validated = False
                while not subchapter_validated:
                    try:
                        subchapter_prompt = f"""
                        Kapitel {chapter['Number']} - {chapter['Title']}
                        Unterkapitel {subchapter['Number']} - {subchapter['Title']}

                        Schreibe den vollständigen Text für dieses Unterkapitel. Konzentriere dich ausschließlich auf den Inhalt des Unterkapitels 
                        und vermeide jegliche Hinweise oder Erklärungen zum Benutzerinput oder Schreibprozess. Gib nur den reinen Text des Unterkapitels zurück.
                        """
                        llm = OllamaLLM()
                        subchapter_content = llm._call(subchapter_prompt)
                        logger.debug(f"[DEBUG] Generierter Unterkapitelinhalt: {subchapter_content}")

                        # Validierung des Unterkapitelinhalts
                        validation_result = validation_agent(user_input, subchapter_content)
                        logger.debug(f"[DEBUG] Validierungsergebnis für Unterkapitel: {validation_result}")

                        log["details"].append(validation_result["log"])

                        if validation_result["log"].get("status") == "completed":
                            subchapter_validated = True
                            chapter_content["Subchapters"].append({
                                "Number": subchapter["Number"],
                                "Title": subchapter["Title"],
                                "Content": subchapter_content
                            })
                            agent_system.store_context(subchapter["Title"], subchapter_content)
                            logger.debug(f"[DEBUG] Unterkapitel erfolgreich gespeichert: {subchapter['Title']}")
                        else:
                            logger.error(f"[DEBUG] Unterkapitel {subchapter['Number']} nicht validiert. Wiederhole...")

                    except Exception as e:
                        logger.error(f"[DEBUG] Fehler beim Verarbeiten des Unterkapitels {subchapter['Number']}: {e}")
                        log["details"].append({
                            "status": "failed",
                            "error": str(e),
                            "subchapter": subchapter["Title"]
                        })
                        break  # Bricht die Schleife für dieses Unterkapitel ab, um Endlosschleifen zu vermeiden

            # Füge Kapitel mit Unterkapiteln zum finalen Text hinzu
            final_text["Chapters"].append(chapter_content)

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
