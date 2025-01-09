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
                timeout=800  # Timeout in Sekunden
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

    def run_agents(self, user_input, min_chapter=0, min_subchapter=0):
        logger.debug(f"run_agents called with user_input: {user_input}, min_chapter: {min_chapter}")
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
            chapter_result = chapter_agent(min_chapter=min_chapter, min_subchapter=min_subchapter)  # min_chapter übergeben
            logger.debug(f"chapter_agent result: {chapter_result}")
            response_data["steps"].append(chapter_result["log"])

            if chapter_result["log"]["status"] == "completed":
                validation_result = chapter_validation_agent(chapter_result["output"])
                response_data["steps"].append(validation_result["log"])

                if validation_result["log"]["status"] == "completed":
                    validated_chapters = chapter_result["output"]
                    # Speichere die validierten Kapitel im Kontext
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
                
        # Schritt 5: Erstellung und Validierung der Zusammenfassungen
        summaries = self.generate_summary(validated_chapters)
        validated_summaries = None

        while not validated_summaries:
            logger.debug("Validiere Zusammenfassungen...")
            validation_result = self.validate_summary(summaries)

            # Überprüfe, ob alle Zusammenfassungen validiert wurden
            if all(summary.get("Validated", False) for summary in validation_result):
                validated_summaries = validation_result
                logger.info("Alle Zusammenfassungen erfolgreich validiert.")
                self.store_context("Validated Summaries", validated_summaries)
            else:
                logger.warning("Eine oder mehrere Zusammenfassungen sind fehlerhaft. Wiederhole...")
                summaries = [
                    {
                        "Chapter": summary["Chapter"],
                        "Summary": summary["Summary"]
                    }
                    for summary in validation_result
                    if not summary.get("Validated", False)
                ]
                
        # Schritt 6: Buch bewerten
        logger.debug("Starte Buchbewertung...")
        weighted_scores = []

        # Bewertung durch Agenten
        weighted_scores.append(evaluate_chapters(validated_chapters)["output"])
        weighted_scores.append(evaluate_paragraphs(validated_chapters)["output"])
        weighted_scores.append(evaluate_book_type(validated_chapters)["output"])
        weighted_scores.append(evaluate_content(validated_chapters)["output"])
        weighted_scores.append(evaluate_grammar(validated_chapters)["output"])
        weighted_scores.append(evaluate_style(validated_chapters)["output"])
        weighted_scores.append(evaluate_tension(validated_chapters)["output"])

        # Endnote berechnen
        final_evaluation = calculate_final_score(weighted_scores)
        response_data["evaluation"] = final_evaluation
        logger.info(f"Buchbewertung abgeschlossen: {final_evaluation}")
        
        # Finales Ergebnis zurückgeben
        #response_data["final_response"] = final_text
        #logger.debug(f"All agents completed. Returning response_data: {response_data}")
        #return response_data


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

def chapter_agent(min_chapter=0, min_subchapter=0):
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
            6. Erstelle mindestens {min_chapter} Kapitel.

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

            # Überprüfe, ob die Mindestanzahl an Kapiteln erfüllt ist
            if len(chapter_list) < min_chapter:
                logger.warning(f"Generierte Kapitel ({len(chapter_list)}) sind weniger als die Mindestanzahl ({min_chapter}). Wiederhole...")
                iteration += 1
                continue

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
                    chapter_dict = build_chapter_dictionary(chapter_list)

            iteration += 1

        for chapter in chapters:
            logger.info(f"Generiere Unterkapitel für Kapitel {chapter['Number']}: {chapter['Title']}")
            subchapter_result = subchapter_agent(chapter, min_subchapter)
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

def subchapter_agent(current_chapter, min_subchapter):
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
            5. Erstelle mindestens {min_subchapter} Unterkapitel.

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

            # Überprüfen, ob die Mindestanzahl an Unterkapiteln erreicht wurde
            if len(subchapter_list) < min_subchapter:
                logger.warning(f"Generierte Unterkapitel ({len(subchapter_list)}) sind weniger als die Mindestanzahl ({min_subchapter}). Wiederhole...")
                continue

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
    
def generate_summary(self, chapters):
    """Erstellt eine Zusammenfassung für jedes Kapitel."""
    summaries = []
    for chapter in chapters.get("Chapters", []):
        try:
            prompt = f"""
            Kapitel: {chapter['Title']}

            Aufgabe: Erstelle eine prägnante Zusammenfassung des Kapitels.
            Fokussiere dich auf die Hauptpunkte und fasse den Inhalt in wenigen Sätzen zusammen.
            """
            llm = OllamaLLM()
            summary = llm._call(prompt).strip()
            summaries.append({"Chapter": chapter["Title"], "Summary": summary})
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung der Zusammenfassung für Kapitel {chapter['Title']}: {e}")
            summaries.append({"Chapter": chapter["Title"], "Summary": f"Fehler: {str(e)}"})
    return summaries

def validate_summary(self, summaries):
    """Validiert die erstellten Zusammenfassungen."""
    validated_summaries = []
    for summary in summaries:
        try:
            prompt = f"""
            Zusammenfassung: {summary['Summary']}

            Aufgabe: Überprüfe, ob diese Zusammenfassung korrekt, präzise und konsistent mit dem Kapitel ist.
            Antworte mit "Ja" oder "Nein" und einer kurzen Begründung.
            """
            llm = OllamaLLM()
            validation_result = llm._call(prompt).strip()
            if validation_result.startswith("Ja"):
                validated_summaries.append({"Chapter": summary["Chapter"], "Summary": summary["Summary"], "Validated": True})
            else:
                logger.warning(f"Zusammenfassung für Kapitel {summary['Chapter']} nicht validiert: {validation_result}")
                validated_summaries.append({"Chapter": summary["Chapter"], "Summary": summary["Summary"], "Validated": False, "Reason": validation_result})
        except Exception as e:
            logger.error(f"Fehler bei der Validierung der Zusammenfassung für Kapitel {summary['Chapter']}: {e}")
            validated_summaries.append({"Chapter": summary["Chapter"], "Summary": summary["Summary"], "Validated": False, "Reason": f"Fehler: {str(e)}"})
    return validated_summaries

# Gewichtung der Buchbewertung

def evaluate_chapters(validated_chapters):
    log = {"agent": "ChapterEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 5
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Konsistenz
        consistency_score = evaluate_consistency(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Konsistenz", "score": consistency_score})
        total_score += consistency_score

        # Länge
        length_score = evaluate_length(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Länge", "score": length_score})
        total_score += length_score

        # Zusammenfassungen
        summary_score = evaluate_summaries(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Kapitelzusammenfassungen", "score": summary_score})
        total_score += summary_score

        # Übergänge
        transitions_score = evaluate_transitions(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Übergänge", "score": transitions_score})
        total_score += transitions_score

        # Relevanz
        relevance_score = evaluate_relevance(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Relevanz", "score": relevance_score})
        total_score += relevance_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}
        
        
def evaluate_paragraphs(validated_chapters):
    log = {"agent": "ParagraphEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 4
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Fokus
        focus_score = evaluate_focus(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Fokus", "score": focus_score})
        total_score += focus_score

        # Verknüpfung
        linkage_score = evaluate_linkage(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Verknüpfung", "score": linkage_score})
        total_score += linkage_score

        # Lesefluss
        flow_score = evaluate_flow(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Lesefluss", "score": flow_score})
        total_score += flow_score

        # Länge
        length_score = evaluate_paragraph_length(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Länge", "score": length_score})
        total_score += length_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}
        
def evaluate_book_type(validated_chapters):
    log = {"agent": "BookTypeEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 3
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Zielgruppe
        audience_score = evaluate_audience(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Zielgruppe", "score": audience_score})
        total_score += audience_score

        # Formateignung
        format_score = evaluate_format(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Formateignung", "score": format_score})
        total_score += format_score

        # Innovativität
        innovation_score = evaluate_innovation(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Innovativität", "score": innovation_score})
        total_score += innovation_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}


def evaluate_content(validated_chapters):
    log = {"agent": "ContentEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 4
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Tiefe der Recherche
        research_score = evaluate_research(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Tiefe der Recherche", "score": research_score})
        total_score += research_score

        # Fokus
        focus_score = evaluate_focus_on_topic(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Fokus", "score": focus_score})
        total_score += focus_score

        # Aktualität
        timeliness_score = evaluate_timeliness(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Aktualität", "score": timeliness_score})
        total_score += timeliness_score

        # Breite des Inhalts
        breadth_score = evaluate_breadth(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Breite des Inhalts", "score": breadth_score})
        total_score += breadth_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}


def evaluate_grammar(validated_chapters):
    log = {"agent": "GrammarEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 50
    sub_weight_count = 2
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Einheitlichkeit
        consistency_score = evaluate_grammar_consistency(validated_chapters)
        log["details"].append({"criteria": "Einheitlichkeit", "score": consistency_score})
        total_score += consistency_score

        # Qualität der Überarbeitung
        revision_score = evaluate_revision_quality(validated_chapters)
        log["details"].append({"criteria": "Qualität der Überarbeitung", "score": revision_score})
        total_score += revision_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}


def evaluate_style(validated_chapters):
    log = {"agent": "StyleEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 4
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Tonalität
        tone_score = evaluate_tone(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Tonalität", "score": tone_score})
        total_score += tone_score

        # Abwechslung
        variety_score = evaluate_variety(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Abwechslung", "score": variety_score})
        total_score += variety_score

        # Sprachbilder
        imagery_score = evaluate_imagery(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Sprachbilder", "score": imagery_score})
        total_score += imagery_score

        # Authentizität
        authenticity_score = evaluate_authenticity(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Authentizität", "score": authenticity_score})
        total_score += authenticity_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}
        

#WENN ES EIN ROMAN IST IST DAS SINNVOLL !!!!
def evaluate_tension(validated_chapters):
    log = {"agent": "TensionEvaluationAgent", "status": "running", "details": []}
    total_score = 0
    max_score = 100
    sub_weight_count = 4
    max_points_per_sub = max_score / sub_weight_count

    try:
        # Aufbau
        buildup_score = evaluate_buildup(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Aufbau", "score": buildup_score})
        total_score += buildup_score

        # Wendepunkte
        twists_score = evaluate_twists(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Wendepunkte", "score": twists_score})
        total_score += twists_score

        # Charakterentwicklung
        character_dev_score = evaluate_character_development(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Charakterentwicklung", "score": character_dev_score})
        total_score += character_dev_score

        # Cliffhanger
        cliffhanger_score = evaluate_cliffhangers(validated_chapters) * max_points_per_sub
        log["details"].append({"criteria": "Cliffhanger", "score": cliffhanger_score})
        total_score += cliffhanger_score

        log.update({"status": "completed", "output": total_score})
        return {"log": log, "output": total_score}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": 0}
    
# Untergewichtungen Funktionen

def evaluate_consistency(validated_chapters):
    """
    Bewertet die Konsistenz der Kapitel mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 20.
    """
    try:
        prompt = """
        Überprüfe die Konsistenz der Kapitel in Bezug auf logische Verbindungen und inhaltliche Stimmigkeit.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)  # Die KI gibt eine Punktzahl zurück
        return min(max(score, 0), 20)  # Sicherstellen, dass der Wert zwischen 0 und 20 liegt
    except Exception as e:
        logger.error(f"Fehler bei der Konsistenzbewertung: {e}")
        return 0

def evaluate_length(validated_chapters):
    """
    Bewertet die Länge der Kapitel mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 20.
    """
    try:
        prompt = """
        Überprüfe, ob die Kapitel in ihrer Länge einheitlich und angemessen sind.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Längenbewertung: {e}")
        return 0

def evaluate_summaries(validated_chapters):
    """
    Bewertet die Qualität der Kapitelzusammenfassungen mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 20.
    """
    try:
        prompt = """
        Überprüfe die Qualität der Zusammenfassungen für die Kapitel.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelzusammenfassungen:
        """
        for chapter in validated_chapters.get("Chapters", []):
            summary = chapter.get("Summary", "Keine Zusammenfassung vorhanden")
            prompt += f"- Kapitel: {chapter['Title']}, Zusammenfassung: {summary}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Kapitelzusammenfassungen: {e}")
        return 0

def evaluate_transitions(validated_chapters):
    """
    Bewertet die Übergänge zwischen den Kapiteln mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 20.
    """
    try:
        prompt = """
        Überprüfe, ob die Übergänge zwischen den Kapiteln logisch und fließend sind.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Übergänge: {e}")
        return 0

def evaluate_relevance(validated_chapters):
    """
    Bewertet die Relevanz der Kapitel mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 20.
    """
    try:
        prompt = """
        Überprüfe, ob jedes Kapitel einen wesentlichen Beitrag zur Gesamtidee des Buches leistet.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Relevanzbewertung: {e}")
        return 0
    
def evaluate_focus(validated_chapters):
    """
    Bewertet den Fokus der Absätze mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe, ob die Absätze der Kapitel jeweils ein klares und spezifisches Thema behandeln.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Fokusbewertung: {e}")
        return 0

def evaluate_linkage(validated_chapters):
    """
    Bewertet die Verknüpfung der Absätze mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe, ob die Absätze logisch miteinander verknüpft sind.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Verknüpfung: {e}")
        return 0

def evaluate_flow(validated_chapters):
    """
    Bewertet den Lesefluss der Absätze mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe, ob die Absätze den natürlichen Lesefluss fördern.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung des Leseflusses: {e}")
        return 0

def evaluate_paragraph_length(validated_chapters):
    """
    Bewertet die Länge der Absätze mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe, ob die Absätze in ihrer Länge weder zu kurz noch zu lang sind.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Absatzlänge: {e}")
        return 0

def evaluate_audience(validated_chapters):
    """
    Bewertet, ob die Buchart den Erwartungen der Zielgruppe entspricht.
    Rückgabe: Punktzahl zwischen 0 und 33.33.
    """
    try:
        prompt = """
        Überprüfe, ob die Buchart den Erwartungen der Zielgruppe entspricht.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return (score / 20) * 33.33  # Skaliert auf maximal 33.33 Punkte
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Zielgruppe: {e}")
        return 0

def evaluate_format(validated_chapters):
    """
    Bewertet, ob die Buchart für das Thema und die Zielgruppe geeignet ist.
    Rückgabe: Punktzahl zwischen 0 und 33.33.
    """
    try:
        prompt = """
        Überprüfe, ob die Buchart für das Thema und die Zielgruppe passend gewählt ist.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return (score / 20) * 33.33  # Skaliert auf maximal 33.33 Punkte
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Formateignung: {e}")
        return 0

def evaluate_innovation(validated_chapters):
    """
    Bewertet, ob das Buch in seiner Art neue Ansätze bietet.
    Rückgabe: Punktzahl zwischen 0 und 33.33.
    """
    try:
        prompt = """
        Überprüfe, ob das Buch in seiner Art innovative Ansätze bietet oder bewährten Mustern folgt.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return (score / 20) * 33.33  # Skaliert auf maximal 33.33 Punkte
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Innovativität: {e}")
        return 0

def evaluate_research(validated_chapters):
    """
    Bewertet die Tiefe der Recherche mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Tiefe der Recherche in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Tiefe der Recherche: {e}")
        return 0

def evaluate_focus_on_topic(validated_chapters):
    """
    Bewertet den Fokus auf das Thema mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe, ob die Kapitel fokussiert auf das Thema sind.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung des Fokus auf das Thema: {e}")
        return 0

def evaluate_timeliness(validated_chapters):
    """
    Bewertet die Aktualität der Inhalte mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Aktualität der Inhalte in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Aktualität: {e}")
        return 0

def evaluate_breadth(validated_chapters):
    """
    Bewertet die Breite der Inhalte mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Breite der Inhalte in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 25)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Breite der Inhalte: {e}")
        return 0

def evaluate_grammar_consistency(validated_chapters):
    """
    Bewertet die grammatikalische Einheitlichkeit der Kapitel mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 50.
    """
    try:
        prompt = """
        Überprüfe die grammatikalische Einheitlichkeit der Kapitel.
        Gib eine Bewertung auf einer Skala von 0 bis 50 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 50)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der grammatikalischen Einheitlichkeit: {e}")
        return 0

def evaluate_revision_quality(validated_chapters):
    """
    Bewertet die Qualität der Überarbeitung der Kapitel mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 50.
    """
    try:
        prompt = """
        Überprüfe die Qualität der Überarbeitung der Kapitel.
        Gib eine Bewertung auf einer Skala von 0 bis 50 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 50)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Überarbeitungsqualität: {e}")
        return 0   


### !!!!
def evaluate_buildup(validated_chapters):
            try:
                prompt = """
                Überprüfe den Aufbau der Spannung in den Kapiteln.
                Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
                Kapitelstruktur:
                """
                for chapter in validated_chapters.get("Chapters", []):
                    prompt += f"- {chapter['Title']}\n"

                llm = OllamaLLM()
                response = llm._call(prompt).strip()
                score = int(response)
                return min(max(score, 0), 25)
            except Exception as e:
                logger.error(f"Fehler bei der Bewertung des Spannungsaufbaus: {e}")
                return 0

def evaluate_twists(validated_chapters):
            try:
                prompt = """
                Überprüfe die Wendepunkte in den Kapiteln.
                Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
                Kapitelstruktur:
                """
                for chapter in validated_chapters.get("Chapters", []):
                    prompt += f"- {chapter['Title']}\n"

                llm = OllamaLLM()
                response = llm._call(prompt).strip()
                score = int(response)
                return min(max(score, 0), 25)
            except Exception as e:
                logger.error(f"Fehler bei der Bewertung der Wendepunkte: {e}")
                return 0

def evaluate_character_development(validated_chapters):
            try:
                prompt = """
                Überprüfe die Charakterentwicklung in den Kapiteln.
                Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
                Kapitelstruktur:
                """
                for chapter in validated_chapters.get("Chapters", []):
                    prompt += f"- {chapter['Title']}\n"

                llm = OllamaLLM()
                response = llm._call(prompt).strip()
                score = int(response)
                return min(max(score, 0), 25)
            except Exception as e:
                logger.error(f"Fehler bei der Bewertung der Charakterentwicklung: {e}")
                return 0

def evaluate_cliffhangers(validated_chapters):
            try:
                prompt = """
                Überprüfe die Cliffhanger in den Kapiteln.
                Gib eine Bewertung auf einer Skala von 0 bis 25 ab.
                Kapitelstruktur:
                """
                for chapter in validated_chapters.get("Chapters", []):
                    prompt += f"- {chapter['Title']}\n"

                llm = OllamaLLM()
                response = llm._call(prompt).strip()
                score = int(response)
                return min(max(score, 0), 25)
            except Exception as e:
                logger.error(f"Fehler bei der Bewertung der Cliffhanger: {e}")
                return 0####### !!!
            
def evaluate_tone(validated_chapters):
    """
    Bewertet die Tonalität des Schreibstils mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Tonalität des Schreibstils in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Tonalität: {e}")
        return 0
    
def evaluate_variety(validated_chapters):
    """
    Bewertet die Abwechslung im Schreibstil mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Abwechslung im Schreibstil der Kapitel.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Abwechslung: {e}")
        return 0
    
def evaluate_imagery(validated_chapters):
    """
    Bewertet die Verwendung von Sprachbildern mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Verwendung von Sprachbildern in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Sprachbilder: {e}")
        return 0
    
def evaluate_authenticity(validated_chapters):
    """
    Bewertet die Authentizität des Schreibstils mithilfe der KI.
    Rückgabe: Punktzahl zwischen 0 und 25.
    """
    try:
        prompt = """
        Überprüfe die Authentizität des Schreibstils in den Kapiteln.
        Gib eine Bewertung auf einer Skala von 0 bis 20 ab.
        Kapitelstruktur:
        """
        for chapter in validated_chapters.get("Chapters", []):
            prompt += f"- {chapter['Title']}\n"

        llm = OllamaLLM()
        response = llm._call(prompt).strip()
        score = int(response)
        return min(max(score, 0), 20)
    except Exception as e:
        logger.error(f"Fehler bei der Bewertung der Authentizität: {e}")
        return 0
    

# Buchbewertung 

def map_score_to_grade(score):
    """
    Wandelt eine Punktzahl (0-100) basierend auf der Notentabelle in eine Note um.
    """
    try:
        if 95 <= score <= 100:
            return "1+"
        elif 90 <= score < 95:
            return "1"
        elif 85 <= score < 90:
            return "1-"
        elif 80 <= score < 85:
            return "2+"
        elif 75 <= score < 80:
            return "2"
        elif 70 <= score < 75:
            return "2-"
        elif 65 <= score < 70:
            return "3+"
        elif 60 <= score < 65:
            return "3"
        elif 55 <= score < 60:
            return "3-"
        elif 50 <= score < 55:
            return "4+"
        elif 45 <= score < 50:
            return "4"
        elif 40 <= score < 45:
            return "4-"
        elif 33 <= score < 40:
            return "5+"
        elif 27 <= score < 33:
            return "5"
        elif 20 <= score < 27:
            return "5-"
        else:
            return "6"
    except Exception as e:
        logger.error(f"Fehler bei der Notenberechnung: {e}")
        return "Fehler"

# Final Score Calculation
def calculate_final_score(weighted_scores):
    """
    Berechnet die Endnote basierend auf den gewichteten Bewertungen.
    """
    try:
        total_score = sum(weighted_scores)
        average_score = total_score / len(weighted_scores)  # Durchschnitt berechnen
        final_grade = map_score_to_grade(average_score)  # Note anhand der Tabelle ermitteln
        return {"score": round(average_score, 2), "grade": final_grade}
    except Exception as e:
        logger.error(f"Fehler bei der Berechnung der Endnote: {e}")
        return {"score": 0, "grade": "Fehler"}




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
        min_chapter = data.get("min_chapter", 0)  # min_chapter erfassen
        min_subchapter = data.get("min_subchapter", 0)  # min_subchapter erfassen
        logger.info(f"Received request: user_input={user_input}, min_chapter={min_chapter}")
        
        # min_chapter an run_agents übergeben
        result = agent_system.run_agents(user_input, min_chapter=min_chapter, min_subchapter=min_subchapter)
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
