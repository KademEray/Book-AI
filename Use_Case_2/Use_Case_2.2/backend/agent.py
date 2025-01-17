from datetime import datetime
import json
import logging
import os
import re
import uuid

from chromadb import PersistentClient
import requests

from ollama import OllamaLLM


logging.basicConfig(
    filename='Use_Case_2/Use_Case_2.2/backend/backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Initialisiere Chroma mit persistentem Speicher
client = PersistentClient(path="./Use_Case_2/Use_Case_2.2/backend/chroma_storage")

# Erstelle oder erhalte eine Sammlung (Collection)
vectorstore = client.get_or_create_collection(
    name="conversation_context"
)
logger = logging.getLogger(__name__)

class AgentSystem:
    def __init__(self):
        """
        Initializes a new instance of the agent class.

        Attributes:
            agents (list): A list to store agent instances.
            session_id (str): A unique identifier for the session, generated using UUID.
        """
        self.agents = []
        self.session_id = str(uuid.uuid4())  # Eindeutige ID für die Sitzung

    def add_agent(self, agent_function, kontrolliert=False):
        """
        Adds an agent to the list of agents.

        Args:
            agent_function (function): The function that defines the agent's behavior.
            kontrolliert (bool, optional): A flag indicating whether the agent is controlled. Defaults to False.

        Returns:
            None
        """
        self.agents.append({"function": agent_function, "kontrolliert": kontrolliert})

    def run_agents(self, user_input, min_chapter=0, min_subchapter=0):
        """
        Executes a series of agents to process the user input and generate a final book evaluation.
        Args:
            user_input (str): The input provided by the user.
            min_chapter (int, optional): Minimum number of chapters to generate. Defaults to 0.
            min_subchapter (int, optional): Minimum number of subchapters to generate. Defaults to 0.
        Returns:
            dict: A dictionary containing the final grade and detailed results of the book evaluation.
        Raises:
            ValueError: If the terminal output is incomplete.
        Steps:
            1. Decision Agent: Determines if an internet search for the synopsis is required.
            2. Synopsis Agent: Generates and validates the synopsis.
            3. Chapter Structure Agent: Generates and validates the chapter structure.
            4. Writing Agent: Writes the chapters based on the validated structure.
            5. Summary Generation and Validation: Creates and validates the summary of the final text.
            6. Book Evaluation: Evaluates the book based on various criteria and calculates the final grade.
        Logs:
            - Logs various debug, info, and error messages throughout the process.
            - Stores intermediate and final results in the context.
        """
        logger.debug(f"run_agents called with user_input: {user_input}, min_chapter: {min_chapter}")
        response_data = {"steps": [], "final_response": ""}
        context = self.get_context()

        # Entscheidung vor der Synopsis-Agent
        logger.debug("Entscheide, ob eine Internetsuche für die Synopsis erforderlich ist...")
        decision_result = decision_agent(context, user_input, task_type="Synopsis")
        response_data["steps"].append(decision_result["log"])

        if decision_result["output"] == "Ja":
            logger.info("Internetsuche wurde bereits durchgeführt. Ergebnisse werden verwendet.")
        else:
            logger.info("Keine Internetsuche erforderlich.")

        # Schritt 2: Synopsis-Agent
        validated_synopsis = None
        while not validated_synopsis:
            logger.debug("Starting synopsis_agent...")
            synopsis_result = synopsis_agent(user_input, context)
            logger.debug(f"synopsis_agent result: {synopsis_result}")

            # Loggen der Ergebnisse des synopsis_agent
            response_data["steps"].append(synopsis_result["log"])

            if synopsis_result["log"]["status"] == "completed":
                logger.debug("Validiere die Synopsis...")
                validation_result = synopsis_validation_agent(user_input, synopsis_result["output"])

                if validation_result["log"]["status"] == "completed":
                    validated_synopsis = synopsis_result["output"]
                    # Speichere die validierte Synopsis
                    synopsis_id = self.store_context("Synopsis", validated_synopsis)
                    logger.info(f"Validierte Synopsis gespeichert mit ID: {synopsis_id}")
                else:
                    logger.error(f"Validierung fehlgeschlagen: {validation_result['log']['output']}. Wiederhole...")
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
        
        while True:
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
                    self.store_context("Final Text", final_text)
                else:
                    logger.error("Writing failed, retrying...")

            # Schritt 5: Erstellung und Validierung der Gesamtszusammenfassung
            summary = generate_summary(final_text=final_text)  # Gesamter Text wird übergeben
            logger.debug("Validiere Zusammenfassung...")
            validation_result = validate_summary(summary)

            if validation_result.get("Validated", False):
                logger.info("Zusammenfassung erfolgreich validiert.")
                self.store_context("Validated Summary", validation_result)
                break  # Beende die äußere Schleife, da alles erfolgreich abgeschlossen ist
            else:
                logger.warning("Zusammenfassung ist fehlerhaft. Wiederhole den gesamten Schreibprozess...")
                final_text = None  # Setze den final_text zurück, um den Schreibprozess erneut zu starten
                
        # Schritt 6: Buch bewerten
        logger.debug("Starte Buchbewertung...")
        weighted_scores_with_details = []

        agents = [
            evaluate_chapters,
            evaluate_paragraphs,
            evaluate_book_type,
            evaluate_content,
            evaluate_grammar,
            evaluate_style,
            evaluate_tension
        ]

        for agent in agents:
            result = agent(final_text)
            weighted_scores_with_details.append(result)

        # Extrahiere die Scores und Details
        scores = [entry["output"] for entry in weighted_scores_with_details]
        details = [entry["log"] for entry in weighted_scores_with_details]

        # Endnote berechnen
        final_evaluation = calculate_final_score(scores)
        logger.info(f"Buchbewertung abgeschlossen: {final_evaluation}")

        # **Kombiniere die finalen Ergebnisse**
        response_data["evaluation"] = {
            "final_grade": final_evaluation,
            "detailed_results": details,
            "final_text": final_text  # Finaler Text enthält bereits Quellen und Titel
        }

        save_evaluation_to_txt(response_data)
        logger.info("Buchbewertung erfolgreich gespeichert.")

        terminal_output = {
            "final_grade": final_evaluation,
            "detailed_results": details
        }

        # Validierung vor Rückgabe
        if not terminal_output.get("final_grade") or not terminal_output.get("detailed_results"):
            logger.error(f"Fehler in terminal_output: {terminal_output}")
            raise ValueError("terminal_output ist unvollständig.")

        logger.debug(f"All agents completed. Returning terminal_output: {terminal_output}")
        return terminal_output



    def get_next_document_id(self):
        """
        Determines the next document ID based on the number of stored documents.

        This method retrieves all stored data from the vector store and calculates
        the next document ID by counting the existing IDs. If an error occurs during
        the retrieval process, it logs the error and returns "1" as a fallback ID.

        Returns:
            str: The next document ID as a string.
        """
        """Ermittelt die nächste ID basierend auf der Anzahl der gespeicherten Dokumente."""
        try:
            all_data = vectorstore.get()  # Alle Daten abrufen
            existing_ids = all_data.get("ids", [])
            return str(len(existing_ids) + 1)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der nächsten ID: {e}")
            return "1"  # Fallback auf ID "1", falls ein Fehler auftritt

    def store_context(self, label, data):
        """
        Stores the context in ChromaDB with a sequential ID.
        Args:
            label (str): The label associated with the context.
            data (str): The context data to be stored.
        Raises:
            Exception: If there is an error while storing the context.
        Logs:
            Info: Logs the label and data being stored along with the document ID.
            Debug: Logs a message indicating successful storage.
            Error: Logs any error that occurs during storage.
        """
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
        """
        Retrieves the stored context from ChromaDB.

        This method fetches all documents from the collection in the vector store.
        If successful, it returns the documents as a single string, with each document
        separated by a newline character. If an error occurs during retrieval, it logs
        the error and returns a default context message.

        Returns:
            str: The concatenated string of all documents or a default context message
                 if an error occurs.
        """
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
        """
        Retrieves all documents from the vector store collection.

        This method attempts to fetch all documents stored in the vector store.
        If no documents are found, it logs a warning and returns a message indicating
        that no documents are available. If documents are successfully retrieved,
        it logs the number of documents and returns them as a single string, with each
        document separated by a newline character.

        Returns:
            str: A string containing all retrieved documents separated by newlines,
                 or a message indicating that no documents are available.

        Raises:
            Exception: If an error occurs during the retrieval process, it logs the error
                       and returns a default context message indicating no documents were found.
        """
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
        """
        Validates if the saved data is directly retrievable.

        This method attempts to retrieve documents from the vector store
        using the session ID stored in the instance. It logs the process
        of validation and handles any exceptions that may occur.

        Returns:
            list: A list of documents if retrieval is successful, otherwise an empty list.

        Raises:
            Exception: If there is an error during the validation process, it logs the error and returns an empty list.
        """
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

def sanitize_filename(filename):
    """
    Entfernt ungültige Zeichen aus einem Dateinamen.
    """
    """
    Sanitize the given filename by removing any invalid characters.
    This function removes characters that are not allowed in filenames on most
    operating systems, such as <, >, :, ", /, \, |, ?, and *. It also trims any
    leading or trailing whitespace from the filename.
    Args:
        filename (str): The filename to be sanitized.
    Returns:
        str: The sanitized filename with invalid characters removed and whitespace trimmed.
    """
    return re.sub(r'[<>:"/\\|?*]', '', filename).strip()

def get_next_book_name(output_dir):
    """
    Findet den nächsten verfügbaren Buchnamen im Format 'book_x', wobei x eine fortlaufende Zahl ist.
    """
    """
    Generates the name for the next book file in the specified output directory.

    This function scans the given directory for files matching the pattern 'book_<number>.txt',
    extracts the numbers from these filenames, and returns the name for the next book file
    with an incremented number.

    Args:
        output_dir (str): The directory where the book files are stored.

    Returns:
        str: The name for the next book file in the format 'book_<next_number>.txt'.
    """
    existing_files = os.listdir(output_dir)
    book_numbers = [
        int(re.search(r'book_(\d+)', file).group(1))
        for file in existing_files if re.match(r'book_\d+\.txt', file)
    ]
    next_number = max(book_numbers, default=0) + 1
    return f"book_{next_number}"

def save_evaluation_to_txt(response_data):
    """
    Speichert die Buchbewertung in einer strukturierten .txt-Datei im Ordner "Ergebnisse".
    """
    """
    Saves the evaluation data to a text file.
    This function creates a directory if it does not exist, determines the next book name,
    validates the structure of the response_data, and writes the evaluation details to a text file.
    Args:
        response_data (dict): A dictionary containing the evaluation data. It must have the following structure:
            {
                "evaluation": {
                    "final_grade": {
                        "score": float,
                        "grade": str
                    },
                    "detailed_results": [
                        {
                            "agent": str,
                            "explanation": str,
                            "output": str
                        },
                        ...
                    ],
                    "final_text": {
                        "Chapters": [
                            {
                                "Number": int,
                                "Title": str,
                                "Subchapters": [
                                    {
                                        "Number": int,
                                        "Title": str,
                                        "Content": str
                                    },
                                    ...
                                ]
                            },
                            ...
                        ]
                    }
                }
            }
    Raises:
        ValueError: If the required keys are missing in the response_data.
        Exception: If there is an error during the file writing process.
    """
    try:
        # Ordner erstellen, falls nicht vorhanden
        output_dir = "Use_Case_2/Use_Case_2.2/Ergebnisse"
        os.makedirs(output_dir, exist_ok=True)

        # Buchname bestimmen
        book_name = get_next_book_name(output_dir)

        # Datei-Pfad
        output_file = os.path.join(output_dir, f"{book_name}.txt")

        # Validierung der response_data Struktur
        if not response_data.get("evaluation"):
            raise ValueError("Fehlender 'evaluation'-Schlüssel in response_data.")

        if not response_data["evaluation"].get("final_grade"):
            raise ValueError("Fehlender 'final_grade'-Schlüssel in response_data['evaluation'].")

        if not response_data["evaluation"].get("detailed_results"):
            raise ValueError("Fehlender 'detailed_results'-Schlüssel in response_data['evaluation'].")

        if not response_data["evaluation"].get("final_text"):
            raise ValueError("Fehlender 'final_text'-Schlüssel in response_data['evaluation'].")

        with open(output_file, "w", encoding="utf-8") as f:
            # Final Grade
            final_grade = response_data["evaluation"]["final_grade"]
            f.write("Finale Bewertung:\n")
            f.write(f"Score: {final_grade['score']}\n")
            f.write(f"Grade: {final_grade['grade']}\n\n")

            # Detailed Results
            f.write("Detaillierte Ergebnisse:\n")
            for result in response_data["evaluation"]["detailed_results"]:
                f.write(f"Agent: {result['agent']}\n")
                f.write(f"Erklärung: {result['explanation']}\n")
                f.write(f"Bewertung: {result['output']}\n\n")

            # Inhaltsverzeichnis
            final_text = response_data["evaluation"]["final_text"]
            f.write("Inhaltsverzeichnis:\n")
            for chapter in final_text["Chapters"]:
                f.write(f"Kapitel {chapter['Number']}: {chapter['Title']}\n")
                for subchapter in chapter["Subchapters"]:
                    f.write(f"  {subchapter['Number']}: {subchapter['Title']}\n")
            f.write("\n")

            # Finaler Text
            f.write("Finaler Text:\n")
            for chapter in final_text["Chapters"]:
                f.write(f"Kapitel {chapter['Number']}: {chapter['Title']}\n")
                for subchapter in chapter["Subchapters"]:
                    f.write(f"  {subchapter['Number']}: {subchapter['Title']}\n")
                    f.write(f"  Inhalt:\n{subchapter['Content']}\n\n")

        logger.debug(f"Die Buchbewertung wurde erfolgreich in {output_file} gespeichert.")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Buchbewertung: {e}")
        raise

def decision_agent(context, input_text, task_type="Synopsis"):
    """
    Entscheidet, ob eine Internetsuche erforderlich ist und führt bei Bedarf eine Suchanfrage aus.

    :param context: Der aktuelle Kontext.
    :param input_text: Der Benutzerinput oder aktuelle Textabschnitt.
    :param task_type: Art der Aufgabe ("Synopsis", "Unterkapitel").
    :return: Ein Dictionary mit Log und Ergebnis ("Ja" oder "Nein") sowie optional der Suchanfrage.
    """
    log = {"agent": "DecisionAgent", "status": "running", "details": []}

    try:
        # Unterschiedliche Anweisungen basierend auf task_type
        task_instruction = {
            "Synopsis": "Entscheide, ob eine Internetsuche notwendig ist, um eine prägnante Synopsis zu erstellen.",
            "Unterkapitel": "Entscheide, ob eine Internetsuche notwendig ist, um präzise Unterkapitel zu erstellen."
        }.get(task_type, "Entscheide, ob eine Internetsuche notwendig ist.")

        prompt = f"""
        Kontext:
        {context}

        Aufgabe:
        {task_instruction}

        Eingabe:
        {input_text}

        Antworte ausschließlich mit "Ja" oder "Nein". Gebe keine zusätzlichen Erklärungen, Kommentare oder andere Informationen zurück.
        """

        logger.debug(f"Prompt für DecisionAgent (Task-Typ: {task_type}):\n{prompt}")
        
        llm = OllamaLLM()
        response = llm._call(prompt).strip().rstrip('.').lower()  # Bereinige die Antwort
        logger.debug(f"Antwort von LLM für DecisionAgent: {response}")

        if response not in ["ja", "nein"]:
            logger.warning("Unerwartete Antwort vom LLM, Standardantwort 'Nein' wird verwendet.")
            response = "nein"
        
        if response == "ja":
            log.update({"status": "completed", "output": "Ja"})

            # Aufruf des SearchQueryAgent
            search_query_result = SearchQueryAgent(input_text=input_text, context=context, task_type=task_type)
            log["details"].append(search_query_result["log"])

            return {
                "log": log,
                "output": "Ja",
                "search_query": search_query_result.get("search_query", "Keine Suchanfrage generiert."),
            }
        elif response == "nein":
            log.update({"status": "completed", "output": "Nein"})
            return {"log": log, "output": "Nein"}
        else:
            logger.warning("Unerwartete Antwort vom LLM, Standardantwort 'Nein' wird verwendet.")
            log.update({"status": "failed", "output": "Nein"})
            return {"log": log, "output": "Nein"}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        logger.error(f"Fehler im DecisionAgent: {e}")
        return {"log": log, "output": "Fehler"}

def SearchQueryAgent(input_text, context, task_type="Synopsis"):
    """
    Erstellt eine präzise Suchanfrage basierend auf dem Benutzerinput und validiert den Output iterativ.
    Sendet die validierte Suchanfrage an den /api/search-Endpunkt und speichert die Ergebnisse in ChromaDB.

    :param user_input: Der Benutzerinput oder aktuelle Textabschnitt.
    :param task_type: Art der Aufgabe ("Synopsis", "Unterkapitel").
    :return: Ein Dictionary mit Log und den Ergebnissen des Suchendpunkts.
    """
    log = {"agent": "SearchQueryAgent", "status": "running", "details": []}
    validated_search_query = None
    iteration = 1
    corrections = None

    while not validated_search_query:
        logger.info(f"Iteration {iteration}: Generierung und Validierung der Suchanfrage...")

        try:
            if isinstance(context, dict):
                context = json.dumps(context, indent=2)
            if isinstance(input_text, dict):
                input_text = json.dumps(input_text, indent=2)
            # Unterschiedliche Anweisungen basierend auf task_type
            task_instruction = {
                "Synopsis": "Erstelle eine kurze und prägnante Suchanfrage mit maximal 10 Wörtern, die die Schlüsselbegriffe enthält, um präzise Informationen zu einer Synopsis zu finden.",
                "Unterkapitel": "Erstelle eine prägnante Suchanfrage mit maximal 10 Wörtern, die relevante Informationen zu einem Unterkapitel liefert."
            }.get(task_type, "Erstelle eine allgemeine Suchanfrage basierend auf dem Benutzerinput.")

            # Basisprompt für die KI
            prompt = f"""
            Kontext:
            {context}

            Benutzerinput:
            {input_text}

            Aufgabe:
            {task_instruction}

            Die Suchanfrage sollte direkt nutzbar sein und ausschließlich relevante Schlüsselbegriffe enthalten.
            """

            # Falls Korrekturhinweise vorliegen, erweitere den Prompt
            if corrections:
                prompt += f"\n\nHinweis zur Verbesserung:\n{corrections}"

            logger.debug(f"Prompt für SearchQueryAgent:\n{prompt}")

            # Aufruf des LLM
            llm = OllamaLLM()
            search_query = llm._call(prompt).strip()

            # Bereinigen der Suchanfrage
            if "`" in search_query:  # Falls die Anfrage ein Backtick enthält
                search_query = re.search(r"`(.*?)`", search_query).group(1)  # Extrahiere nur den Inhalt innerhalb der Backticks

            logger.debug(f"Bereinigte Suchanfrage: {search_query}")

            log["details"].append({"iteration": iteration, "generated_query": search_query})

            # Validierungs-Agent aufrufen
            validation_result = validate_search_query(input_text=input_text, context=context, search_query=search_query)
            log["details"].append(validation_result["log"])

            if validation_result["log"]["status"] == "completed":
                validated_search_query = search_query
                log.update({"status": "completed", "output": validated_search_query})
                logger.info(f"Validierte Suchanfrage: {validated_search_query}")
            else:
                corrections = validation_result["log"]["output"].get("corrections", "Keine Korrekturvorschläge erhalten.")
                logger.warning(f"Validierung fehlgeschlagen: {validation_result['log']['output']}. Wiederhole...")
                iteration += 1

        except Exception as e:
            log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
            logger.error(f"Fehler im SearchQueryAgent: {e}")
            return {"log": log, "search_query": f"Fehler: {str(e)}"}

    # Nach erfolgreicher Validierung: Anfrage an den Endpunkt senden
    try:
        logger.info("Sende validierte Suchanfrage an den /api/search-Endpunkt...")
        api_url = "http://127.0.0.1:5000/api/search"  # Lokaler Endpunkt
        logger.debug(f"Validierte Suchanfrage: {validated_search_query}")
        response = requests.post(api_url, json={"request_input": validated_search_query}, timeout=10)
        logger.debug(f"Antwort vom /api/search-Endpunkt: {response}")
        response.raise_for_status()

        search_results = response.json()
        log.update({"status": "completed", "search_results": search_results})
        logger.info(f"Suchergebnisse erhalten: {search_results}")

        # Ergebnisse in ChromaDB speichern
        label = "Search Results"
        data = {
            "query": validated_search_query,
            "results": search_results
        }
        agent_system.store_context(label, data)  # Nutzung der bestehenden Funktion
        logger.info("Suchergebnisse erfolgreich in ChromaDB gespeichert.")

        return {"log": log, "search_results": search_results}
    except requests.exceptions.RequestException as e:
        log.update({"status": "failed", "output": f"Fehler beim Senden der Suchanfrage: {str(e)}"})
        logger.error(f"Fehler beim Senden der Suchanfrage: {e}")
        return {"log": log, "search_results": f"Fehler: {str(e)}"}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler beim Speichern in ChromaDB: {str(e)}"})
        logger.error(f"Fehler beim Speichern in ChromaDB: {e}")
        return {"log": log, "search_results": f"Fehler: {str(e)}"}

def validate_search_query(input_text, context, search_query):
    """
    Validiert die generierte Suchanfrage und gibt bei Fehlern Korrekturhinweise.

    :param user_input: Der Benutzerinput oder aktuelle Textabschnitt.
    :param search_query: Die generierte Suchanfrage.
    :return: Ein Dictionary mit Log und dem Validierungsergebnis.
    """
    log = {"agent": "ValidationAgent", "status": "running", "details": []}
    try:
        # Prompt für die Validierung
        prompt = f"""
        Kontext:
        {context}

        Benutzerinput:
        {input_text}

        Generierte Suchanfrage:
        {search_query}

        Aufgabe:
        Überprüfe, ob die generierte Suchanfrage prägnant, relevant und klar formuliert ist.
        Antworte mit:
        - "Ja" am Anfang, wenn die Suchanfrage korrekt ist, gefolgt von einer kurzen Begründung.
        - "Nein" am Anfang, wenn die Suchanfrage nicht korrekt ist, gefolgt von einer detaillierten Begründung und Verbesserungsvorschlägen.
        """
        logger.debug(f"Prompt für ValidationAgent:\n{prompt}")

        # Aufruf des LLM
        llm = OllamaLLM()
        validation_response = llm._call(prompt).strip()
        logger.debug(f"Antwort von LLM zur Validierung: {validation_response}")

        if validation_response.startswith("Ja"):
            log.update({"status": "completed", "output": "Validierung erfolgreich."})
            logger.info("Suchanfrage erfolgreich validiert.")
        else:
            reason = validation_response.split("Begründung:", 1)[-1].strip() if "Begründung:" in validation_response else "Keine Begründung erhalten."
            corrections = validation_response.split("Verbesserungsvorschläge:", 1)[-1].strip() if "Verbesserungsvorschläge:" in validation_response else "Keine Verbesserungsvorschläge erhalten."
            log.update({
                "status": "failed",
                "output": {
                    "reason": reason,
                    "corrections": corrections
                }
            })
            logger.warning(f"Validierung fehlgeschlagen. Begründung: {reason}, Korrekturvorschläge: {corrections}")

    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        logger.error(f"Fehler im ValidationAgent: {e}")

    return {"log": log}

# Synopsis-Agent
def synopsis_agent(user_input, context):
    """
    Generates a concise synopsis based on the provided user input and context.
    Args:
        user_input (str): The input provided by the user that needs to be summarized.
        context (str): The context in which the user input should be interpreted.
    Returns:
        dict: A dictionary containing the log and the output. The log includes the agent name, status, and details.
              The output is either the generated synopsis or an error message if an exception occurred.
    Raises:
        Exception: If an error occurs during the generation of the synopsis.
    """
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
    
def synopsis_validation_agent(user_input, output):
    """
    Validates the given output as a synopsis based on the user input.
    This function uses a language model to check if the provided output is a good synopsis
    for the given user input. It logs the process and returns a log dictionary with the 
    validation status and result.
    Args:
        user_input (str): The user's input or request.
        output (str): The output to be validated as a synopsis.
    Returns:
        dict: A dictionary containing the log with the validation status and result.
    """
    log = {"agent": "synopsis_validation_agent", "status": "processing"}
    try:
        logger.debug("[DEBUG] synopsis_validation_agent gestartet")
        logger.debug(f"[DEBUG] Benutzerinput: {user_input}")
        logger.debug(f"[DEBUG] Ausgabe zum Validieren: {output}")

        prompt = f"""
        Überprüfe die folgende Ausgabe darauf, ob sie als Synopsis gut ist.

        Benutzeranfrage:
        {user_input}

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
            logger.info("Validierung erfolgreich. Ausgabe ist inhaltlich korrekt.")
            log.update({
                "status": "completed",
                "output": "Validierung erfolgreich. Ausgabe ist inhaltlich korrekt."
            })
        else:
            reason = validation_result.split("Begründung:")[1].strip() if "Begründung:" in validation_result else "Unzureichende Begründung erhalten."
            logger.warning(f"Validierung fehlgeschlagen: {reason}")
            log.update({
                "status": "failed",
                "output": f"Validierung fehlgeschlagen: {reason}"
            })

    except Exception as e:
        log.update({
            "status": "failed",
            "output": f"Fehler bei der Validierung: {str(e)}"
        })
        logger.error(f"[DEBUG] Fehler in synopsis_validation_agent: {e}")

    return {"log": log}

# Validierungs-Agent
def validation_agent(user_input, output):
    """
    Validates the given user input and output using a series of validation agents.
    Args:
        user_input (Any): The input provided by the user that needs to be validated.
        output (Any): The output that needs to be validated.
    Returns:
        dict: A dictionary containing the log of the validation process with the following keys:
            - "status" (str): The status of the validation process, either "completed" or "failed".
            - "output" (str): A message describing the result of the validation process.
    Raises:
        Exception: If an error occurs during the validation process, it is caught and logged.
    """
    agents = [
        validation_agent_content,
        validation_agent_logic
    ]
    
    for agent in agents:
        try:
            result = agent(user_input, output)
            if result["log"].get("status") != "completed":
                logger.error(f"[ERROR] Validierung fehlgeschlagen bei {agent.__name__}: {result['log']['output']}")
                return {"log": {"status": "failed", "output": f"Validierung fehlgeschlagen bei {agent.__name__}"}}  # Abbruch
        except Exception as e:
            logger.error(f"[DEBUG] Fehler in Validierungsagent {agent.__name__}: {e}")
            return {"log": {"status": "failed", "output": f"Fehler in Validierungsagent {agent.__name__}: {str(e)}"}}  # Abbruch

    # Alle Validierungen bestanden
    logger.info("[INFO] Alle Validierungen erfolgreich abgeschlossen.")
    return {"log": {"status": "completed", "output": "Alle Validierungen erfolgreich abgeschlossen."}}

## 1. Inhaltlicher Validierungsagent (Original-Agent erweitert)
def validation_agent_content(user_input, output):
    """
    Validates the given output against the user input and context using a language model.
    Args:
        user_input (str): The input provided by the user.
        output (str): The output to be validated.
    Returns:
        dict: A dictionary containing the log with the validation status and output message.
    The log dictionary contains:
        - agent (str): The name of the agent ("ValidationAgent").
        - status (str): The status of the validation process ("processing", "completed", or "failed").
        - output (str): The result of the validation, including reasons for success or failure.
    The function performs the following steps:
        1. Logs the start of the validation process and the provided inputs.
        2. Retrieves the context from the agent system.
        3. Constructs a prompt for the language model to validate the output.
        4. Calls the language model with the constructed prompt.
        5. Parses the validation result and updates the log accordingly.
        6. Handles any exceptions that occur during the process and updates the log with error information.
    """
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

        Sehr wichtig Antworte immer am Anfang mit:
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

## 2. Logischer Validierungsagent
def validation_agent_logic(user_input, output):
    """
    Validates the logical consistency of the given output using a language model.
    Args:
        user_input (str): The input provided by the user.
        output (str): The output to be validated for logical consistency.
    Returns:
        dict: A dictionary containing the log of the validation process with the following keys:
            - "agent" (str): The name of the agent performing the validation.
            - "status" (str): The status of the validation process ("processing", "completed", or "failed").
            - "output" (str): The result of the validation, including any error messages or reasons for failure.
    """
    log = {"agent": "LogicValidationAgent", "status": "processing"}

    try:
        logger.debug("[DEBUG] LogicValidationAgent gestartet")
        logger.debug(f"[DEBUG] Ausgabe zum Validieren: {output}")

        prompt = f"""
        Überprüfe die folgende Ausgabe darauf, ob sie logisch konsistent ist, unabhängig vom Kontext. Achte darauf, ob Widersprüche, unsinnige Aussagen oder logische Fehler enthalten sind.

        Ausgabe:
        {output}

        Sehr wichtig Antworte immer am Anfang mit:
        1. "Ja" oder "Nein", ob die Ausgabe inhaltlich korrekt ist.
        2. Begründung, warum die Ausgabe korrekt oder falsch ist.
        """
        llm = OllamaLLM()
        validation_result = llm._call(prompt)
        logger.debug(f"[DEBUG] Validierungsergebnis von LLM: {validation_result}")

        if "Ja" in validation_result:
            logger.info("Validierung erfolgreich. Ausgabe ist inhaltlich korrekt.")
            log.update({
                "status": "completed",
                "output": "Validierung erfolgreich. Ausgabe ist logisch konsistent."
            })
        else:
            reason = validation_result.split("Begründung:")[1].strip() if "Begründung:" in validation_result else "Unzureichende Begründung erhalten."
            logger.warning(f"Validierung fehlgeschlagen: {reason}")
            log.update({
                "status": "failed",
                "output": f"Validierung fehlgeschlagen: {reason}"
            })

    except Exception as e:
        log.update({
            "status": "failed",
            "output": f"Fehler bei der Validierung: {str(e)}"
        })
        logger.error(f"[DEBUG] Fehler in LogicValidationAgent: {e}")

    return {"log": log}


def chapter_agent(min_chapter=0, min_subchapter=0):
    """
    Generates and validates chapters and subchapters for a document based on stored contexts.
    Args:
        min_chapter (int, optional): Minimum number of chapters to generate. Defaults to 0.
        min_subchapter (int, optional): Minimum number of subchapters to generate for each chapter. Defaults to 0.
    Returns:
        dict: A dictionary containing the log and the output. The log includes the status and details of the generation process. 
              The output contains the validated chapter dictionary if the process is successful, or an error message if it fails.
    Raises:
        ValueError: If no context is available, no chapters are generated, or subchapter generation fails.
    """
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
    """
    Generates and validates a list of subchapters for a given chapter using a language model.
    Args:
        current_chapter (dict): A dictionary containing the current chapter's details, including "Number" and "Title".
        min_subchapter (int): The minimum number of subchapters required.
    Returns:
        dict: A dictionary containing the log of the operation and the updated chapter with the generated subchapters.
    Raises:
        Exception: If there is an error during the generation or validation of subchapters.
    The function performs the following steps:
    1. Initializes a log dictionary to track the status and details of the operation.
    2. Uses a language model (OllamaLLM) to generate a list of subchapters based on the given chapter's title and number.
    3. Validates the generated subchapters to ensure they meet the specified requirements.
    4. If the generated subchapters do not meet the minimum required number or have structural issues, the function attempts to correct them.
    5. Adds the validated subchapters to the current chapter's "Subchapters" list.
    6. Updates the log with the status of the operation and returns the log along with the updated chapter.
    Note:
        The function logs detailed debug and error messages to help with troubleshooting and validation.
    """
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
    """
    Extracts a list of chapters or subchapters from a given response string.
    Args:
        response (str): The response string containing chapters or subchapters.
        is_subchapter (bool, optional): Flag indicating whether to extract subchapters. Defaults to False.
        chapter_number (int, optional): The chapter number to extract subchapters from. Required if is_subchapter is True.
    Returns:
        list: A list of extracted chapters or subchapters.
    Raises:
        ValueError: If no valid chapters or subchapters are found or if an error occurs during extraction.
    """
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
    """
    Validates the structure of a list of chapter or subchapter titles.
    Args:
        item_list (list): A list of strings representing chapter or subchapter titles.
        is_subchapter (bool): A flag indicating whether the items are subchapters. 
                              If True, the items are treated as subchapters. 
                              If False, the items are treated as chapters.
    Returns:
        bool: True if all items in the list match the expected pattern, False otherwise.
    Logs:
        Logs an error message if the list is empty or if any item does not match the expected pattern.
        Logs an info message if all items are valid.
        Logs an error message if an exception occurs during validation.
    """
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
    """
    Builds a dictionary representation of chapters and subchapters from a list of strings.
    Args:
        chapter_list (list of str): A list of strings where each string represents either a chapter or a subchapter.
            - Chapter format: "Kapitel <number>: <title>"
            - Subchapter format: "Subchapter <chapter_number>.<subchapter_number>: <title>"
    Returns:
        dict: A dictionary with the structure:
            {
                "Chapters": [
                    {
                        "Number": <chapter_number>,
                        "Title": <chapter_title>,
                        "Subchapters": [
                            {
                                "Number": <subchapter_number>,
                                "Title": <subchapter_title>
                            },
                            ...
                        ]
                    },
                    ...
                ]
    Raises:
        ValueError: If the chapter list is empty or no valid chapters are found.
        Exception: For any other errors encountered during processing.
    """
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
    """
    Validates the structure and content of chapters and their subchapters.
    Args:
        chapter_dict (dict): A dictionary containing the chapters and their details.
        **kwargs: Additional keyword arguments.
    Returns:
        dict: A log dictionary containing the status of the validation, output, and details.
    The function performs the following steps:
    1. Retrieves the context for chapter validation.
    2. Converts the chapter structure into text format for validation.
    3. Sends a validation request to a language model (LLM) for the overall chapter structure.
    4. If the chapter structure is invalid, logs the reason and correction suggestions.
    5. Validates the subchapters of each chapter if they exist.
    6. If any subchapter structure is invalid, logs the reason and correction suggestions.
    7. If all chapters and subchapters are valid, logs the success and stores the context.
    The function handles ValueError and general exceptions, logging appropriate error messages and updating the log dictionary.
    """
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
    """
    Processes validated chapters and generates content for each subchapter using a language model.
    Args:
        user_input (str): The input provided by the user.
        validated_chapters (dict): A dictionary containing validated chapters with their titles and subchapters.
    Returns:
        dict: A dictionary containing the log of the process and the final generated text for the chapters.
    The function performs the following steps:
    1. Initializes a log dictionary to track the status and details of the process.
    2. Iterates through each chapter in the validated chapters.
    3. For each chapter, iterates through its subchapters and generates content using a language model.
    4. Validates the generated content for each subchapter.
    5. If validation is successful, stores the content and updates the context.
    6. If any error occurs during the process, logs the error and updates the status to failed.
    7. Returns the log and the final generated text for the chapters.
    Raises:
        ValueError: If no chapter could be successfully processed.
    """
    log = {"agent": "WritingAgent", "status": "running", "details": []}
    final_text = {"Chapters": []}

    try:
        logger.debug("WritingAgent gestartet")
        logger.debug(f"[DEBUG] validated_chapters: {validated_chapters}")

        for chapter in validated_chapters.get("Chapters", []):
            logger.debug(f"[DEBUG] Verarbeite Kapitel: {chapter['Title']}")

            chapter_content = {"Number": chapter["Number"], "Title": chapter["Title"], "Subchapters": []}

            for subchapter in chapter.get("Subchapters", []):
                subchapter_validated = False

                # Entscheidung vor dem Schreiben des Unterkapitels
                decision_result = decision_agent(
                    context=agent_system.get_context(),
                    input_text=subchapter["Title"],
                    task_type="Unterkapitel"
                )
                log["details"].append(decision_result["log"])

                if decision_result["output"] == "Ja":
                    logger.info(f"Internetsuche erforderlich für Unterkapitel: {subchapter['Title']}")
                    # Optionale Integration einer Internetsuche hier

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
    
def generate_summary(final_text):
    """Erstellt eine Zusammenfassung des gesamten Textes."""
    """
    Generates a summary of the provided text.
    This function takes a text input and generates a concise summary that covers the main points from each chapter in a logical order without omitting details.
    Args:
        final_text (str): The entire text to be summarized.
    Returns:
        dict: A dictionary containing the summary of the text. If an error occurs, the dictionary contains an error message.
    """
    try:
        prompt = f"""
        Aufgabe: Erstelle eine prägnante Zusammenfassung des gesamten Textes.
        Die Zusammenfassung sollte eine logische Reihenfolge haben und die Hauptpunkte aus jedem Kapitel abdecken, ohne Details auszulassen.

        Text:
        {final_text}
        """
        llm = OllamaLLM()
        summary = llm._call(prompt).strip()
        return {"Summary": summary}
    except Exception as e:
        logger.error(f"Fehler bei der Erstellung der Zusammenfassung: {e}")
        return {"Summary": f"Fehler: {str(e)}"}

def validate_summary(summary):
    """Validiert die erstellte Gesamtszusammenfassung."""
    """
    Validates the logical coherence of a given summary.
    This function uses a language model to check if the provided summary makes logical sense.
    It performs a rough validation to ensure the text has a correct structure, allowing for minor deviations.
    If the summary is deemed nonsensical, it returns a validation failure with a reason.
    Args:
        summary (dict): A dictionary containing the summary to be validated. 
                        The dictionary must have a key 'Summary' with the summary text as its value.
    Returns:
        dict: A dictionary containing the original summary and the validation result.
              If validated, the dictionary includes:
                - "Summary": The original summary text.
                - "Validated": True.
              If not validated, the dictionary includes:
                - "Summary": The original summary text.
                - "Validated": False.
                - "Reason": The reason for validation failure.
    Raises:
        Exception: If an error occurs during the validation process, it logs the error and returns a failure response.
    """
    try:
        prompt = f"""
        Zusammenfassung: {summary['Summary']}

        Aufgabe: Überprüfe, ob diese Zusammenfassung logisch sinn ergibt. Bedenke das ist nur eine Grobe Validierung um zu schauen ob der Text eine korrekte Struktur hat. Bedenke zu dem auch dass die Zusammenfassung nicht alle Chapters beinhaltet, es dient nur als eine ganz grobe Validierung deswegen sei nicht streng. Kleine Abweichungen sind erlaubt, nur wenn die Zusammenfassung unsinn ist dann Nein aber wenn es sinn macht dann Ja. Antworte mit nur "Ja" oder "Nein" 
        und gib eine Begründung, falls "Nein".
        """
        llm = OllamaLLM()
        validation_result = llm._call(prompt).strip()
        if validation_result.startswith("Ja"):
            return {"Summary": summary["Summary"], "Validated": True}
        else:
            logger.warning(f"Zusammenfassung nicht validiert: {validation_result}")
            # Speichere die fehlerhafte Zusammenfassung und die Validierungsantwort
            agent_system.store_context(
                "Failed Summary Validation",
                {
                    "Summary": summary["Summary"],
                    "Validation Feedback": validation_result,
                },
            )
            return {
                "Summary": summary["Summary"],
                "Validated": False,
                "Reason": validation_result,
            }
    except Exception as e:
        logger.error(f"Fehler bei der Validierung der Zusammenfassung: {e}")
        # Speichere die fehlerhafte Zusammenfassung und die Fehlermeldung
        agent_system.store_context(
            "Failed Summary Validation",
            {
                "Summary": summary["Summary"],
                "Error": f"Fehler: {str(e)}",
            },
        )
        return {
            "Summary": summary["Summary"],
            "Validated": False,
            "Reason": f"Fehler: {str(e)}",
        }

def evaluate_chapters(final_text):
    """
    Bewertet die Kapitel basierend auf Struktur, Konsistenz und Übergängen.
    """
    """
    Evaluates the chapters of a book based on their structure, consistency, and transitions.
    Args:
        final_text (str): The text of the book to be evaluated.
    Returns:
        dict: A dictionary containing the evaluation log, the output score, and the explanation.
            - log (dict): Contains the agent name, status, and details of the evaluation.
            - output (int): The evaluation score on a scale from 0 to 100.
            - explanation (str): The explanation for the given score and suggestions for improvement.
    Raises:
        ValueError: If no numerical score is found in the response.
    """
    log = {"agent": "ChapterEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte die Kapitel des Buches basierend auf ihrer Struktur, Konsistenz und den Übergängen.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und den Rest des Textes
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_paragraphs(final_text):
    """
    Bewertet die Absätze hinsichtlich Lesefluss, Fokus und Verknüpfung.
    """
    """
    Evaluates the paragraphs of a given text based on their readability, focus, and logical coherence.
    Args:
        final_text (str): The text to be evaluated.
    Returns:
        dict: A dictionary containing the evaluation log, the output score, and the explanation.
            - log (dict): Contains details about the evaluation process.
                - agent (str): The name of the agent performing the evaluation.
                - status (str): The status of the evaluation ('running', 'completed', or 'failed').
                - details (list): Additional details about the evaluation process.
                - output (int): The evaluation score (0-100).
                - explanation (str): Explanation of the score and suggestions for improvement.
            - output (int): The evaluation score (0-100).
            - explanation (str): Explanation of the score and suggestions for improvement.
    Raises:
        ValueError: If no numerical score is found in the response.
    """
    log = {"agent": "ParagraphEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte die Absätze des Buches basierend auf ihrem Lesefluss, Fokus und logischer Verknüpfung.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und den Rest des Textes
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_book_type(final_text):
    """
    Bewertet die Buchart in Bezug auf Zielgruppe und Thema.
    """
    """
    Evaluates the type of a book based on its suitability for the target audience and theme.
    Args:
        final_text (str): The text of the book to be evaluated.
    Returns:
        dict: A dictionary containing the log of the evaluation process, the output score, and the explanation.
            - log (dict): Contains details about the evaluation process.
                - agent (str): The name of the agent performing the evaluation.
                - status (str): The status of the evaluation ('running', 'completed', or 'failed').
                - details (list): Additional details about the evaluation process.
                - output (int): The evaluation score (0-100).
                - explanation (str): The explanation for the given score.
            - output (int): The evaluation score (0-100).
            - explanation (str): The explanation for the given score.
    Raises:
        ValueError: If no number is found in the response.
    """
    log = {"agent": "BookTypeEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte die Buchart basierend auf ihrer Eignung für Zielgruppe und Thema.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und die Erklärung
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_content(final_text):
    """
    Bewertet den Inhalt basierend auf Tiefe, Relevanz und Fokus auf das Thema.
    """
    """
    Evaluates the content of a book based on depth, relevance, and focus on the topic.
    Args:
        final_text (str): The text content of the book to be evaluated.
    Returns:
        dict: A dictionary containing the evaluation log, output score, and explanation.
            - log (dict): Contains details about the evaluation process, including agent name, status, and any errors.
            - output (int): The evaluation score on a scale from 0 to 100.
            - explanation (str): The explanation for the given score and suggestions for improvement.
    Raises:
        ValueError: If no numerical score is found in the response.
    """
    log = {"agent": "ContentEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte den Inhalt des Buches basierend auf Tiefe, Relevanz und Fokus auf das Thema.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und die Erklärung
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_grammar(final_text):
    """
    Bewertet Grammatik und Rechtschreibung.
    """
    """
    Evaluates the grammar and spelling of the provided text and returns a score along with an explanation.
    Args:
        final_text (str): The text to be evaluated.
    Returns:
        dict: A dictionary containing the log of the evaluation process, the output score (0-100), 
              and an explanation of the score. The dictionary has the following structure:
              {
                  "log": {
                      "agent": "GrammarEvaluationAgent",
                      "status": "running" | "completed" | "failed",
                      "details": [],
                      "output": int,  # The evaluation score (0-100)
                      "explanation": str  # Explanation of the score
                  },
                  "output": int,  # The evaluation score (0-100)
                  "explanation": str  # Explanation of the score
              }
    Raises:
        ValueError: If no numerical score is found in the response.
    """
    log = {"agent": "GrammarEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte die Grammatik und Rechtschreibung des Buches.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und die Erklärung
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_style(final_text):
    """
    Bewertet den Schreibstil hinsichtlich Abwechslung, Tonalität und Authentizität.
    """
    """
    Evaluates the writing style of a given text based on variety, tone, and authenticity.
    Args:
        final_text (str): The text to be evaluated.
    Returns:
        dict: A dictionary containing the evaluation log, the output score, and the explanation.
            - log (dict): Contains details about the evaluation process.
                - agent (str): The name of the agent performing the evaluation.
                - status (str): The status of the evaluation ('running', 'completed', 'failed').
                - details (list): Additional details about the evaluation process.
                - output (int): The evaluated score (0-100).
                - explanation (str): The explanation for the given score.
            - output (int): The evaluated score (0-100).
            - explanation (str): The explanation for the given score.
    Raises:
        ValueError: If no number is found in the response.
    """
    log = {"agent": "StyleEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte den Schreibstil des Buches basierend auf Abwechslung, Tonalität und Authentizität.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und die Erklärung
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}

def evaluate_tension(final_text):
    """
    Bewertet die Spannung des Buches basierend auf Wendepunkten, Aufbau und Charakterentwicklung.
    """
    """
    Evaluates the tension of a book based on turning points, structure, and character development.
    Args:
        final_text (str): The text of the book to be evaluated.
    Returns:
        dict: A dictionary containing the log of the evaluation process, the tension score (0-100), 
              and an explanation of the score. The dictionary has the following keys:
              - "log": A dictionary with the following keys:
                  - "agent": The name of the agent performing the evaluation.
                  - "status": The status of the evaluation process ("running", "completed", or "failed").
                  - "details": Additional details about the evaluation process.
                  - "output": The tension score (0-100).
                  - "explanation": The explanation for the given score.
                  - "error" (optional): The error message if the evaluation failed.
              - "output": The tension score (0-100).
              - "explanation": The explanation for the given score.
    Raises:
        ValueError: If no number is found in the response from the language model.
    """
    log = {"agent": "TensionEvaluationAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Aufgabe: Bewerte die Spannung des Buches basierend auf Wendepunkten, Aufbau und Charakterentwicklung.
        Text:
        {final_text}

        Gib eine Bewertung so streng wie mögliche auf einer Skala von 0 bis 100 ab. Ohne /100 sondern nur deine Bewertung.
        Erkläre, warum du diese Bewertung vergeben hast, und schlage Verbesserungen vor.
        """
        llm = OllamaLLM()
        response = llm._call(prompt).strip()

        # Extrahiere die erste Zahl und die Erklärung
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            explanation = response[match.end():].strip()
            log.update({
                "status": "completed",
                "output": min(max(score, 0), 100),
                "explanation": explanation
            })
        else:
            raise ValueError("Keine Zahl in der Antwort gefunden.")

        return {"log": log, "output": log["output"], "explanation": log["explanation"]}
    except Exception as e:
        log.update({"status": "failed", "output": 0, "error": str(e)})
        return {"log": log, "output": 0, "explanation": "Fehler bei der Bewertung"}


# Buchbewertung 

def map_score_to_grade(score):
    """
    Wandelt eine Punktzahl (0-100) basierend auf der Notentabelle in eine Note um.
    """
    """
    Maps a numerical score to a corresponding grade.
    The grading scale is as follows:
    - 95 <= score <= 100: "1+"
    - 90 <= score < 95: "1"
    - 85 <= score < 90: "1-"
    - 80 <= score < 85: "2+"
    - 75 <= score < 80: "2"
    - 70 <= score < 75: "2-"
    - 65 <= score < 70: "3+"
    - 60 <= score < 65: "3"
    - 55 <= score < 60: "3-"
    - 50 <= score < 55: "4+"
    - 45 <= score < 50: "4"
    - 40 <= score < 45: "4-"
    - 33 <= score < 40: "5+"
    - 27 <= score < 33: "5"
    - 20 <= score < 27: "5-"
    - score < 20: "6"
    Args:
        score (int): The numerical score to be converted to a grade.
    Returns:
        str: The corresponding grade as a string.
    Raises:
        Exception: If an error occurs during the grade calculation.
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
def calculate_final_score(weighted_scores_with_details):
    """
    Berechnet die Endnote basierend auf gewichteten Bewertungen.
    """
    """
    Calculates the final score based on weighted evaluations.
    Args:
        weighted_scores_with_details (list): A list of either numerical scores or dictionaries containing 
                                             'output' (score) and 'log' (details).
    Returns:
        dict: A dictionary containing:
            - 'score' (float): The weighted average score rounded to two decimal places.
            - 'grade' (str): The final grade determined by the weighted average score.
            - 'details' (list): An empty list (details are removed as they are redundant).
    Raises:
        Exception: If an error occurs during the calculation process.
    """
    try:
        # Gewichtungen für universelle Bewertung
        weights = [1.5, 1.5, 1, 2, 1.5, 1.5, 1]  # Kapitel, Absätze, Buchart, Inhalt, Grammatik, Stil, Spannung

        # Validierung der Eingabestruktur
        if all(isinstance(entry, (int, float)) for entry in weighted_scores_with_details):
            logger.info("Numerische Werte in weighted_scores_with_details erkannt.")
            # Konvertiere numerische Werte in das erwartete Format
            weighted_scores_with_details = [
                {"output": score, "log": {"agent": f"Agent {i + 1}", "explanation": "Keine Erklärung verfügbar"}}
                for i, score in enumerate(weighted_scores_with_details)
            ]
        elif not all(isinstance(entry, dict) and "output" in entry and "log" in entry for entry in weighted_scores_with_details):
            logger.error(f"Unerwartetes Format in weighted_scores_with_details: {weighted_scores_with_details}")
            return {"score": 0, "grade": "Fehler", "details": []}

        # Überprüfen, ob Gewichtungen und Ergebnisse übereinstimmen
        if len(weighted_scores_with_details) != len(weights):
            logger.warning(f"Anzahl der Ergebnisse ({len(weighted_scores_with_details)}) stimmt nicht mit der Anzahl der Gewichtungen ({len(weights)}) überein.")
            weights = weights[:len(weighted_scores_with_details)]

        weighted_total = 0
        total_weight = sum(weights)

        for i, result in enumerate(weighted_scores_with_details):
            score = result["output"]  # Hole die Punktzahl
            weight = weights[i]
            weighted_total += score * weight

        # Berechne den gewichteten Durchschnitt
        weighted_average = weighted_total / total_weight

        # Bestimme die Note
        final_grade = map_score_to_grade(weighted_average)

        return {
            "score": round(weighted_average, 2),
            "grade": final_grade,
            "details": []  # Entfernt category und explanation, da sie redundant sind
        }
    except Exception as e:
        logger.error(f"Fehler bei der Berechnung der Endnote: {e}")
        return {"score": 0, "grade": "Fehler", "details": []}

# Initialisiere das Agentensystem
agent_system = AgentSystem()
if not agent_system.validate_saved_data():
    logger.warning("Es gibt ungültige oder nicht abrufbare gespeicherte Daten.")
agent_system.add_agent(synopsis_agent, kontrolliert=True)
agent_system.add_agent(synopsis_validation_agent, kontrolliert=False)
agent_system.add_agent(chapter_agent, kontrolliert=True)
agent_system.add_agent(chapter_validation_agent, kontrolliert=False)
agent_system.add_agent(writing_agent, kontrolliert=True)
agent_system.add_agent(generate_summary, kontrolliert=True)
agent_system.add_agent(validate_summary, kontrolliert=False)
agent_system.add_agent(evaluate_chapters, kontrolliert=False)
agent_system.add_agent(evaluate_paragraphs, kontrolliert=False)
agent_system.add_agent(evaluate_book_type, kontrolliert=False)
agent_system.add_agent(evaluate_content, kontrolliert=False)
agent_system.add_agent(evaluate_grammar, kontrolliert=False)
agent_system.add_agent(evaluate_style, kontrolliert=False)
agent_system.add_agent(evaluate_tension, kontrolliert=False)
