from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import requests
from datetime import datetime
import re

# Flask-Setup
app = Flask(__name__)

class OllamaLLM:
    """Wrapper für Ollama LLM."""
    def _call(self, prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 12000,
                    }
                },
                timeout=3000
            )
            response.raise_for_status()
            return response.json().get("response", "Fehler: Keine Antwort von Ollama.")
        except requests.exceptions.RequestException as e:
            return f"Fehler bei der Verbindung zu Ollama: {str(e)}"

# Benutzerdefinierte Embedding-Funktion
class CustomEmbeddingFunction(Embeddings):
    def __init__(self, n_features=128):
        print(f"Initializing HashingVectorizer with {n_features} features.")
        self.vectorizer = HashingVectorizer(n_features=n_features, norm=None, alternate_sign=False)
        self.dimension = n_features

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.vectorizer.transform(texts).toarray()
            return embeddings.tolist()
        except Exception as e:
            print(f"Fehler bei der Batch-Embedding-Berechnung: {e}")
            return [[0.0] * self.dimension] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Fehler bei der Query-Embedding-Berechnung: {e}")
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

    def add_agent(self, agent_function, kontrolliert=False):
        self.agents.append({"function": agent_function, "kontrolliert": kontrolliert})

    def run_agents(self, user_input, chat_id="default"):
        response_data = {"steps": [], "final_response": ""}
        validated_synopsis = None
        validated_chapters = None
        chapter_dict = None
        final_text = {}
        context = self.get_context(chat_id)  # Kontext abrufen

        # Schritt 1: Synopsis-Agent ausführen
        synopsis_result = synopsis_agent(user_input, context, chat_id)
        response_data["steps"].append(synopsis_result["log"])
        if synopsis_result["log"]["status"] != "completed":
            raise ValueError("Synopsis konnte nicht erstellt werden.")
        validated_synopsis = synopsis_result["output"]

        # Schritt 2: Kapitelstruktur-Agent ausführen
        chapter_result = chapter_agent(user_input, validated_synopsis, chat_id)
        response_data["steps"].append(chapter_result["log"])
        if chapter_result["log"]["status"] != "completed":
            raise ValueError("Kapitelstruktur konnte nicht erstellt werden.")
        validated_chapters = chapter_result["output"]

        # Schritt 3: Dictionary-Agent ausführen
        dictionary_result = dictionary_agent(validated_chapters, chat_id)
        response_data["steps"].append(dictionary_result["log"])
        if dictionary_result["log"]["status"] != "completed":
            raise ValueError("Kapitel-Dictionary konnte nicht erstellt werden.")
        chapter_dict = dictionary_result["output"]

        # Schritt 4: Kapitel-Dictionary validieren
        validation_result = chapter_validation_agent(chapter_dict, chat_id)
        response_data["steps"].append(validation_result["log"])
        if validation_result["log"]["status"] != "completed":
            raise ValueError("Kapitel-Dictionary konnte nicht validiert werden.")

        # Schritt 5: Inhalte für jedes Kapitel generieren und validieren
        writing_result = writing_agent(user_input, chapter_dict, chat_id)
        response_data["steps"].append(writing_result["log"])
        if writing_result["log"]["status"] != "completed":
            raise ValueError("Kapitelinhalte konnten nicht erstellt werden.")
        final_text = writing_result["output"]

        # Finale Ausgabe
        print("\nFinales Kapitel-Dictionary mit Inhalten:")
        print(json.dumps(final_text, indent=4))

        response_data["final_response"] = final_text
        return response_data


    def get_context(self, chat_id):
        """Abrufen des Kontexts aus der Chroma-Datenbank basierend auf der Chat-ID."""
        try:
            results = vectorstore.get()
            context_data = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                if metadata.get('chat_id') == chat_id:
                    context_data.append(doc)
            return "\n".join(context_data)
        except Exception as e:
            print(f"Fehler beim Abrufen des Kontexts: {e}")
            return ""

    def store_context(self, chat_id, user_input, final_output):
        """Speichern des Benutzerinputs und der validierten finalen Ausgabe in der Chroma-Datenbank."""
        try:
            vectorstore.add_texts(
                texts=[f"User: {user_input}\nAssistant: {final_output}"],
                metadatas=[{"chat_id": chat_id, "timestamp": datetime.now().isoformat()}]
            )
            print(f"Kontext für Chat-ID {chat_id} erfolgreich gespeichert.")
        except Exception as e:
            print(f"Fehler beim Speichern des Kontexts: {e}")

# Synopsis-Agent
def synopsis_agent(user_input, context, chat_id=None):
    log = {"agent": "SynopsisAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Kontext:
        {context}

        Aufgabe: Erstelle eine prägnante Synopsis basierend auf dem folgenden Benutzerinput:
        {user_input}
        """
        llm = OllamaLLM()
        response = llm._call(prompt)

        log.update({"status": "completed", "output": response})
        return {"log": log, "output": response}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": f"Fehler: {str(e)}"}

# Validierungs-Agent
def validation_agent(user_input, output, chat_id):
    log = {"agent": "ValidationAgent", "status": "processing"}

    try:
        context = agent_system.get_context(chat_id)  # Kontext abrufen
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

    return {"log": log}

# Chapter-Agent
def chapter_agent(user_input, validated_synopsis, chat_id=None):
    log = {"agent": "ChapterAgent", "status": "running", "details": []}
    try:
        prompt = f"""
        Basierend auf der validierten Synopsis, erstelle eine logische und strukturierte Liste von Kapiteln:

        Validierte Synopsis:
        {validated_synopsis}

        Die Kapitel sollten klar benannt und in einer sinnvollen Reihenfolge organisiert sein.
        """
        llm = OllamaLLM()
        response = llm._call(prompt)

        log.update({"status": "completed", "output": response})
        return {"log": log, "output": response}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": f"Fehler: {str(e)}"}

def clean_response(response):
    """Bereinigt die Antwort und extrahiert das JSON."""
    # Entferne unzulässige Zeichen
    response = response.strip().replace("```", "").replace("\n", " ").strip()
    response = response.replace("„", '"').replace("“", '"').replace("‚", '"').replace("‘", '"')
    response = re.sub(r"[^\x20-\x7E]+", "", response)  # Entferne nicht druckbare Zeichen
    response = re.sub(r"\s+", " ", response)  # Entferne überschüssige Leerzeichen

    # Extrahiere das JSON-Objekt aus der Antwort
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Kein gültiges JSON gefunden.")
        response = response[json_start:json_end]
        # Validierung und Laden des JSON
        return json.loads(response)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Fehler beim Extrahieren des JSON: {str(e)}")

def adjust_chapter_keys(chapter_dict):
    """Passt die Kapitelnummerierung an."""
    adjusted_dict = {"Chapters": []}
    for index, chapter in enumerate(chapter_dict.get("Chapters", []), start=1):
        chapter["Number"] = index  # Konvertiere die Nummer in eine Zahl
        adjusted_dict["Chapters"].append(chapter)
    return adjusted_dict

def validate_json_structure(chapter_dict):
    """Validiert die JSON-Struktur."""
    print(f"Validiere JSON-Struktur: {json.dumps(chapter_dict, indent=4)}")
    if "Chapters" not in chapter_dict or not isinstance(chapter_dict["Chapters"], list):
        raise ValueError("Die JSON-Struktur enthält keine gültige 'Chapters'-Liste.")

    expected_numbers = list(range(1, len(chapter_dict["Chapters"]) + 1))
    actual_numbers = [int(chapter.get("Number")) for chapter in chapter_dict["Chapters"]]

    if actual_numbers != expected_numbers:
        raise ValueError(f"Fehlende oder inkorrekte Kapitelnummerierung. Erwartet: {expected_numbers}, Erhalten: {actual_numbers}")

    for chapter in chapter_dict["Chapters"]:
        if not all(key in chapter for key in ["Number", "Title", "Content"]):
            raise ValueError(f"Kapitel fehlt eines der Felder 'Number', 'Title' oder 'Content': {chapter}")
        if not isinstance(chapter["Number"], int):
            raise ValueError(f"Ungültige Kapitelnummer: {chapter['Number']}")
        if not isinstance(chapter["Title"], str) or not chapter["Title"]:
            raise ValueError(f"Ungültiger Titel im Kapitel: {chapter}")
        if not isinstance(chapter["Content"], str):
            raise ValueError(f"Ungültiger Inhalt im Kapitel: {chapter}")

def dictionary_agent(chapter_list, chat_id, **kwargs):
    """Generiert ein JSON-Objekt basierend auf den Kapitel-Details."""
    log = {"agent": "DictionaryAgent", "status": "running", "details": []}
    context = agent_system.get_context(chat_id)

    while True:
        try:
            prompt = f"""
            Kontext:
            {context}

            Erstelle ein gültiges JSON-Objekt, das eine Dokumentstruktur repräsentiert. Das JSON sollte ein Array von Kapiteln enthalten mit den Feldern:
            - `Number`: Die Nummer des Kapitels.
            - `Title`: Der Titel des Kapitels.
            - `Content`: Der Inhalt des Kapitels.

            Das JSON muss:
            1. Korrekte Syntax haben, alle Schlüssel und Strings in Anführungszeichen (") setzen.
            2. Keine führenden oder abschließenden Kommata enthalten.
            3. Mindestens 3 Kapitel mit fortlaufender Nummerierung ab 1 enthalten.

            Beispielstruktur:
            {{
                "Chapters": [
                    {{"Number": 1, "Title": "Introduction", "Content": "Details of the introduction."}},
                    {{"Number": 2, "Title": "Main Part", "Content": "Details of the main content."}}
                ]
            }}

            Gib nur das JSON zurück.
            """
            llm = OllamaLLM()
            response = llm._call(prompt)
            print(f"Raw Response from dictionary_agent:\n{response}")

            # Extrahiere und validiere das JSON
            chapter_dict = clean_response(response)
            chapter_dict = adjust_chapter_keys(chapter_dict)
            validate_json_structure(chapter_dict)

            # Wenn alles korrekt ist, wird das JSON zurückgegeben
            log.update({"status": "completed", "output": chapter_dict})
            agent_system.store_context(chat_id, chapter_list, json.dumps(chapter_dict))
            return {"log": log, "output": chapter_dict}

        except ValueError as e:
            print(f"Fehler beim Generieren oder Validieren des JSON: {e}")
            continue


def extract_dictionary(output, variable_name="X_Dict"):
    """Extrahiert JSON aus einem Textoutput."""
    try:
        # Suche nach dem JSON-Objekt im Text
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError(f"'{variable_name}' konnte nicht gefunden werden.")
        json_str = output[json_start:json_end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Fehler bei der Extraktion des JSON: {e}")
    
def chapter_validation_agent(chapter_dict, chat_id, **kwargs):
    log = {"agent": "ChapterValidationAgent", "status": "running", "details": []}
    while True:
        try:
            context = agent_system.get_context(chat_id)
            prompt = f"""
            Kontext:
            {context}

            Überprüfen Sie die folgende Kapitelstruktur (Python-Dictionary) mit dem Namen "X_Dict":

            {json.dumps(chapter_dict, indent=4)}

            Anforderungen:
            1. Alle Schlüssel müssen mit 'Kapitel X' beginnen (z.B. 'Kapitel 1').
            2. Die Werte dürfen keine leeren Felder enthalten.
            3. Es dürfen keine Kapitelnummern übersprungen oder wiederholt werden.
            4. Die Kapitelnummern müssen aufeinanderfolgend sein.

            Antworten Sie mit 'Ja', wenn das Dictionary gültig ist, oder 'Nein' mit einer detaillierten Begründung und Korrekturvorschlägen.
            """
            llm = OllamaLLM()
            response = llm._call(prompt)

            if response.strip().startswith("Ja"):
                log.update({
                    "status": "completed",
                    "output": "Kapitelstruktur validiert. Keine Änderungen erforderlich."
                })
                agent_system.store_context(chat_id, chapter_dict, "Kapitelstruktur validiert.")
                return {"log": log, "output": chapter_dict}
            else:
                print(f"Validierungsfehler: {response}")
                raise ValueError(f"Validierung fehlgeschlagen: {response}")
        except ValueError as e:
            print(f"Fehler bei der Validierung: {e}")
            continue

def writing_agent(user_input, chapter_dict, chat_id):
    log = {"agent": "WritingAgent", "status": "running", "details": []}
    final_text = {"Chapters": []}

    try:
        for chapter in chapter_dict["Chapters"]:
            validated = False

            while not validated:
                prompt = f"""
                Kapitel {chapter['Number']} - {chapter['Title']}

                Schreibe den vollständigen Text für dieses Kapitel. Konzentriere dich ausschließlich auf den Inhalt des Kapitels und vermeide jegliche Hinweise oder Erklärungen zum Benutzerinput oder zum Schreibprozess. Gib nur den reinen Text des Kapitels zurück.
                """
                llm = OllamaLLM()
                chapter_content = llm._call(prompt)

                # Kapiteltext validieren
                validation_result = validation_agent(user_input, chapter_content, chat_id)
                log["details"].append(validation_result["log"])

                if validation_result["log"]["status"] == "completed":
                    validated = True
                    final_text["Chapters"].append({
                        "Number": chapter["Number"],
                        "Title": chapter["Title"],
                        "Content": chapter_content
                    })
                    agent_system.store_context(chat_id, chapter["Title"], chapter_content)

        log.update({"status": "completed", "output": final_text})
        return {"log": log, "output": final_text}

    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": {}}


# Initialisiere das Agentensystem
agent_system = AgentSystem()
agent_system.add_agent(synopsis_agent, kontrolliert=True)
agent_system.add_agent(chapter_agent, kontrolliert=True)
agent_system.add_agent(dictionary_agent, kontrolliert=True)
agent_system.add_agent(chapter_validation_agent, kontrolliert=True)
agent_system.add_agent(writing_agent, kontrolliert=True)


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        chat_id = data.get("chat_id", "default")
        print(f"Received request: user_input={user_input}, chat_id={chat_id}")
        
        result = agent_system.run_agents(user_input, chat_id)
        print(f"Result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error during request processing: {e}")
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
