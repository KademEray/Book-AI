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
                json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
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
        context = self.get_context(chat_id)  # Kontext abrufen

        for agent in self.agents:
            validated = False

            while not validated:
                # Wähle die Eingabedaten basierend auf dem Agententyp
                if agent["function"].__name__ == "dictionary_agent":
                    input_data = validated_chapters
                elif agent["function"].__name__ == "chapter_validation_agent":
                    input_data = chapter_dict
                elif agent["function"].__name__ == "synopsis_agent":
                    input_data = context  # Kontext für synopsis_agent
                else:
                    input_data = validated_synopsis if agent["function"].__name__ == "chapter_agent" else context

                # Agent ausführen (chat_id wird immer übergeben)
                if agent["function"].__name__ == "synopsis_agent":
                    result = agent["function"](user_input, input_data, chat_id)
                elif agent["function"].__name__ == "chapter_agent":
                    result = agent["function"](user_input, validated_synopsis, chat_id)
                elif agent["function"].__name__ in ["dictionary_agent", "chapter_validation_agent"]:
                    result = agent["function"](input_data, chat_id)
                else:
                    result = agent["function"](input_data)  # Fallback, falls kein chat_id benötigt

                print(f"Agent: {result['log']['agent']}, Status: {result['log']['status']}, Output: {result['log']['output']}")
                response_data["steps"].append(result["log"])

                # Validierungslogik
                if agent["kontrolliert"]:
                    validation_result = validation_agent(user_input, result["output"], chat_id)
                    response_data["steps"].append(validation_result["log"])
                    print(f"Validierung: {validation_result['log']['output']}")

                    if validation_result["log"]["status"] == "completed":
                        validated = True
                        if agent["function"].__name__ == "synopsis_agent":
                            validated_synopsis = result["output"]
                        elif agent["function"].__name__ == "chapter_agent":
                            validated_chapters = result["output"]
                        elif agent["function"].__name__ == "dictionary_agent":
                            chapter_dict = result["output"]

                            # Wiederholungslogik für Dictionary-Agent
                            while True:
                                extracted_dict = extract_dictionary(chapter_dict)
                                if isinstance(extracted_dict, dict):
                                    print(f"Validiertes und extrahiertes Dictionary:\n{extracted_dict}")
                                    break  # Beende die Wiederholung, wenn das Dictionary gültig ist
                                else:
                                    print(f"Ungültiges Dictionary extrahiert: {extracted_dict}. Wiederhole den Agenten...")
                                    result = agent["function"](validated_chapters, chat_id)
                                    chapter_dict = result["output"]
                                    validation_result = validation_agent(user_input, chapter_dict, chat_id)
                                    if validation_result["log"]["status"] != "completed":
                                        print("Validierung erneut fehlgeschlagen. Wiederhole...")

                        self.store_context(chat_id, user_input, result["output"])
                    else:
                        print("Validierung fehlgeschlagen. Wiederhole Schritt...")
                else:
                    validated = True

        # Final Response Composition
        response_data["final_response"] = (
            f"Synopsis:\n\n{validated_synopsis}\n\nKapitelstruktur (Dictionary):\n\n{chapter_dict}"
            if chapter_dict
            else validated_synopsis or context
        )
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

def dictionary_agent(chapter_list, chat_id):
    log = {"agent": "DictionaryAgent", "status": "running", "details": []}
    try:
        context = agent_system.get_context(chat_id)  # Kontext abrufen
        prompt = f"""
        Kontext:
        {context}

        Erstellen Sie ein Python-Dictionary mit dem Namen "X_Dict", das die Kapitelstruktur wie folgt repräsentiert:
        - Schlüssel: Kapitelnummer (z.B. 'Kapitel 1')
        - Wert: Kapitelinhalt als Text

        Beispiel:
        X_Dict = {{
            'Kapitel 1': 'Einleitung',
            'Kapitel 2': 'Hauptteil',
            'Kapitel 3': 'Schluss'
        }}

        Kapitelstruktur:
        {chapter_list}

        Geben Sie ausschließlich den Python-Dictionary zurück, ohne zusätzliche Erklärungen oder Text.
        """
        llm = OllamaLLM()
        response = llm._call(prompt)
        log.update({"status": "completed", "output": response})
        
        # Kontext speichern
        agent_system.store_context(chat_id, chapter_list, response)
        return {"log": log, "output": response}
    except Exception as e:
        log.update({"status": "failed", "output": f"Fehler: {str(e)}"})
        return {"log": log, "output": f"Fehler: {str(e)}"}

def chapter_validation_agent(chapter_dict, chat_id):
    log = {"agent": "ChapterValidationAgent", "status": "running", "details": []}
    try:
        # Kontext abrufen
        context = agent_system.get_context(chat_id)
        
        # Präziser Prompt
        prompt = f"""
        Kontext:
        {context}

        Überprüfen Sie die folgende Kapitelstruktur, die in Form eines Python-Dictionarys vorliegt. Das Dictionary trägt den Namen "X_Dict".

        Kriterien:
        1. Ist die Reihenfolge der Kapitel logisch aufgebaut?
        2. Sind die Kapitel klar und prägnant benannt?
        3. Stimmen die Inhalte der Kapitel mit den Titeln überein?
        4. Sind alle Kapitel vollständig und enthalten keine Lücken?

        Hier ist das Dictionary:
        {chapter_dict}

        Anforderungen an Ihre Antwort:
        - Beginnen Sie mit "Ja" oder "Nein", ob die Kapitelstruktur korrekt ist.
        - Geben Sie eine detaillierte Begründung, falls "Nein".
        - Falls Korrekturen nötig sind, schlagen Sie eine überarbeitete Kapitelstruktur vor.
        """
        
        # LLM-Aufruf
        llm = OllamaLLM()
        response = llm._call(prompt)
        
        # Prüfung der Antwort auf "Ja" oder "Nein"
        if response.strip().startswith("Ja"):
            log.update({
                "status": "completed",
                "output": "Kapitelstruktur validiert. Keine Änderungen erforderlich."
            })
            # Kontext speichern
            agent_system.store_context(chat_id, chapter_dict, "Kapitelstruktur validiert.")
        elif response.strip().startswith("Nein"):
            log.update({
                "status": "failed",
                "output": f"Validierung fehlgeschlagen: {response}"
            })
        else:
            log.update({
                "status": "failed",
                "output": f"Unerwartete Antwort vom Modell: {response}"
            })

        return {"log": log, "output": response}
    
    except Exception as e:
        log.update({
            "status": "failed",
            "output": f"Fehler: {str(e)}"
        })
        return {"log": log, "output": f"Fehler: {str(e)}"}

# Initialisiere das Agentensystem
agent_system = AgentSystem()
agent_system.add_agent(synopsis_agent, kontrolliert=True)
agent_system.add_agent(chapter_agent, kontrolliert=True)
agent_system.add_agent(dictionary_agent, kontrolliert=True)
agent_system.add_agent(chapter_validation_agent, kontrolliert=True)


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
