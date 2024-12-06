from flask import Flask, request, jsonify
import requests

# Flask-Setup
app = Flask(__name__)

# Agentensystem
class AgentSystem:
    def __init__(self):
        self.agents = []

    def add_agent(self, agent_function, kontrolliert=False):
        self.agents.append({"function": agent_function, "kontrolliert": kontrolliert})

    def run_agents(self, user_input):
        response_data = {"steps": [], "final_response": ""}
        context = user_input  # Initialer Kontext ist die Benutzer-Eingabe
        previous_output = None
        for agent in self.agents:
            result = agent["function"](context)
            response_data["steps"].append(result["log"])
            context = result["output"]  # Übergabe des Outputs an den nächsten Agenten
            
            # Kontrolliere den Output, falls aktiviert
            if agent["kontrolliert"]:
                validation_result = validation_agent(user_input, result["output"])
                response_data["steps"].append(validation_result["log"])

                # Falls die Validierung fehlschlägt, wiederhole den vorherigen Agenten
                while validation_result["log"]["status"] == "failed":
                    reason = validation_result["log"]["output"]
                    result = agent["function"](context + f"\nBegründung für Anpassung: {reason}")
                    response_data["steps"].append(result["log"])
                    context = result["output"]

                    # Erneute Validierung
                    validation_result = validation_agent(user_input, result["output"])
                    response_data["steps"].append(validation_result["log"])

        response_data["final_response"] = context
        return response_data

# Agent 1: Synopsis erstellen
def synopsis_agent(user_input):
    log = {
        "agent": "Synopsis Agent",
        "status": "running",
        "output": ""
    }
    
    try:
        # Prompt für die Synopsis
        prompt_template = """
        Aufgabe: Schreibe eine detailreiche Synopsis des folgenden Textes.
        Die Synopsis soll alle Kernfragen und Aspekte des Themas ausführlich darstellen.

        Text:
        {input}

        Ausgabe:
        """
        prompt = prompt_template.format(input=user_input)

        # Simulierter LLM-Output
        if "Begründung für Anpassung" in user_input:
            synopsis = "Angepasste Synopsis: Apple bleibt durch innovative Strategien und Markenbindung erfolgreich."
        else:
            synopsis = """
            Die Analyse stellt fest, dass Apple trotz scheinbarer Nachteile im Vergleich zur Konkurrenz weiterhin erfolgreich bleibt. Der Kern liegt in Apples einzigartigen Marketingstrategien, der starken Markenbindung und dem geschlossenen Ökosystem, das Kunden langfristig bindet.
            """
        
        log["status"] = "completed"
        log["output"] = synopsis
        return {"log": log, "output": synopsis}
        
    except Exception as e:
        log["status"] = "failed"
        log["output"] = str(e)
        return {"log": log, "output": str(e)}

# Validierungs-Agent: Überprüft den Output inhaltlich
def validation_agent(user_input, output):
    log_validation = {"agent": "ValidationAgent", "status": "processing"}
    try:
        # Prüfen, ob der Output relevant ist
        if len(output.strip()) == 0:
            log_validation.update({"status": "failed", "output": "Schlecht: Kein Output vorhanden."})
        elif all(keyword.lower() in output.lower() for keyword in user_input.split()[:5]):
            log_validation.update({"status": "completed", "output": "Gut: Der Output passt inhaltlich gut zum Input."})
        else:
            log_validation.update({"status": "failed", "output": "Schlecht: Der Output greift den Input nicht richtig auf."})
    except Exception as e:
        log_validation.update({"status": "failed", "error": str(e), "output": "Schlecht: Fehler bei der Validierung."})
    return {"log": log_validation}

# Initialisiere das Agentensystem
agent_system = AgentSystem()
agent_system.add_agent(synopsis_agent, kontrolliert=True)

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        print(f"Eingehende Anfrage: {data}")

        # Starte die Agentenverarbeitung
        response_data = agent_system.run_agents(user_input)
        return jsonify(response_data)
    except Exception as e:
        print(f"Fehler bei der Anfrage: {str(e)}")
        return jsonify({"error": f"Fehler bei der Verarbeitung: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
