# **Use Case 1.2**



## ✨ **Beschreibung**

**Use Case 1.2** ist ein innovativer Ansatz, um Inhalte mithilfe des KI-Modells `dolphin3.0-llama3.1-8b` zu erstellen und zu validieren. Das Ziel ist die automatische Generierung und Überprüfung von Texten, die speziell auf den gewünschten Output abgestimmt sind. Dieser Use Case fokussiert sich auf:

1. **Inhaltserstellung**: Texte werden basierend auf einem vorgegebenen Prompt erstellt.
2. **Validierung**: Ein Agent überprüft die Inhalte auf Qualität und Konsistenz.

Das System ist darauf ausgelegt, effektive Inhalte mit minimaler Benutzereingabe zu erstellen.

---

## 🧠 **Verwendetes KI-Modell**

Das Projekt verwendet das Modell **`dolphin3.0-llama3.1-8b`**, welches für folgende Eigenschaften bekannt ist:

- **Hohe Effizienz** bei der Verarbeitung von Prompts.
- **Optimierte Leistung** für die Erstellung längerer Texte.
- **Unterstützung für größere Kontextfenster**, ideal für umfangreiche Inhalte.

---

## 🤖 **Agenten im Einsatz**

In **Use Case 1.2** wird ein **Validierungsagent** eingesetzt, der speziell dafür entwickelt wurde, die erzeugten Inhalte auf Richtigkeit, Relevanz und Kohärenz zu überprüfen. Dieser Agent arbeitet mit den folgenden Schritten:

1. **Erstellung**: Der Agent generiert Inhalte basierend auf dem Prompt.
2. **Prüfung**: Die generierten Inhalte werden auf Konsistenz und Qualität überprüft.

---

## 🛠 **Besonderheiten von Use Case 1.2**

- **Single-Agent-Architektur**: Verwendet nur einen Validierungsagenten für schnelle und effiziente Ergebnisse.
- **Niedriger Ressourcenverbrauch**: Optimiert für lokale Ausführung mit minimaler Hardwareanforderung.
- **Flexibilität**: Ideal für kürzere und prägnante Inhalte.

---

## 🚀 **Anleitung zur Ausführung**

### **Voraussetzungen**

- **Python-Version**: 3.10
- **LM Studio** (mit Modellen, die mindestens 128k Token unterstützen)

### **Installation**

1. **Abhängigkeiten installieren**:

   ```bash
   pip install -r requirements.txt
   ```

2. **LM Studio einrichten**:
   - Gehe zum **Developer Tab**.
   - Wähle das Modell **`dolphin3.0-llama3.1-8b`** aus.
   - Passe die **Context Length** an (Standard: 55k, für längere Inhalte: 128k).
   - Lade das Modell und öffne die **Servereinstellungen**:
     - Port: `1234`
     - **Enable CORS** aktivieren.

### **Ausführung**

Führe das Projekt mit folgendem Befehl aus:

```bash
python main.py
```

---

## 🌟 **Ergebnisse**

Nach der Ausführung finden Sie die generierten Inhalte im Ordner **`Ergebnisse`**. Standardmäßig wird der Inhalt anhand eines Prompts wie:

> *"Erstelle mir einen Aufsatz über Machine Learning."*

erstellt. Dieser Inhalt wird validiert und gespeichert.

---

## 📂 **Projektstruktur**

### **Hauptverzeichnis**

- **`README.md`**: Diese Datei mit allen Anweisungen und Beschreibungen.
- **`requirements.txt`**: Liste aller Abhängigkeiten für das Projekt.
- **`Ergebnisse/`**: Ordner, in dem die generierten Inhalte gespeichert werden.
- **`backend/`**: Beinhaltet die Hauptlogik des Projekts.

### **Backend-Verzeichnis**

Das `backend/`-Verzeichnis enthält folgende Dateien:

1. **`backend.py`**: Die zentrale Steuerungseinheit für die Verarbeitung der Prompts und Generierung der Inhalte.
2. **`chatAgent.py`**: Enthält die Implementierung des Validierungsagenten, der die generierten Inhalte prüft.
3. **`main.py`**: Einstiegspunkt für die Ausführung des Use Cases. Hier werden alle Module geladen und der Ablauf gesteuert.
4. **`agent.py`**: Kernkomponenten für die KI-Interaktion, die die Kommunikation zwischen Modell und Backend erleichtern.
5. **`duckduckgo.py`**: Modul zur Integration von externen Suchdiensten, um die Generierung mit zusätzlichem Wissen zu unterstützen.
6. **`ollama.py`**: Ein Modul, das spezifisch für die Interaktion mit dem `dolphin3.0-llama3.1-8b` Modell optimiert wurde.

### **Ergebnisse-Verzeichnis**

- **`Use_Case_1.2.txt`**: Standarddatei, in der die Inhalte gespeichert werden, die aus diesem Use Case generiert wurden.

---

## 📝 **Hinweis**

Dieses Projekt wurde so konzipiert, dass es lokal ausgeführt werden kann, um unnötige Abfragen an externe Server zu vermeiden. Dies gewährleistet höhere Sicherheit und schnellere Verarbeitung und weniger kosten.

---
