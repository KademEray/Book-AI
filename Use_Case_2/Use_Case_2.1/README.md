# **Use Case 2.1**



## ✨ **Beschreibung**

**Use Case 2.1** erweitert die Funktionalitäten von Use Case 1, indem zwei Validierungsagenten gleichzeitig verwendet werden. Dies ermöglicht eine gründlichere Prüfung der generierten Inhalte und stellt sicher, dass diese den höchsten Qualitätsstandards entsprechen. Der Fokus liegt auf:

1. **Inhaltserstellung**: Texte werden basierend auf einem vorgegebenen Prompt erstellt.
2. **Duale Validierung**: Zwei Agenten überprüfen die Inhalte unabhängig voneinander.

Dieses System bietet erhöhte Präzision und Redundanz bei der Validierung.

---

## 🧠 **Verwendetes KI-Modell**

Das Projekt verwendet das Modell **`llama-3.2-3b-instruct`**, welches für folgende Eigenschaften bekannt ist:

- **Hohe Effizienz** bei der Verarbeitung von Prompts.
- **Optimierte Leistung** für die Erstellung längerer Texte.
- **Unterstützung für größere Kontextfenster**, ideal für umfangreiche Inhalte.

---

## 🤖 **Agenten im Einsatz**

In **Use Case 2.1** kommen zwei **Validierungsagenten** zum Einsatz. Diese arbeiten unabhängig voneinander und ermöglichen eine umfassendere Prüfung der generierten Inhalte. Die Schritte sind:

1. **Erstellung**: Die Inhalte werden generiert.
2. **Prüfung durch Agent 1**: Inhalte werden auf Konsistenz und Qualität überprüft.
3. **Prüfung durch Agent 2**: Eine zweite, unabhängige Überprüfung gewährleistet zusätzliche Sicherheit.

---

## 🔄 **Unterschiede zu Use Case 1**

1. **Anzahl der Agenten**:
   - **Use Case 1.1**: Ein einzelner Validierungsagent.
   - **Use Case 2.1**: Zwei unabhängige Validierungsagenten.

2. **Validierungsprozess**:
   - **Use Case 1.1**: Einzelne Prüfung durch einen Agenten.
   - **Use Case 2.1**: Doppelte Prüfung durch 2 Agenten.

3. **Fehlersicherheit**:
   - **Use Case 2.1** bietet durch die duale Validierung eine größere Fehlersicherheit.

---

## 🛠 **Besonderheiten von Use Case 2.1**

- **Duale-Agenten-Architektur**: Zwei Agenten erhöhen die Zuverlässigkeit der Inhalte.
- **Erweiterte Fehlersicherheit**: Unabhängige Überprüfung reduziert das Risiko von Ungenauigkeiten.
- **Ideal für kritische Inhalte**: Eignet sich besonders für komplexe Projekte, die höchste Genauigkeit erfordern.

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
   - Wähle das Modell **`llama-3.2-3b-instruct`** aus.
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
2. **`chatAgent.py`**: Enthält die Implementierung der Validierungsagenten.
3. **`main.py`**: Einstiegspunkt für die Ausführung des Use Cases. Hier werden alle Module geladen und der Ablauf gesteuert.
4. **`agent.py`**: Kernkomponenten für die KI-Interaktion, die die Kommunikation zwischen Modell und Backend erleichtern.
5. **`duckduckgo.py`**: Modul zur Integration von externen Suchdiensten, um die Generierung mit zusätzlichem Wissen zu unterstützen.
6. **`ollama.py`**: Ein Modul, das spezifisch für die Interaktion mit dem `llama-3.2-3b-instruct` Modell optimiert wurde.

### **Ergebnisse-Verzeichnis**

- **`Use_Case_2.1.txt`**: Standarddatei, in der die Inhalte gespeichert werden, die aus diesem Use Case generiert wurden.

---

## 📝 **Hinweis**

Dieses Projekt wurde so konzipiert, dass es lokal ausgeführt werden kann, um unnötige Abfragen an externe Server zu vermeiden. Dies gewährleistet höhere Sicherheit und schnellere Verarbeitung.

---
