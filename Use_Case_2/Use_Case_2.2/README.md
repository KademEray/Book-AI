# **Use Case 2.2**



## ✨ **Beschreibung**

**Use Case 2.2** stellt eine weitere Iteration der dualen Validierungsarchitektur dar, wobei ein noch umfassenderer Ansatz zur Erstellung und Überprüfung von Inhalten verfolgt wird. Dieses System nutzt fortschrittlichere Validierungsalgorithmen und spezialisierte Agenten, um eine tiefere Analyse und genauere Ergebnisse zu gewährleisten. Die Hauptaspekte dieses Use Cases sind:

1. **Erweiterte Inhaltserstellung**: Erzeugung von detaillierten und komplexen Texten basierend auf einem vorgegebenen Prompt.
2. **Verfeinerte Dual-Validierung**: Zwei Agenten arbeiten mit erweiterten Prüfmechanismen, um die generierten Inhalte kritisch zu analysieren.
3. **Optimierung durch spezielle Modelle**: Einsatz des Modells **`dolphin3.0-llama3.1-8b`**, das für komplexe Textgenerierung entwickelt wurde.

---

## 🧠 **Verwendetes KI-Modell**

Das Modell **`dolphin3.0-llama3.1-8b`** ist speziell für folgende Zwecke optimiert:

- **Erstellung hochkomplexer Inhalte**.
- **Erweiterte Kontextverarbeitung** für längere Absätze und detaillierte Kapitel.
- **Fortschrittliche Validierungsunterstützung**, die ideal für kritische Anwendungen ist.

---

## 🤖 **Agenten im Einsatz**

In **Use Case 2.2** kommen zwei spezialisierte **Validierungsagenten** zum Einsatz, die eine tiefere Analyse der Inhalte ermöglichen. Die Funktionen umfassen:

1. **Erstellung**: Inhalte werden basierend auf einem komplexen Prompt erstellt.
2. **Prüfung durch Agent 1**: Der erste Agent analysiert die Inhalte auf strukturelle Kohärenz und sprachliche Korrektheit.
3. **Prüfung durch Agent 2**: Der zweite Agent führt eine inhaltliche Validierung durch, einschließlich Faktenüberprüfung und Relevanzbewertung.

Die Zusammenarbeit dieser Agenten gewährleistet, dass die Inhalte sowohl technisch als auch inhaltlich einwandfrei sind.

---

## 🔄 **Unterschiede zu Use Case 2.1**

1. **Verwendetes Modell**:
   - **Use Case 2.1**: `llama-3.2-3b-instruct`
   - **Use Case 2.2**: `dolphin3.0-llama3.1-8b` (leistungsstärker, ideal für komplexe Texte).

2. **Validierungsprozesse**:
   - **Use Case 2.1**: Standard-Dual-Validierung.
   - **Use Case 2.2**: Erweiterte Prüfmechanismen und tiefere Analyse.

3. **Anwendungsszenarien**:
   - **Use Case 2.1**: Geeignet für kürzere und weniger komplexe Inhalte.
   - **Use Case 2.2**: Optimal für ausführliche und anspruchsvolle Projekte.

---

## 🛠 **Besonderheiten von Use Case 2.2**

- **Fortgeschrittene Dual-Agenten-Architektur**: Noch detailliertere Prüfungen durch spezialisierte Agenten.
- **Höhere Präzision**: Dank des optimierten Modells und fortschrittlicher Algorithmen.
- **Skalierbarkeit**: Eignet sich für Projekte mit umfangreichen Anforderungen und großen Datenmengen.

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

> *"Erstelle mir eine detaillierte Analyse über künstliche Intelligenz und ihre Auswirkungen."*

erstellt. Dieser Inhalt wird von beiden Agenten geprüft und final gespeichert.

---

## 📂 **Projektstruktur**

### **Hauptverzeichnis**

- **`README.md`**: Diese Datei mit allen Anweisungen und Beschreibungen.
- **`requirements.txt`**: Liste aller Abhängigkeiten für das Projekt.
- **`Ergebnisse/`**: Ordner, in dem die generierten Inhalte gespeichert werden.
- **`backend/`**: Beinhaltet die Hauptlogik des Projekts.

### **Backend-Verzeichnis**

Das `backend/`-Verzeichnis enthält folgende Dateien:

1. **`backend.py`**: Zentrale Steuerungseinheit für die Verarbeitung der Prompts und Generierung der Inhalte.
2. **`chatAgent.py`**: Implementierung der spezialisierten Validierungsagenten.
3. **`main.py`**: Einstiegspunkt für die Ausführung des Use Cases. Hier werden alle Module geladen und der Ablauf gesteuert.
4. **`agent.py`**: Kernkomponenten für die KI-Interaktion, die die Kommunikation zwischen Modell und Backend erleichtern.
5. **`duckduckgo.py`**: Modul zur Integration von externen Suchdiensten, um die Generierung mit zusätzlichem Wissen zu unterstützen.
6. **`ollama.py`**: Ein Modul, das spezifisch für die Interaktion mit dem `dolphin3.0-llama3.1-8b` Modell optimiert wurde.

### **Ergebnisse-Verzeichnis**

- **`Use_Case_2.2.txt`**: Standarddatei, in der die Inhalte gespeichert werden, die aus diesem Use Case generiert wurden.

---

## 📝 **Hinweis**

Dieses Projekt wurde so konzipiert, dass es lokal ausgeführt werden kann, um unnötige Abfragen an externe Server zu vermeiden. Dies gewährleistet höhere Sicherheit und schnellere Verarbeitung.

---