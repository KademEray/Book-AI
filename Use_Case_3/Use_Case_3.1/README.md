# **Use Case 3.1**



## ✨ **Beschreibung**

**Use Case 3.1** ist ein fortschrittliches System, das eine dreifache Validierungsarchitektur verwendet. Ziel ist es, Inhalte mit maximaler Präzision zu erstellen und zu überprüfen, indem drei unabhängige Validierungsagenten gleichzeitig arbeiten. Dieser Ansatz gewährleistet höchste Qualität und Zuverlässigkeit der generierten Inhalte. 

Hauptmerkmale dieses Use Cases:

1. **Inhaltserstellung**: Erstellung von strukturierten und detaillierten Texten basierend auf einem komplexen Prompt.
2. **Dreifache Validierung**: Inhalte werden von drei Agenten geprüft, um unterschiedliche Perspektiven und Prüfmethoden zu berücksichtigen.
3. **Erweiterte Analyse**: Fokus auf sprachliche, strukturelle und inhaltliche Genauigkeit.

---

## 🧠 **Verwendetes KI-Modell**

Das Modell **`llama-3.2-3b-instruct`** wird in **Use Case 3.1** eingesetzt. Seine Hauptvorteile sind:

- **Effizienz in der Textverarbeitung**.
- **Hohe Präzision** bei der Generierung umfangreicher und komplexer Inhalte.
- **Unterstützung für größere Kontexte**, ideal für mehrstufige Validierungen.

---

## 🤖 **Agenten im Einsatz**

In **Use Case 3.1** werden drei Validierungsagenten eingesetzt, die unabhängig voneinander arbeiten, um eine tiefgehende und umfassende Prüfung sicherzustellen. Die Aufgaben sind wie folgt:

1. **Agent 1**: Fokussiert auf sprachliche und grammatikalische Überprüfung.
2. **Agent 2**: Bewertet die Struktur und Kohärenz des Textes.
3. **Agent 3**: Fokussiert auf inhaltliche Genauigkeit und Relevanz.

Die Zusammenarbeit dieser Agenten ermöglicht eine vielschichtige Validierung und stellt sicher, dass alle Aspekte des Textes von höchster Qualität sind.

---

## 🔄 **Unterschiede zu Use Case 2**

1. **Anzahl der Agenten**:
   - **Use Case 2.1/2.2**: Zwei Agenten.
   - **Use Case 3.1**: Drei Agenten.

2. **Validierungsprozesse**:
   - **Use Case 2.1/2.2**: Duale Validierung mit grundlegenden und erweiterten Prüfungen.
   - **Use Case 3.1**: Dreifache Validierung mit spezifischen Prüfbereichen für jeden Agenten.

3. **Anwendungsszenarien**:
   - **Use Case 2.1/2.2**: Für komplexe, aber kleinere Projekte geeignet.
   - **Use Case 3.1**: Optimal für Projekte mit höchsten Qualitätsanforderungen und umfangreichen Inhalten.

---

## 🛠 **Besonderheiten von Use Case 3.1**

- **Dreifache-Agenten-Architektur**: Maximale Genauigkeit durch spezialisierte Prüfungen.
- **Höchste Qualitätsstandards**: Geeignet für Projekte mit höchsten Anforderungen an sprachliche, strukturelle und inhaltliche Präzision.
- **Flexibilität und Tiefe**: Ideal für umfangreiche und anspruchsvolle Textprojekte.

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

> *"Erstelle einen ausführlichen Bericht über die Entwicklung und Auswirkungen von künstlicher Intelligenz."*

erstellt. Die Inhalte durchlaufen alle drei Validierungsstufen und werden abschließend gespeichert.

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
2. **`chatAgent.py`**: Implementierung der spezialisierten Validierungsagenten.
3. **`main.py`**: Einstiegspunkt für die Ausführung des Use Cases. Hier werden alle Module geladen und der Ablauf gesteuert.
4. **`agent.py`**: Kernkomponenten für die KI-Interaktion, die die Kommunikation zwischen Modell und Backend erleichtern.
5. **`duckduckgo.py`**: Modul zur Integration von externen Suchdiensten, um die Generierung mit zusätzlichem Wissen zu unterstützen.
6. **`ollama.py`**: Ein Modul, das spezifisch für die Interaktion mit dem `llama-3.2-3b-instruct` Modell optimiert wurde.

### **Ergebnisse-Verzeichnis**

- **`Use_Case_3.1.txt`**: Standarddatei, in der die Inhalte gespeichert werden, die aus diesem Use Case generiert wurden.

---

## 📝 **Hinweis**

Dieses Projekt wurde so konzipiert, dass es lokal ausgeführt werden kann, um unnötige Abfragen an externe Server zu vermeiden. Dies gewährleistet höhere Sicherheit und schnellere Verarbeitung.

---
