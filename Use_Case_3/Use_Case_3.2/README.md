# **Use Case 3.2**



## ✨ **Beschreibung**

**Use Case 3.2** repräsentiert den fortschrittlichsten Ansatz innerhalb der Use Case 3-Reihe. Dieses System kombiniert eine dreifache Validierungsarchitektur mit einem hochentwickelten KI-Modell, um Inhalte mit maximaler Präzision, Tiefe und Relevanz zu erstellen. Es ist speziell für anspruchsvolle Projekte konzipiert, die höchste Qualitätsstandards erfordern. 

Hauptmerkmale dieses Use Cases:

1. **Komplexe Inhaltserstellung**: Erstellung von detaillierten und umfangreichen Texten auf Basis eines komplexen Prompts.
2. **Dreifache Validierung**: Drei spezialisierte Agenten überprüfen die Inhalte unabhängig voneinander, um Qualität und Genauigkeit sicherzustellen.
3. **Erweiterte Analyse**: Spezieller Fokus auf kreative, kohärente und faktenbasierte Inhalte.
4. **Optimiertes Modell**: Verwendung des leistungsstarken Modells **`dolphin3.0-llama3.1-8b`**.

---

## 🧠 **Verwendetes KI-Modell**

Das Modell **`dolphin3.0-llama3.1-8b`** wurde ausgewählt, um den Anforderungen komplexer Projekte gerecht zu werden. Seine Hauptvorteile sind:

- **Leistungsstarke Textgenerierung** für detaillierte Inhalte.
- **Erweiterte Kontextverarbeitung**, ideal für lange und anspruchsvolle Dokumente.
- **Verbesserte Validierungsfähigkeiten**, um alle Aspekte der Inhalte zu optimieren.

---

## 🤖 **Agenten im Einsatz**

In **Use Case 3.2** kommen drei spezialisierte Validierungsagenten zum Einsatz, die jeweils spezifische Prüfungen durchführen:

1. **Agent 1**: Überprüfung von Grammatik, Stil und Sprachfluss.
2. **Agent 2**: Analyse der Struktur und Kohärenz des Textes.
3. **Agent 3**: Validierung inhaltlicher Relevanz, Fakten und kreativer Elemente.

Diese Agenten arbeiten unabhängig voneinander und gewährleisten eine mehrstufige, tiefgreifende Prüfung aller erstellten Inhalte.

---

## 🔄 **Unterschiede zu Use Case 3.1**

1. **Verwendetes Modell**:
   - **Use Case 3.1**: `llama-3.2-3b-instruct`
   - **Use Case 3.2**: `dolphin3.0-llama3.1-8b` (leistungsstärker, ideal für kreative und komplexe Inhalte).

   

2. **Anwendungsszenarien**:
   - **Use Case 3.1**: Geeignet für Projekte mit hohem Qualitätsanspruch.
   - **Use Case 3.2**: Ideal für Projekte mit maximalem Anspruch an Kreativität und Genauigkeit.

---

## 🛠 **Besonderheiten von Use Case 3.2**

- **Erweiterte Dreifach-Agenten-Architektur**: Maximale Prüfgenauigkeit durch spezialisierte Agenten.
- **Kreative Optimierung**: Das Modell und die Agenten sind auf kreative und originelle Inhalte ausgerichtet.
- **Höchste Präzision und Tiefe**: Für umfassende und anspruchsvolle Projekte.

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

> *"Erstelle eine umfassende Analyse über die Entwicklung und Bedeutung der künstlichen Intelligenz in verschiedenen Industrien."*

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
6. **`ollama.py`**: Ein Modul, das spezifisch für die Interaktion mit dem `dolphin3.0-llama3.1-8b` Modell optimiert wurde.

### **Ergebnisse-Verzeichnis**

- **`Use_Case_3.2.txt`**: Standarddatei, in der die Inhalte gespeichert werden, die aus diesem Use Case generiert wurden.

---

## 📝 **Hinweis**

Dieses Projekt wurde so konzipiert, dass es lokal ausgeführt werden kann, um unnötige Abfragen an externe Server zu vermeiden. Dies gewährleistet höhere Sicherheit und schnellere Verarbeitung.

---
