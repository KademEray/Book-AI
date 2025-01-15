# Book-AI

## Mitglieder
- Atakan Yalcin
- Eray Kadem 
- Ibrahim Demba Seck
- Karim Hamade

## Beschreibung
In dem Projekt geht es darum mithilfe einer Lokalen KI einen Buch zu erstellen dabei gibt es 3 Hauptschritte:
- Synopsis
    - Synopsis wird erstellt
    - Synopsis wird validiert
- Kapitel
    - Kapitel wird erstellt
    - Kapitel wird validiert
    - Unterkapitel werden erstellt
    - Unterkapitel werden validiert
- Inhaltes
    - Inhalt wird erstellt
    - Inhalt wird validiert

Danach kommt die Evaluierung, dabei wird eine Zusammenfassung erstellt und validiert wenn dieser fehlschlägt wird ab dem Prozess des Inhalt erstellens wiederholt, wenn die validierung erfolgreich war kommt die Buchbewertung.

Buchbewertung Gewichtung:
- Kapitel 15%
- Absätze 15%
- Buchart 10%
- Inhalt 20%
- Grammatik 15%
- Stil 15%
- Spannung 10%

Die Buchbewertung dient dazu da um den Nutzer darauf hinzuweisen welche mängeln das Buch hat.

### Hinweis:

Der Grund warum hier mit einer Lokalen KI gearbeitet wird liegt daran dass in einer einzelnen Ausführung über 100 Anfragen an der KI erstellt werden kann und diese auch einen großen Input Token haben kann.


## Requirements
- Python version: 3.10
- LM Studio (Nur Modelle mit einen mindest Limit von 128k Token)

## Installation
```bash
pip install -r requirements.txt
```

## Ausführung
LM Studio:
- Developer Tab
- Modell auswählen 
- Context Length anpassen (öfters reicht 55k aber bei einem längeren Buch das Maximum(mind. 128k))
- Load Modell
- Settings vom Server öffnen:
    - Port 1234
    - Enable CORS activated
    
```bash
python main.py
```


## Use Cases

Unterschiede der Use Cases:
- **Use Case 1**: Verwendet einen Validierungsagenten, um den Inhalt durch die KI zu validieren.
- **Use Case 2**: Setzt zwei Validierungsagenten ein, die unabhängig voneinander arbeiten.
- **Use Case 3**: Integriert drei Validierungsagenten für mehrstufige Validierungen des Inhalts.
- **Modellunterschiede**:
  - **Use Case X.1**: Nutzt das Modell „llama-3.2-3b-instruct“.
  - **Use Case X.2**: Nutzt das Modell „dolphin3.0-llama3.1-8b“ für komplexere Inhalte.


## Ergebnisse

Unter dem Ordner der einzelnen Use Cases befindet sich einen Ordner mit dem Namen Ergebnisse, darin befindet sich einen Buch der von dem Use Case erstellt wurde. Bei der Erstellung wurde dem keine Anzahl an Chapters oder Subchapter gegeben mit dem Promt: "Erstelle mir einen Aufsatz über Machine Learning"

## Vorschau
![alt text](<vorschau.jpg>)

