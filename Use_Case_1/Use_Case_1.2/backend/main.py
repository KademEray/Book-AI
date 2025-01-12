import requests
import json
import os
import shutil
import psutil
import sys
import time
from subprocess import Popen

BASE_URL = "http://localhost:5000/api"

def stop_backend():
    """
    Beendet den Prozess, der auf Port 5000 läuft.
    """
    try:
        for conn in psutil.net_connections(kind='inet'):  # Direkte Netzwerkverbindungen prüfen
            if conn.laddr.port == 5000:  # Überprüfe, ob der Prozess Port 5000 nutzt
                pid = conn.pid
                if pid:  # Überprüfen, ob PID vorhanden ist
                    proc = psutil.Process(pid)
                    print(f"Beende Backend-Prozess mit PID {pid}, der Port 5000 nutzt.")
                    proc.terminate()
                    proc.wait(timeout=5)  # Warten, bis der Prozess beendet ist
                    return True
    except Exception as e:
        print(f"Fehler beim Beenden des Backends: {e}")
    print("Kein Backend-Prozess gefunden, der auf Port 5000 läuft.")
    return False

def wait_until_backend_stopped():
    """
    Wartet, bis kein Prozess mehr auf Port 5000 läuft.
    """
    while True:
        backend_running = any(
            conn.laddr.port == 5000 for conn in psutil.net_connections(kind='inet')
        )
        if backend_running:
            print("Backend läuft noch, warte...")
            time.sleep(1)
        else:
            print("Backend erfolgreich gestoppt.")
            break

def start_backend():
    """
    Startet das Backend.
    """
    print("Backend wird gestartet...")
    backend_path = "./Use_Case_1/Use_Case_1.2/backend/backend.py"
    try:
        Popen(["python", backend_path], stdout=sys.stdout, stderr=sys.stderr)
        time.sleep(3)  # Kurz warten, bis das Backend hochgefahren ist

        # Überprüfung, ob der Server läuft
        try:
            requests.get(BASE_URL, timeout=5)
            print("Backend erfolgreich gestartet.")
        except requests.ConnectionError:
            print("Fehler: Backend konnte nicht gestartet werden.")
    except Exception as e:
        print(f"Fehler beim Starten des Backends: {e}")

def show_commands():
    """
    Zeigt die verfügbaren Befehle an.
    """
    print("\nVerfügbare Befehle:")
    print("  - Normale Eingaben für den Chat")
    print("  - `/book` für Buch-Generierung (min_chapter und min_subchapter)")
    print("  - `/search` für eine Suche")
    print("  - `/save` um den gesamten Chat zu speichern")
    print("  - `/clear` um Backend zu stoppen, Dateien zu löschen und neu zu starten")
    print("  - `/help` zeigt diese Befehlsliste erneut an")
    print("  - `/exit` um Backend zu stoppen und den Chat zu beenden\n")

def chat():
    try:
        # Logdatei löschen
        log_path = "./Use_Case_1/Use_Case_1.2/backend/backend.log"
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"Logdatei {log_path} gelöscht.")

        # Chroma Storage Ordner löschen
        chroma_storage_path = "./Use_Case_1/Use_Case_1.2/backend/chroma_storage"
        if os.path.exists(chroma_storage_path):
            shutil.rmtree(chroma_storage_path)
            print(f"Chroma Storage Ordner {chroma_storage_path} gelöscht.")

    except Exception as e:
        print(f"Fehler beim Bereinigen und Neustarten: {e}")

    start_backend()

    print("\nWillkommen zum Book-AI!")
    show_commands()

    chat_history = []  # Liste zum Speichern des Chatverlaufs

    while True:
        user_input = input("Du: ")
        if user_input.startswith("/book"):
            print("Buch-Generierung gestartet.")
            min_chapter = input("Minimale Kapitelzahl (Standard 0): ").strip() or "0"
            min_subchapter = input("Minimale Unterkapitelzahl (Standard 0): ").strip() or "0"
            prompt = input("Gib deinen Buch-Prompt ein: ")
            payload = {
                "user_input": prompt,
                "min_chapter": int(min_chapter),
                "min_subchapter": int(min_subchapter)
            }
            try:
                response = requests.post(f"{BASE_URL}/generate", json=payload)
                result = response.json()
                print(f"KI: {json.dumps(result, indent=4, ensure_ascii=False)}")
                chat_history.append({"user": prompt, "bot": result})  # Chatverlauf speichern
            except Exception as e:
                print(f"Fehler bei der Anfrage: {e}")

        elif user_input.startswith("/search"):
            print("Suche gestartet.")
            query = input("Gib deinen Suchbegriff ein: ")
            payload = {"request_input": query}
            try:
                response = requests.post(f"{BASE_URL}/search", json=payload)
                result = response.json()
                print(f"Suchergebnisse: {json.dumps(result, indent=4, ensure_ascii=False)}")
                chat_history.append({"user": query, "bot": result})  # Chatverlauf speichern
            except Exception as e:
                print(f"Fehler bei der Suche: {e}")

        elif user_input.startswith("/save"):
            print("Speichern des Chatverlaufs gestartet.")
            filename = input("Wie soll die Datei heißen (z.B. chatlog.json): ").strip()
            if not filename.endswith(".json"):
                filename += ".json"
            try:
                payload = {
                    "chat_history": chat_history,
                    "filename": filename
                }
                response = requests.post(f"{BASE_URL}/save_chat", json=payload)
                result = response.json()
                print(f"Speicherstatus: {result.get('message', 'Unbekannter Status')}")
            except Exception as e:
                print(f"Fehler beim Speichern: {e}")

        elif user_input.startswith("/clear"):
            print("Backend wird gestoppt und bereinigt...")
            if stop_backend():
                wait_until_backend_stopped()

            try:
                # Logdatei löschen
                log_path = "./Use_Case_1/Use_Case_1.2/backend/backend.log"
                if os.path.exists(log_path):
                    os.remove(log_path)
                    print(f"Logdatei {log_path} gelöscht.")

                # Chroma Storage Ordner löschen
                chroma_storage_path = "./Use_Case_1/Use_Case_1.2/backend/chroma_storage"
                if os.path.exists(chroma_storage_path):
                    shutil.rmtree(chroma_storage_path)
                    print(f"Chroma Storage Ordner {chroma_storage_path} gelöscht.")
                
                # `chat_history` zurücksetzen
                chat_history.clear()
                print("Chat-Verlauf wurde geleert.")

                # Backend neu starten
                start_backend()
            except Exception as e:
                print(f"Fehler beim Bereinigen und Neustarten: {e}")

        elif user_input.startswith("/exit"):
            print("Backend und Chat beenden...")
            if stop_backend():
                wait_until_backend_stopped()

            try:
                # Logdatei löschen
                log_path = "./Use_Case_1/Use_Case_1.2/backend/backend.log"
                if os.path.exists(log_path):
                    os.remove(log_path)
                    print(f"Logdatei {log_path} gelöscht.")

                # Chroma Storage Ordner löschen
                chroma_storage_path = "./Use_Case_1/Use_Case_1.2/backend/chroma_storage"
                if os.path.exists(chroma_storage_path):
                    shutil.rmtree(chroma_storage_path)
                    print(f"Chroma Storage Ordner {chroma_storage_path} gelöscht.")

            except Exception as e:
                print(f"Fehler beim Löschen der Ressourcen: {e}")

            print("Chat beendet. Auf Wiedersehen!")
            sys.exit(0)

        elif user_input.startswith("/help"):
            show_commands()

        else:
            payload = {"user_input": user_input}
            try:
                response = requests.post(f"{BASE_URL}/chat", json=payload)
                result = response.json()
                if "final_response" in result:
                    bot_response = result['final_response']
                    print(f"KI: {bot_response}")
                else:
                    bot_response = json.dumps(result, indent=4, ensure_ascii=False)
                    print(f"KI: {bot_response}")
                chat_history.append({"user": user_input, "bot": bot_response})  # Chatverlauf speichern
            except Exception as e:
                print(f"Fehler bei der Anfrage: {e}")


if __name__ == "__main__":
    chat()

