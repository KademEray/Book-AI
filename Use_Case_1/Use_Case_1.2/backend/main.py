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
    Terminates the backend process running on port 5000.
    This function checks for any network connections using port 5000.
    If a process is found using this port, it attempts to terminate the process.
    Returns:
        bool: True if a process using port 5000 was found and terminated, False otherwise.
    Raises:
        Exception: If an error occurs while attempting to terminate the process.
    Prints:
        str: Messages indicating the status of the process termination.
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
    Continuously checks if the backend service running on port 5000 has stopped.
    This function enters an infinite loop where it checks if there is any active
    network connection on port 5000. If such a connection exists, it prints a message
    indicating that the backend is still running and waits for 1 second before checking again.
    Once the connection on port 5000 is no longer found, it prints a message indicating
    that the backend has successfully stopped and exits the loop.
    Note:
        This function uses the `psutil` library to check network connections and the `time`
        library to introduce a delay between checks. Ensure these libraries are installed
        and imported before calling this function.
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
    Starts the backend server by executing the backend.py script.
    This function attempts to start the backend server by running the backend.py script
    located at the specified path. It waits for a short period to allow the server to
    initialize and then checks if the server is running by sending a request to the
    BASE_URL.
    Raises:
        Exception: If there is an error while starting the backend.
    Prints:
        "Backend wird gestartet...": When the backend start process begins.
        "Backend erfolgreich gestartet.": If the backend starts successfully.
        "Fehler: Backend konnte nicht gestartet werden.": If the backend fails to start.
        "Fehler beim Starten des Backends: {e}": If there is an exception during the start process.
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
    Displays a list of available commands for the chat application.
    Commands:
        - Normal inputs for the chat
        - `/book` for book generation (requires min_chapter and min_subchapter)
        - `/search` for a search
        - `/save` to save the entire chat
        - `/clear` to stop the backend, delete files, and restart
        - `/help` to display this list of commands again
        - `/exit` to stop the backend and end the chat
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
    """
    Starts the chat application, handles user commands, and interacts with the backend.
    The function performs the following tasks:
    - Deletes the log file and Chroma storage folder if they exist.
    - Starts the backend service.
    - Displays a welcome message and available commands.
    - Enters a loop to process user inputs and commands.
    Commands:
    - /book: Starts book generation with user-defined parameters.
    - /search: Initiates a search with a user-provided query.
    - /save: Saves the chat history to a specified JSON file.
    - /clear: Stops the backend, clears logs, storage, and chat history, then restarts the backend.
    - /exit: Stops the backend, clears logs and storage, and exits the chat application.
    - /help: Displays available commands.
    For other inputs, it sends the input to the backend chat endpoint and displays the response.
    Exceptions:
    - Catches and prints exceptions that occur during file operations, backend interactions, and HTTP requests.
    """
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

