#dataManager.py

import json

def write(message, path):
    try:
        # Lade bestehende Daten
        if not path.exists() or path.stat().st_size == 0:
            data = []  # Initialisiere leeres Array, wenn Datei leer ist
        else:
            with open(path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("Die JSON-Datei enthält kein Array.")

        # Füge neue Nachricht hinzu
        data.append(message)

        # Schreibe die aktualisierte Datei
        with open(path, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    except json.JSONDecodeError:
        print("Fehler: Die Datei ist keine gültige JSON-Datei. Initialisiere sie neu.")
        with open(path, 'w') as file:
            json.dump([message], file, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")

def count(path):
    count = 0
    with open(path, 'r') as file:
        # Load the file as a JSON array
        data = json.load(file)
        if isinstance(data, list):
            count = len(data)
        else:
            raise ValueError("The JSON file does not contain an array.")
    return count
