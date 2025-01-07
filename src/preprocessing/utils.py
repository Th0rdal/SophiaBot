import json
import os

def load_data(file_path):
    """Lädt Daten aus einer JSON-Datei."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Datei {file_path} nicht gefunden.")
        return []

    print(f"[INFO] Lade Daten aus {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] {len(data)} Artikel geladen.")
    return data