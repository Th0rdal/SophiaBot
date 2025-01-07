import json
import re
import os
from utils import *

def clean_text(text):
    """Bereinigt den Text: entfernt unnötige Whitespaces und Sonderzeichen."""
    # Entferne überflüssige Whitespaces
    text = re.sub(r"\s+", " ", text)
    # Entferne Sonderzeichen (außer Satzzeichen)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)
    # Trimme Leerzeichen am Anfang/Ende
    return text.strip()


def save_cleaned_data(articles, output_file):
    """Speichert die bereinigten Artikel in einer JSON-Datei."""
    print(f"[INFO] Speichere bereinigte Daten in {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)



def main():
    input_file = "./raw/History_dump.json"
    output_file = "./processed/History_cleaned.json"

    # Lade die Rohdaten
    articles = load_data(input_file)
    if not articles:
        print("[ERROR] Keine Artikel zu verarbeiten. Beende das Programm.")
        return

    # Bereinige die Artikeltexte
    print("[INFO] Beginne mit dem Bereinigen der Artikeltexte ...")
    total_original_length = 0
    total_cleaned_length = 0

    for article in articles:
        original_length = len(article["text"])
        article["text"] = clean_text(article["text"])
        cleaned_length = len(article["text"])

        total_original_length += original_length
        total_cleaned_length += cleaned_length

    total_removed_characters = total_original_length - total_cleaned_length
    print(f"[INFO] Alle Artikel wurden erfolgreich bereinigt: {len(articles)} Artikel.")
    print(f"[INFO] Ursprüngliche Gesamtzeichenzahl: {total_original_length}")
    print(f"[INFO] Bereinigte Gesamtzeichenzahl: {total_cleaned_length}")
    print(f"[INFO] Insgesamt entfernte Zeichen: {total_removed_characters}")

    # Speichere die bereinigten Artikel
    save_cleaned_data(articles, output_file)


if __name__ == "__main__":
    main()
