# preprocessing_main.py
from .download_wiki import fetch_wikipedia_articles
from .clean import clean_text, save_cleaned_data
from .utils import load_data
from pathlib import Path
import os

wiki_category = "Ancient Rome" #category des wikipedia downloads

# Projekt-Root relativ zu diesem Skript
project_root = Path(__file__).parent.parent.parent

# Relativer Pfad zur Datei
input_file_for_clean = project_root / "resources" / "raw" / f"{wiki_category}_dump.json"
output_file_after_clean = project_root / "resources" / "processed" / f"{wiki_category}_cleaned.json"
if not os.path.exists(output_file_after_clean.parent):
    os.makedirs(output_file_after_clean.parent)

if not os.path.exists(input_file_for_clean.parent):
    os.makedirs(input_file_for_clean.parent)

def preprocessing_main():

    # Beispiel: Wikipedia-Daten herunterladen
    fetch_wikipedia_articles(wiki_category, max_articles=80, language="en")
    print(f"=" * 60)

    # ---------------------------------------------------------------------------------

    # Lade die Rohdaten
    articles = load_data(input_file_for_clean)
    if not articles:
        print("[ERROR] Keine Artikel zu verarbeiten. Beende das Programm.")
        return

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
    print(f"    [INFO] Ursprüngliche Gesamtzeichenzahl: {total_original_length}")
    print(f"    [INFO] Bereinigte Gesamtzeichenzahl: {total_cleaned_length}")
    print(f"    [INFO] Insgesamt entfernte Zeichen: {total_removed_characters}")

    # Speichere die bereinigten Artikel
    save_cleaned_data(articles, output_file_after_clean)
    print(f"=" * 60)

    # ---------------------------------------------------------------------------------