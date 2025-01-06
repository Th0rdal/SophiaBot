# preprocessing_main.py
from download_wiki import fetch_wikipedia_articles
from clean import clean_text, save_cleaned_data
from likelihood import load_cleaned_data, calculate_title_log_likelihood, filter_titles_by_prompt
from utils import load_data

def main():
    # Beispiel: Wikipedia-Daten herunterladen
    articles = fetch_wikipedia_articles("History", max_articles=20, language="en")
    print(f"=" * 60)

    # ---------------------------------------------------------------------------------
    input_file = "./raw/History_dump.json"
    output_file = "./processed/History_cleaned.json"

    # Lade die Rohdaten
    articles = load_data(input_file)
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
    save_cleaned_data(articles, output_file)
    print(f"=" * 60)

    # ---------------------------------------------------------------------------------

    # Bereinigte Daten laden
    cleaned_file_path = "./processed/History_cleaned.json"
    articles = load_data(cleaned_file_path)

    # Prompt definieren
    prompt = "Describe historical events:"

    # Titel-basiertes Filtern durchführen
    top_n = 5
    filtered_articles = filter_titles_by_prompt(articles, prompt, top_n=top_n)

    # Ergebnisse anzeigen
    print(f"[INFO] Top {top_n} Artikel basierend auf Titeln:")
    for article, score in filtered_articles:
        print(f"    Title: {article['title']}, Log-Likelihood: {score:.4f}")

if __name__ == "__main__":
    main()
