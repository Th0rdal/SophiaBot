# preprocessing_main.py
from download_wiki import fetch_wikipedia_articles
from clean import clean_text, save_cleaned_data
from likelihood import *
from utils import load_data

wiki_category = "Ancient Rome" #category des wikipedia downloads
top_article_count = 5 #anzahl der artikel auf die gefiltert werden soll

def main():
    # Beispiel: Wikipedia-Daten herunterladen
    fetch_wikipedia_articles(wiki_category, max_articles=80, language="en")
    print(f"=" * 60)

    # ---------------------------------------------------------------------------------
    input_file = f"./raw/{wiki_category}_dump.json"
    output_file = f"./processed/{wiki_category}_cleaned.json"

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

    cleaned_file_path = f"./processed/{wiki_category}_cleaned.json"
    articles = load_cleaned_data(cleaned_file_path)

    # Prompt definieren
    prompt = "What are important kinds of philosophy?"


    filtered_articles = filter_titles_by_prompt(articles, prompt, top_n=top_article_count)

    # Ergebnisse anzeigen
    print(f"[INFO] Top {top_n} Artikel basierend auf Titeln:")
    for article, score in filtered_articles:
        print(f"Title: {article['title']}, Log-Likelihood: {score:.4f}")

    # Artikeltexte als Tokens speichern
    token_output_file = f"./processed/{wiki_category}_tokenized.json"
    save_articles_as_tokens([article for article, _ in filtered_articles], token_output_file)

if __name__ == "__main__":
    main()
