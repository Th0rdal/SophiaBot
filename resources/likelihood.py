import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# GPT-2 Modell und Tokenizer laden
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def load_cleaned_data(file_path):
    """Lädt die bereinigten Artikel und Titel aus einer JSON-Datei."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] {len(data)} Artikel aus {file_path} geladen.")
    return data


def calculate_title_log_likelihood(prompt, title):
    """Berechnet den Log-Likelihood-Score basierend auf dem Titel."""
    # Kombiniere Prompt und Titel
    input_text = f"{prompt} {title}"

    # Tokenisierung
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids

    # Berechne die Logits und den Loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-Entropy Loss

    # Log-Likelihood = Negative des Loss multipliziert mit der Tokenanzahl
    log_likelihood = -loss.item() * input_ids.size(1)
    return log_likelihood


def filter_titles_by_prompt(articles, prompt, top_n=5):
    """Filtert Artikel basierend auf der Relevanz des Titels zum Prompt."""
    scores = []
    for article in articles:
        title = article["title"]
        score = calculate_title_log_likelihood(prompt, title)
        scores.append((article, score))

    # Sortiere nach Score (höher = besser)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]  # Gib die Top-N-Artikel zurück


def main():
    # Bereinigte Daten laden
    cleaned_file_path = "./processed/History_cleaned.json"
    articles = load_cleaned_data(cleaned_file_path)

    # Prompt definieren
    prompt = "Describe historical events:"

    # Titel-basiertes Filtern durchführen
    top_n = 5
    filtered_articles = filter_titles_by_prompt(articles, prompt, top_n=top_n)

    # Ergebnisse anzeigen
    print(f"[INFO] Top {top_n} Artikel basierend auf Titeln:")
    for article, score in filtered_articles:
        print(f"Title: {article['title']}, Log-Likelihood: {score:.4f}")


if __name__ == "__main__":
    main()
