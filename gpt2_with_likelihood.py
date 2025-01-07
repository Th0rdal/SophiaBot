import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

class AI:
    def __init__(self):
        self.modelName = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Explizit Pad-Token setzen

        self.temperature = 1.0
        self.maxLength = 100
        self.top_k = 50
        self.top_p = 0.9
        self.repetitionPenalty = 1.2
        self.doSample = True

    def load_cleaned_articles(self, file_path):
        """Lädt bereinigte Wikipedia-Artikel."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def calculate_title_log_likelihood(self, prompt, title):
        """Berechnet den Log-Likelihood-Score basierend auf dem Titel."""
        input_text = f"{prompt} {title}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        log_likelihood = -loss.item() * input_ids.size(1)
        return log_likelihood

    def filter_articles_by_query(self, articles, query, top_n=5):
        """Filtert die Artikel basierend auf dem Query."""
        scores = []
        for article in articles:
            title = article["title"]
            score = self.calculate_title_log_likelihood(query, title)
            scores.append((article, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_articles = scores[:top_n]

        # Zeige die Titel der gefilterten Artikel an
        print(f"[INFO] Gefilterte Artikel basierend auf dem Query '{query}':")
        for article, score in top_articles:
            print(f"  - {article['title']} (Score: {score:.4f})")

        return top_articles

    def truncate_context(self, context, max_tokens):
        """Trunkiert den Kontext basierend auf der maximalen Tokenanzahl."""
        tokens = self.tokenizer.encode(context, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_responses(self, query, articles_path, num_responses=2):
        """Generiert mehrere Antworten basierend auf dem Query und Artikeln."""
        articles = self.load_cleaned_articles(articles_path)
        filtered_articles = self.filter_articles_by_query(articles, query, top_n=3)
        context = " ".join([article["text"] for article, _ in filtered_articles])

        query_tokens = self.tokenizer.encode(query, truncation=True)
        max_context_tokens = self.maxLength - len(query_tokens)
        context = self.truncate_context(context, max_context_tokens)
        input_text = f"{query}\n\n{context}"

        responses = []
        temperature_variation = [self.temperature, self.temperature + 0.5]
        top_k_variation = [self.top_k, self.top_k + 20]
        top_p_variation = [self.top_p, self.top_p - 0.1]
        max_new_tokens_variation = [50, 70]

        for i in range(num_responses):
            torch.manual_seed(42 + i)  # Unterschiedlicher Seed für jede Antwort
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.maxLength)
            input_ids = inputs.input_ids

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens_variation[i],
                    do_sample=self.doSample,
                    temperature=temperature_variation[i],
                    top_k=top_k_variation[i],
                    top_p=top_p_variation[i],
                    repetition_penalty=self.repetitionPenalty,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            responses.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        return responses

    def answer_with_choice(self, query, articles_path):
        """Generiert zwei Antworten und fragt den Nutzer nach der besseren."""
        responses = self.generate_responses(query, articles_path, num_responses=2)

        print("\nResponse 1:")
        print(responses[0])
        print("\nResponse 2:")
        print(responses[1])

        choice = input("Which response is better? (1/2): ").strip()
        chosen_response = responses[0] if choice == "1" else responses[1]
        print("\nChosen Response:")
        print(chosen_response)

        return chosen_response


if __name__ == "__main__":
    ai = AI()
    articles_path = "./resources/processed/Ancient Rome_cleaned.json"

    # Query per Hand eingeben
    query = input("Please enter your query: ")

    # Generiere Antworten und lasse den Nutzer auswählen
    ai.answer_with_choice(query, articles_path)
