import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

class AI:
    def __init__(self):
        self.modelName = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Explizit Pad-Token setzen




    def generate_responses(self, query, articles, num_responses=2, max_response_length=10):
        """Generiert mehrere Antworten basierend auf Query und gefilterten Artikeln."""
        # Filtere relevante Artikel
        filtered_articles = self.filter_articles_by_query(articles, query, top_n=3)
        context = " ".join([article["text"] for article, _ in filtered_articles])

        # Berechne verfügbare Länge für den Kontext
        query_tokens = self.tokenizer.encode(query, truncation=True)
        max_context_tokens = 512 - len(query_tokens)
        context = self.truncate_context(context, max_context_tokens)

        # Kombiniere Query mit Kontext
        input_text = f"{query}\n\n{context}"

        # Generiere mehrere Antworten
        responses = []
        for i in range(num_responses):
            torch.manual_seed(42 + i)  # Unterschiedlicher Seed für Diversität
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            input_ids = inputs.input_ids

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_response_length + (5 * i),  # Variiere Länge
                    do_sample=True,
                    temperature=1.0 + (0.5 * i),  # Variiere Temperatur
                    top_k=50 + (10 * i),  # Variiere top_k
                    top_p=0.9 - (0.1 * i),  # Variiere top_p
                    repetition_penalty=1.2 + (0.1 * i),  # Variiere Penalty
                    pad_token_id=self.tokenizer.pad_token_id
                )
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            truncated_output = " ".join(decoded_output.split()[:max_response_length])  # Harte Begrenzung
            responses.append(truncated_output)
        return responses

    def manual_training_with_articles(self, feedback_file, articles_path):
        """Manuelle Eingabe und Auswahl der besten Antworten basierend auf Artikeln."""
        feedback_data = []
        articles = self.load_cleaned_articles(articles_path)

        while True:
            # Eingabe des Querys
            query = input("Enter your query (or type 'exit' to stop): ").strip()
            if query.lower() == "exit":
                break

            # Generiere zwei Antworten basierend auf Artikeln
            responses = self.generate_responses(query, articles, 2, 40)
            print("\nResponse 1:")
            print(responses[0])
            print("\nResponse 2:")
            print(responses[1])

            # Auswahl der besten Antwort oder Eingabe einer eigenen
            choice = input("Choose the best response (1/2) or enter your own response: ").strip()
            if choice == "1":
                selected_response = responses[0]
            elif choice == "2":
                selected_response = responses[1]
            else:
                selected_response = choice  # Manuell eingegebene Antwort

            # Speichern der Trainingsdaten
            feedback_data.append({"query": query, "response": selected_response})
            print("[INFO] Response saved for training.\n")

        # Speichere die gesammelten Feedback-Daten
        with open(feedback_file, "w", encoding="utf-8") as f:
            for entry in feedback_data:
                f.write(json.dumps(entry) + "\n")
        print(f"[INFO] All feedback saved to {feedback_file}.")

if __name__ == "__main__":
    ai = AI()

    feedback_file = "./manual_feedback.jsonl"
    articles_path = "./resources/processed/Ancient Rome_cleaned.json"

    # Starte manuelle Trainingsdaten-Erstellung basierend auf Artikeln
    ai.manual_training_with_articles(feedback_file, articles_path)
