#gpt2.py
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from transformers import Trainer
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys


# Projekt-Root relativ zu diesem Skript hinzufügen
project_root = Path(__file__).parent.parent

# Importiere aus preprocessing
from preprocessing.preprocessing_main import wiki_category

# Relativer Pfad zur Datei
file_path_training = project_root / "resources" / "fineTuning" / "training.json"
file_path_cleaned_articles = project_root / "resources" / "processed" / f"{wiki_category}_cleaned.json"

class AI:
    def __init__(self):
        self.modelName = "gpt2"
        self.save_path = "./" + self.modelName + "-finetuned"
        self.prePromt = "Always give all sources used"
        self.save_path = project_root / "src" / f"{self.modelName}-finetuned"

        self.load()

        self.temperature = 1.0
        self.maxLength = 100
        self.top_k = 50
        self.top_p = 0.9
        self.repetitionPenalty = 1.2
        self.doSample = True
        self.max_new_tokens = 10

    def load(self):
        if os.path.exists(self.save_path):
            print("Found saved model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.save_path, output_attentions=True)
        else:
            print("No saved model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
            self.model = AutoModelForCausalLM.from_pretrained(self.modelName, output_attentions=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer(self, query):
        articles = self.load_cleaned_articles(file_path_cleaned_articles)

        # Filtere relevante Artikel
        filtered_articles = self.filter_articles_by_query(articles, query, top_n=3)
        context = " ".join([article["text"] for article, _ in filtered_articles])

        # Berechne verfügbare Länge für den Kontext
        query_tokens = self.tokenizer.encode(query, truncation=True)
        max_context_tokens = self.maxLength - len(query_tokens) - 2  # 2 für Trennzeichen (\n\n)
        context = self.truncate_context(context, max_context_tokens)

        # Kombiniere Query mit Kontext
        input_text = f"{query}\n\n{context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Sicherheitscheck für Eingabelänge
        if input_ids.size(1) > self.maxLength:
            raise ValueError(f"Input length exceeds max_length ({self.maxLength}). Adjust query or context.")

        # Modellgenerierung
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,  # Anzahl der Tokens, die zusätzlich generiert werden
                do_sample=self.doSample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetitionPenalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    # this should not be here
    def chat(self):
        print("Chat with GPT-2 (type 'exit' to stop):")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            # Encode the input and generate a response
            response = self.answer(user_input)
            print(f"Bot: {response}")

    def showOutputProbability(self, prompt, maxWords=10, maxPossibilitesPerWord=5):

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate tokens step by step and explain each one
        output_tokens = input_ids
        for _ in range(maxWords):  # Generate up to 10 additional tokens
            with torch.no_grad():
                outputs = self.model(output_tokens)
                logits = outputs.logits

            # Convert logits to probabilities
            probs = torch.softmax(logits[:, -1, :], dim=-1)[0]

            # Get top 5 predictions
            top_k = maxPossibilitesPerWord
            top_probs, top_indices = torch.topk(probs, top_k)

            # Display top predictions
            print("\nTop predictions for the next word:")
            for i in range(top_k):
                token = self.tokenizer.decode(top_indices[i].item())
                probability = top_probs[i].item()
                print(f"{token}: {probability:.2%}")

            # Select the most likely next token (you can change this logic if needed)
            next_token = top_indices[0].unsqueeze(0).unsqueeze(0)

            # Append next token to the output
            output_tokens = torch.cat((output_tokens, next_token), dim=1)

    def showAttentionWeights(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Forward pass
        outputs = self.model(input_ids)
        # Extract attention weights from the last layer
        attentions = outputs.attentions[-1][0][0].detach().numpy()

        # Decode the tokens back to words
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = [token.replace("Ġ", "") for token in tokens]  # Remove 'Ġ' from tokens

        # Plot the attention heatmap with token labels
        plt.figure(figsize=(10, 8))
        # Select the first head (if using multi-head attention)
        plt.matshow(attentions, cmap="viridis")

        plt.colorbar()
        plt.title("Attention Weights (Last Layer)")

        # Set token labels on X and Y axes
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.show()

    def showOutputWithLime(self):
        pass

    def train(self, dataset_name="wikitext", split="train", epochs=1, batch_size=8):
        # Load the dataset from the JSON file
        with open(file_path_training, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert the JSON data to a HuggingFace Dataset
        dataset = Dataset.from_dict({
            "prompt": [sample["prompt"] for sample in data],
            "correct_answer": [sample[sample["correct_answer"]] for sample in data]
        })

        # Tokenize the dataset
        def tokenize_function(examples):
            inputs = [
                f"{prompt} {answer}"
                for prompt, answer in zip(examples["prompt"], examples["correct_answer"])
            ]
            tokenized = self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=128
            )
            tokenized["labels"] = torch.tensor(tokenized["input_ids"].copy(), dtype=torch.long)
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.save_path),
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            save_total_limit=2,
            save_steps=500,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="no"
        )

        # Use HuggingFace's Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Save the model after training
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model saved to {self.save_path}")

    #private functions for internal calculations
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

    def filter_articles_by_query(self, articles, query, top_n=3):
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

# Start the chat
if __name__ == '__main__':
    ai = AI()
    #ai.chat()
    #ai.train(epochs=3, batch_size=4)
    #ai.showOutputProbability("The weather is")
    ai.showAttentionWeights("The weather today is nice!")