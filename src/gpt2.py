from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from transformers import Trainer
import json
import os

class AI:
    def __init__(self):
        self.modelName = "gpt2"
        self.save_path = "./" + self.modelName + "-finetuned"

        self.load()

        self.temperature = 1.0
        self.maxLength = 100
        self.top_k = 50
        self.top_p = 0.9
        self.repetitionPenalty = 1.2
        self.doSample = True

    def load(self):
        if os.path.exists(self.save_path):
            print("Found saved model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.save_path)
        else:
            print("No saved model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
            self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer(self, query):
        articles = self.load_cleaned_articles("./resources/processed/Ancient Rome_cleaned.json")

        # Filtere relevante Artikel
        filtered_articles = self.filter_articles_by_query(articles, query, top_n=3)
        context = " ".join([article["text"] for article, _ in filtered_articles])

        # Berechne verfügbare Länge für den Kontext
        query_tokens = self.tokenizer.encode(query, truncation=True)
        max_context_tokens = 512 - len(query_tokens)
        context = self.truncate_context(context, max_context_tokens)

        # Kombiniere Query mit Kontext
        input_text = f"{query}\n\n{context}"

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,  # Explicitly set input_ids
                max_length=self.maxLength,
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

    def train(self, dataset_name="wikitext", split="train", epochs=1, batch_size=8):
        # Load the dataset from the JSON file
        with open("C:\\Users\\patrick\\PycharmProjects\\SophiaBot\\resources\\fineTuning\\training.json", "r", encoding="utf-8") as f:
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
            output_dir=self.save_path,
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
    ai.train(epochs=3, batch_size=4)
