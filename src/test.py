from lime.lime_text import LimeTextExplainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class GPT2LimeWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, texts):
        # Convert the input texts to token IDs
        input_ids = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Get the logits (raw predictions)
        logits = outputs.logits

        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits[:, -1, :], dim=-1)

        # Return probabilities as a list of lists
        return probs.cpu().numpy()

# Load your fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("C:\\Users\\patrick\\PycharmProjects\\SophiaBot\\gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("C:\\Users\\patrick\\PycharmProjects\\SophiaBot\\gpt2-finetuned")

# Create an instance of the wrapper
gpt2_wrapper = GPT2LimeWrapper(model, tokenizer)

# Initialize the LimeTextExplainer
explainer = LimeTextExplainer(class_names=["Generated Text"])

# Input text to explain
text_input = "The future of AI is"

# Use LIME to explain the prediction
exp = explainer.explain_instance(text_input, gpt2_wrapper.predict)


# Option 1: Print the explanation in the terminal
print("\nLIME Explanation (Terminal Output):")
explanation_list = exp.as_list()
for feature, weight in explanation_list:
    print(f"{feature}: {weight:.4f}")

