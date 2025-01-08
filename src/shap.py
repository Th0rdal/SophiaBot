from lime.lime_text import LimeTextExplainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import shap

# Load your fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("C:\\Users\\patrick\\PycharmProjects\\SophiaBot\\gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("C:\\Users\\patrick\\PycharmProjects\\SophiaBot\\gpt2-finetuned")


# Define a SHAP explainer
explainer = shap.Explainer(model)

# Encode input text
input_text = "Artificial intelligence is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Explain the prediction
shap_values = explainer(input_ids)

# Visualize the explanation
shap.plots.text(shap_values)