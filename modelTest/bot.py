from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator

# Step 1: Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-V2.5-1210"
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Prepare the dataset
raw_data = [
    {"text": "The product is amazing!", "label": 1},
    {"text": "I didn't like the movie.", "label": 0},
    {"text": "The service was okay.", "label": 2}
]
dataset = Dataset.from_list(raw_data)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Split the dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Step 3: Create a DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

# Step 4: Training Loop with Accelerate
accelerator = Accelerator()

# Prepare model, optimizer, and dataloaders
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, train_dataloader, eval_dataloader = accelerator.prepare(
    model, train_dataloader, eval_dataloader
)

# Training loop
for epoch in range(3):  # Number of epochs
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        # The `DataCollatorWithPadding` ensures tensors are prepared, so no need to convert manually
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        # Print loss for every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")