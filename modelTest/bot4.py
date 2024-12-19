from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat():
    print("Chat with GPT-2 (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Encode the input and generate a response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,  # Explicitly set input_ids
                max_length=100,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Bot: {response}")


# Start the chat
if __name__ == '__main__':
    chat()
