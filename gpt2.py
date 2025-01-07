from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AI:
    def __init__(self):
        self.modelName = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)

        self.temperature = 1.0
        self.maxLength = 100
        self.top_k = 50
        self.top_p = 0.9
        self.repetitionPenalty = 1.2
        self.doSample = True

    def answer(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
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

    def train(self):
        #TODO implementing
        pass


# Start the chat
if __name__ == '__main__':
    ai = AI()
    ai.chat()
