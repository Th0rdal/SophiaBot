from langchain_ollama import OllamaLLM



model = OllamaLLM(model="llama3")

while True:
    inputText = input("Enter a prompt: ")
    result = model.invoke(input=inputText)
    print(result)