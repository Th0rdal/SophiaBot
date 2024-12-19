from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier(["THANKS"])
for result in results:
    print(f"{result}")


import torch
speech = pipeline("automatic-speech-recognition-v2", model="faceb")