import json

def write(message):
    with open('resources/fineTuning/training.json', 'a') as file:
        json.dump(message, file, indent=4)
