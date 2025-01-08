#Manager.py
import math
from datetime import datetime

from dill.source import indent

import dataManager
from gpt2 import AI
import random, os
from pathlib import Path
from preprocessing.preprocessing_main import preprocessing_main

import json

from Explanationtype import Explanationtype

# Projekt-Root relativ zu diesem Skript hinzufügen
project_root = Path(__file__).parent.parent
training_path = project_root / "resources" / "fineTuning" / "training.json"
config_path = project_root / "config.json"


def calculateDecayRate(initialValue, targetValue, count):
    return - (math.log(targetValue / initialValue)/count)


class Manager:

    def __init__(self):
        self.ai1 = None
        self.lastTest = None
        self.ai2 = None
        self.ai = AI()

        with open(config_path) as json_file:
            data = json.load(json_file)
        self.dataPath = training_path
        self.getTrainingDataProbability = data["get_training_data_probability"]
        self.getTrainingDataFlag = False
        self.trainingThreshold = data["training_threshold"]
        self.trainingDataCounter = dataManager.count(self.dataPath)
        self.answerList = []

        self.explanationType = Explanationtype(4)
        self.maxWords = 10
        self.maxPossibilitiesPerWord = 5

        self.targetProbability = data["target_probability"]
        self.initialProbability = data["initial_probability"]
        self.count = data["training_count"]
        self.decayRate = data["decay_rate"]

    def answer(self, text):
        if self.getTrainingDataFlag:
            data = {"prompt": self.lastTest, "ai1": self.ai1, "ai2": self.ai2}
            if text == "1":
                data["correct_answer"] = "ai1"
                dataManager.write(data, self.dataPath)
                self.trainingDataCounter += 1
            elif text == "2":
                data["correct_answer"] = "ai2"
                dataManager.write(data, self.dataPath)
                self.trainingDataCounter += 1
            self.getTrainingDataFlag = False
            if self.trainingDataCounter >= self.trainingThreshold:
                self.trainAI()
        else:
            if random.random() < self.getTrainingDataProbability:
                return self.train(text), True
            else:
                return self.ai.answer(text), False

    def train(self, text):
        self.getTrainingDataFlag = True
        self.lastTest = text
        self.ai1 = self.ai.answer(text)
        self.ai2 = self.ai.answer(text)
        self.answerList.append(self.ai1)
        self.answerList.append(self.ai2)
        result = ""
        result += "--------------------------------------------------------------\nAnswer Option 1:\n" + self.ai1 + "\n"
        result += "--------------------------------------------------------------\nAnswer Option 2:\n" + self.ai2 + "\n"
        return result

    def trainAI(self):
        self.ai.train()
        self.ai.load()
        # Erstelle neuen Pfad mit Datum
        new_path = self.dataPath.parent / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_training.json"
        # Verschiebe die Datei
        os.rename(self.dataPath, new_path)
        # Initialisiere eine leere JSON-Datei
        with open(self.dataPath, 'w') as file:
            json.dump([], file, indent=4)
        self.calculateDelay()

    def calculateDelay(self):
        self.count += 1
        self.getTrainingDataProbability = max(self.initialProbability * math.exp(self.count * self.decayRate * -1), self.targetProbability)
        with open(config_path, "r+") as file:
            data = json.load(file)
            file.seek(0)
            file.truncate(0)
            data["get_training_data_probability"] = self.getTrainingDataProbability
            data["training_count"] = self.count
            file.write(json.dumps(data, indent=4))

    def explain(self, prompt):
        answer = self.ai.answer(prompt)
        if self.explanationType == Explanationtype.ALL:
            self.ai.explain(prompt, self.maxWords, self.maxPossibilitiesPerWord)
        elif self.explanationType == Explanationtype.OUTPUT_PROBABILITY:
            self.ai.showOutputProbability(prompt, self.maxWords, self.maxPossibilitiesPerWord)
        elif self.explanationType == Explanationtype.ATTENTION_WEIGHTS:
            self.ai.showAttentionWeights(prompt)
        elif self.explanationType == Explanationtype.LIME:
            self.ai.showOutputWithLime(prompt)
        return answer, False

    def resetConfig(self):
        data = {
                "get_training_data_probability": 1,
                "training_threshold": 10,
                "target_probability": 0.05,
                "initial_probability": 1,
                "training_count": 0,
                "decay_rate": 0.009986
        }
        with open(config_path, "w") as file:
            json.dump(data, file, indent=4)

if __name__ == '__main__':
    preprocessing_main()
    print("Finished preprocessing")

    m = Manager()
    m.answer("was ist rom")
    m.answer("1")
    exit(0)

    aiCallFunction = m.answer
    explType = None

    userinput = input("Should AI output explanation be added? (y/n):\n")
    if userinput == "y" or userinput == "Y" or userinput == "yes" or userinput == "Yes": # add explanation to responses
        explanationTypeText = "What type of output explanation should be added?:\nOutput probability (1)\nAttention weights (2)\nLime (3)\nAll of the above (4)\n"
        while True:
            userinput = input(explanationTypeText)
            try:
                explType = Explanationtype(int(userinput))
            except ValueError:
                print("Invalid input. Please try again.")
                continue
            break
        if explType == Explanationtype.ALL or explType == Explanationtype.OUTPUT_PROBABILITY:
            while True:
                words = input("Please enter the maximum number of words to predict:")
                poss = input("Please enter the maximum possibilities per word:")
                try:
                    m.maxWords = int(words)
                    m.maxPossibilitiesPerWord = int(poss)
                except ValueError:
                    print("One or both of the values was/were not an integer. Please try again.")
                    continue
                break
        m.explanationType = explType
        aiCallFunction = m.explain
    while True:
        userinput = input("Ask the AI a question:\n")
        if userinput == "END":
            break
        answer, isTraining = aiCallFunction(userinput)
        print(answer)
        if isTraining:
            userinput = input("Feedback: Please choose between the 2 options:\n")
            if userinput != "1" and userinput != "2":
                print("None of the options was chosen. Data collection for training failed.\n")
                continue
            aiCallFunction(userinput)

        print("\n")