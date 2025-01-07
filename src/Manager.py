from datetime import datetime

import dataManager
from gpt2 import AI
import random, os

class Manager:

    def __init__(self):
        self.ai1 = None
        self.lastTest = None
        self.ai2 = None
        self.ai = AI()

        self.dataPath = "../resources/fineTuning/training.json"
        self.getTrainingDataProbability = 1
        self.getTrainingDataFlag = False
        self.trainingThreshold = 10
        self.trainingDataCounter = dataManager.count(self.dataPath)
        self.answerList = []

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
        if random.random() < self.getTrainingDataProbability:
            return self.train(text)
        else:
            return self.ai.answer(text)

    def train(self, text):
        self.getTrainingDataFlag = True
        self.lastTest = text
        self.ai1 = self.ai.answer(text)
        self.ai2 = self.ai.answer(text)
        self.answerList.append(self.ai1)
        self.answerList.append(self.ai2)
        result = "Please choose between the 2 options below:\n"
        result += "Option 1:\n" + self.ai1 + "\n"
        result += "Option 2:\n" + self.ai2 + "\n"
        return result

    def trainAI(self):
        self.ai.train()
        self.ai.load()
        os.rename(self.dataPath, self.dataPath.rsplit('/', 1)[0] + "/" + datetime.now().strftime("%Y-%m-%d") + "_training.json")
        open(self.dataPath, 'w').close()

if __name__ == '__main__':
    m = Manager()
    print(m.answer("Was ist der unterschied zwischen sonne und mond"))
    print(m.answer("1"))