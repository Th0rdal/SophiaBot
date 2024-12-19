import trainingWriter
from gpt2 import AI
import random

class Manager:

    def __init__(self):
        self.ai1 = None
        self.lastTest = None
        self.ai2 = None
        self.ai = AI()
        self.trainProbability = 1
        self.trainFlag = False
        self.answerList = []

    def answer(self, text):
        if self.trainFlag:
            data = {"prompt": self.lastTest, "ai1": self.ai1, "ai2": self.ai2}
            if text == "1":
                data["correct_answer"] = "ai1"
                trainingWriter.write(data)
            elif text == "2":
                data["correct_answer"] = "ai2"
                trainingWriter.write(data)
            self.trainFlag = False
        if random.random() < self.trainProbability:
            return self.train(text)
        else:
            return self.ai.answer(text)

    def train(self, text):
        self.trainFlag = True
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
        #TODO implementing
        self.ai.train()

if __name__ == '__main__':
    m = Manager()
    print(m.answer("Was ist der unterschied zwischen sonne und mond"))
    print(m.answer("1"))