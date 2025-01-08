#dataManager.py

import json

def write(message, path):
    """
    Writes the message to the training dataset. Expects a json file with at least a [] inside.
    :param message: The feedback data
    :param path: The path to the training data file
    :return: None
    """
    with open(path, 'rb+') as file:
        file.seek(0, 2) # set cursor to end of file

        while True:
            file.seek(file.tell() - 1)
            last_char = file.read(1)

            if last_char == b"]":
                break
            file.seek(-1, 1)

        if file.tell() > 2:
            file.seek(file.tell() - 2)
            file.write(b",\n\t")
        else:
            file.seek(file.tell() - 1)
            file.write(b"\n\t")
        for char in json.dumps(message, indent=4).encode('utf-8'):
            file.write(bytes([char]))
            if char == ord('\n'):
                file.write(b"\t")

        file.write(b"\n]")

def count(path):
    """
    Counts the feedback data in the training file
    :param path: The path to the training data file
    :return: The number of feedback data in the training file
    """
    count = 0
    with open(path, 'r') as file:
        # Load the file as a JSON array
        data = json.load(file)
        if isinstance(data, list):
            count = len(data)
        else:
            raise ValueError("The JSON file does not contain an array.")
    return count
