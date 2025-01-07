import json

def write(message, path):
    with open(path, 'rb+') as file:
        file.seek(0, 2) # set cursor to end of file

        while True:
            file.seek(file.tell() - 1)
            last_char = file.read(1)

            if last_char == b"]":
                break
            file.seek(-1, 1)

        if file.tell() > 2:
            file.seek(file.tell() - 3)
            file.write(b",\n\t")

        for char in json.dumps(message, indent=4).encode('utf-8'):
            file.write(bytes([char]))
            if char == ord('\n'):
                file.write(b"\t")

        file.write(b"\n]")

def count(path):
    count = 0
    with open(path, 'r') as file:
        # Load the file as a JSON array
        data = json.load(file)
        if isinstance(data, list):
            count = len(data)
        else:
            raise ValueError("The JSON file does not contain an array.")
    return count
