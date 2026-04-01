import json

with open("raw/sentences.json") as f:
    data = json.load(f)

sentences = [entry["sentence"] for entry in data["data"]]

with open("norm/sentences.txt", "w") as f:
    f.write("\n".join(sentences))
