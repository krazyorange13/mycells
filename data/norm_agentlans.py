sentences = []

with open("raw/agentlans_test.txt") as f:
    sentences.extend(f.readlines())

with open("raw/agentlans_train.txt") as f:
    sentences.extend(f.readlines())

sentences = [sentence.strip() for sentence in sentences]

with open("norm/agentlans.txt", "w") as f:
    f.write("\n".join(sentences))
