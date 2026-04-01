with open("raw/harvsents.txt") as f:
    lines = f.readlines()

sentences = []
for line in lines:
    if line.lstrip()[0].isnumeric():
        sentence = " ".join(line.lstrip().split()[1:])
        sentences.append(sentence)

with open("norm/harvsents.txt", "w") as f:
    f.write("\n".join(sentences))
