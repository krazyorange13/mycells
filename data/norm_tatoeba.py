with open("raw/eng_sentences.tsv") as f:
    lines = f.readlines()

sentences = []
for line in lines:
    sentences.append(" ".join(line.split()[2:]))

with open("norm/tatoeba.txt", "w") as f:
    f.write("\n".join(sentences))
