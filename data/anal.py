import matplotlib.pyplot as plt


def load_lengths(path):
    with open(path) as f:
        sentences = f.readlines()
    lengths = [len(sentence) for sentence in sentences]
    return lengths


lengths = []
lengths.extend(load_lengths("norm/tatoeba.txt"))
# lengths.extend(load_lengths("norm/agentlans.txt"))
# lengths.extend(load_lengths("norm/sentences.txt"))
# lengths.extend(load_lengths("norm/harvsents.txt"))
# lengths.extend(load_lengths("norm/british.txt"))

plt.title("sentence lengths")
plt.xlabel("length")
plt.ylabel("frequency")
plt.hist(lengths, bins=list(range(0, 256 + 1, 16)), zorder=3)
plt.xticks(list(range(0, 256 + 1, 16)))
plt.yticks(list(range(0, 1000000 + 1, 50000)))
plt.grid(zorder=0)

plt.show()
