with open("norm/tatoeba.txt") as f:
    sentences = f.readlines()


def truncate(sentence):
    words = sentence.split()
    truncated = len(words) // 3
    incomplete = " ".join(words[:-truncated])
    return incomplete


sentences_complete = sentences
sentences_incomplete = [truncate(sentence) for sentence in sentences_complete]

for complete, incomplete in zip(sentences_complete, sentences_incomplete):
    print(complete, "|", incomplete)
