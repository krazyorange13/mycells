import os

import lxml.etree as etree


def find_files(path):
    entries = []
    with os.scandir(path) as results:
        for result in results:
            if result.is_dir():
                entries.extend(find_files(result))
            else:
                entries.append(result)
    return entries


def get_files():
    root = "raw/british_national_corpus/Texts"
    files = find_files(root)
    return files


def get_sentences(tree):
    sentence_strs = []
    sentences = tree.xpath("//s")
    for sentence in sentences:
        words = sentence.xpath("w | c")
        sentence_strs.append(
            "".join([word.text for word in words if word.text is not None])
        )
    return sentence_strs


sentences = []

files = get_files()
print(len(files))
for file in files:
    tree = etree.parse(file.path)
    file_sentences = get_sentences(tree)
    sentences.extend(file_sentences)
    print(end=".", flush=True)


with open("norm/british.txt", "w") as f:
    f.write("\n".join(sentences))
