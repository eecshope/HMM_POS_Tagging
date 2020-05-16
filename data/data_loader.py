import os
from tqdm import tqdm


class DataSet:
    def __init__(self, data_path):
        assert os.path.exists(data_path)
        file = open(data_path, encoding='utf-8')
        self.sentences = list([])  # list of lists
        self.pos_tags = list([])  # list of lists

        # read data and split the words and the corresponding part of speech
        for line in tqdm(file):
            segments = line.strip().split()  # ['hello/adj', 'i/pron', ...]
            sentence, pos_tag = list([]), list([])
            for segment in segments:
                word, tag = segment.split('/')
                sentence.append(word)
                pos_tag.append(tag)
            self.sentences.append(sentence)
            self.pos_tags.append(pos_tag)

    def build_vocab(self, for_word=True):
        vocab = dict({})
        count = 0
        if for_word:
            for sentence in self.sentences:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = count
                        count += 1
        else:
            for pos_tag in self.pos_tags:
                for tag in pos_tag:
                    if tag not in vocab:
                        vocab[tag] = count
                        count += 1

        return vocab
