import numpy as np
from hmmlearn import hmm


class HMMTagger:
    def __init__(self, word_vocab, tag_vocab):
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.vocab_size = len(word_vocab)
        self.n_tag = len(tag_vocab)
        self.transform_probs = np.zeros([self.n_tag, self.n_tag]) * 1.0
        self.emission_probs = np.zeros([self.n_tag, self.vocab_size + 1]) * 1.0  # reserve 1 for <unk>
        self.hidden_prior = np.zeros([self.n_tag]) * 1.0

        # build a inv_tag vocab
        self.inv_tag_vocab = dict({})
        for tag in tag_vocab:
            self.inv_tag_vocab[tag_vocab[tag]] = tag

        # initialize the model
        self.model = hmm.MultinomialHMM(self.n_tag)

    def train(self, sentences: list, pos_tags: list):
        """
        Train the HMMTagger
        :param sentences: a list, containing lists of words
        :param pos_tags: a list, containing lists of tags
        :return: None
        """
        # transform from string to idx
        pos_tags = [[self.tag_vocab[tag] for tag in pos_tag] for pos_tag in pos_tags]
        sentences = [[self.word_vocab[word] for word in sentence] for sentence in sentences]

        # get the hidden_prior_probs
        for pos_tag in pos_tags:
            for tag in pos_tag:
                self.hidden_prior[tag] += 1
        self.hidden_prior = self.hidden_prior / np.sum(self.hidden_prior)

        # get the transfer probs
        for pos_tag in pos_tags:
            for tag0, tag1 in zip(pos_tag[:-1], pos_tag[1:]):
                self.transform_probs[tag0][tag1] += 1
        # smooth
        for tag_idx in range(self.n_tag):
            self.transform_probs[tag_idx] = self.laplace_smoothing(self.transform_probs[tag_idx], self.n_tag)

        # get the emission probs
        for pos_tag, sentence in zip(pos_tags, sentences):
            for tag, word in zip(pos_tag, sentence):
                self.emission_probs[tag][word] += 1
        # smooth
        for tag_idx in range(self.n_tag):
            self.emission_probs[tag_idx] = self.laplace_smoothing(self.emission_probs[tag_idx], self.vocab_size+1)

        # set the model's components
        self.model.startprob_ = self.hidden_prior
        self.model.transmat_ = self.transform_probs
        self.model.emissionprob_ = self.emission_probs

    def predict(self, sequence):
        ids = [self.word_vocab[word] if word in self.word_vocab else self.vocab_size for word in sequence]
        ids = np.array([ids]).reshape([-1, 1])
        logprob, box = self.model.decode(ids)
        returned_tag = [self.inv_tag_vocab[idx] for idx in box.reshape([-1])]
        return returned_tag

    @staticmethod
    def laplace_smoothing(counts, vocab_size):
        assert isinstance(counts, np.ndarray)

        n = np.sum(counts)
        probs = (counts+1) / (n+vocab_size)
        return probs

    @staticmethod
    def good_turing_smoothing(counts):
        k = 5

        nr = np.zeros([1 + k + 1])
        for count in counts:
            if count <= k + 1:
                nr[int(count)] += 1
        r = np.zeros([1 + k])
        for i in range(1, k + 1):
            r[i] = (i + 1) * nr[i + 1] / nr[i]

        sum_candidates = np.sum(counts)
        for i in range(counts.shape[-1]):
            if counts[i] <= k:
                counts[i] = r[int(counts[i])]
        probs = counts / sum_candidates
        p0 = (1 - probs) / nr[1]

        return probs, p0
