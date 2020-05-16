from model import hmm_model
from data import data_loader
import _pickle as pkl

dataset = data_loader.DataSet('hmm-dataset/train.txt')
word_vocab = dataset.build_vocab()
tag_vocab = dataset.build_vocab(False)
model = hmm_model.HMMTagger(word_vocab, tag_vocab)

model.train(dataset.sentences, dataset.pos_tags)

with open('parameters/params.pkl', 'wb') as file:
    pkl.dump(model, file)
