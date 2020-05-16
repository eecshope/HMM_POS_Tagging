import _pickle as pkl
from data.data_loader import DataSet


def test_func(sentences, pos_tags):
    count = 0
    correct = 0
    global model

    for sentence, tag in zip(sentences, pos_tags):
        pred = model.predict(sentence)
        count += len(sentence)
        for pred_tag, true_tag in zip(pred, tag):
            if pred_tag == true_tag:
                correct += 1

    return correct/count


with open('parameters/params.pkl', 'rb') as file:
    model = pkl.load(file)
    print(type(model))

train_set = DataSet('hmm-dataset/train.txt')
print("Training acc is {}".format(test_func(train_set.sentences, train_set.pos_tags)))

test_set = DataSet('hmm-dataset/test.txt')
print("Test acc is {}".format(test_func(test_set.sentences, test_set.pos_tags)))
