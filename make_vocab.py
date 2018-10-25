from preprocess import *
import nltk.tokenize as tokenizer
import json
import pickle
from sklearn.model_selection import train_test_split


data_folder = 'data/'

def main():
    train_path = "SemEval2010_task8_training/TRAIN_FILE.TXT"
    train_file = open(data_folder + train_path, "r").readlines()
    params = json.load(open("params.json"))

    train_sent, name_labels, dir_labels, e1, e2, p1, p2 = list(), list(), list(), list(), list(), list(), list()

    #processes the train set
    for i in range(8000):
        start = i * 4
        line = train_file[start]
        line = line.split("\t")[1].replace('"',"").replace("\n", "").replace("<", "").replace(">", "").replace("'", "").lower()
        tokens = tokenizer.TreebankWordTokenizer().tokenize(line)

        # gets the position of e1 and e2 words
        e1_clause, e2_clause, e1_s, e1_e, e2_s, e2_e = get_e1_e2(tokens)
        e1.append(e1_clause)
        e2.append(e2_clause)
        train_sent.append(tokens)

        # rel shows the relation and dir shows its direction
        rel = train_file[start + 1].replace("\n", "")
        if rel == "Other":
            name_labels.append("Other")
            dir_labels.append(1)
        else:
            name_labels.append(rel[:rel.find("(")])
            if rel[-3:-1] == "e2":
                dir_labels.append(0)
            else:
                dir_labels.append(1)
        # positional vectors for the sentences regarding e1 and e2
        pos_1, pos_2 = get_pos(tokens, e1_s, e1_e, e2_s, e2_e)
        p1.append(pos_1)
        p2.append(pos_2)

    test_file = open(data_folder + "SemEval2010_task8_testing/TEST_FILE.txt", "r")
    test_sent, test_name_labels, test_dir_labels, test_e1, test_e2, test_p1, test_p2 = list(), list(), list(), list(), list(), list(), list()

    # process the test data
    for l in test_file.readlines():
        line = l.split("\t")[1].replace('"', "").replace("\n", "").replace("<", "").replace(">", "").replace("'", "").lower()
        tokens = tokenizer.TreebankWordTokenizer().tokenize(line)
        test_e1_clause, test_e2_clause, test_e1_s, test_e1_e, test_e2_s, test_e2_e = get_e1_e2(tokens)
        test_e1.append(test_e1_clause)
        test_e2.append(test_e2_clause)
        test_sent.append(tokens)
        test_pos_1, test_pos_2 = get_pos(tokens, test_e1_s, test_e1_e, test_e2_s, test_e2_e)
        test_p1.append(test_pos_1)
        test_p2.append(test_pos_2)

    # gets all the vocabs in train set
    vocabs, chars, params["max_char"], params["max_len"] = get_vocabs_chars(train_sent)
    print("There are", len(vocabs), "words in the dataset")

    # train and test sentences are converted to their word IDs
    train_sent = words_to_id(vocabs, train_sent)
    test_sent = words_to_id(vocabs, test_sent)

    print("Converting tags to numbers")
    # relations are converted to their IDs
    name_labels, tag_dict = tags_to_id(name_labels, set(name_labels))
    labels = [(name, dir) for name, dir in zip(name_labels, dir_labels)]

    params["n_outputs"] = len(tag_dict.values())
    tag_dict = {idx: tag for tag, idx in tag_dict.items()}

    # reads the word embedding file
    if params["pretrain"]:
        print("Loading GloVe pretrained vectors")
        embeddings = read_embedding(vocabs, 'embeddings/glove.300.txt')
    else:
        embeddings = None

    X = np.array(train_sent)
    y = np.array(labels)

    # split train to train and dev
    X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(X, y, range(len(train_sent)),
                                                                                     test_size=0.2, random_state=33)

    json.dump(params, open("params.json", "w"))
    true_dev_labels = [true_label(idx[0], idx[1], tag_dict) for idx in y_dev]
    train_batches = get_batches(X_train, y_train, vocabs.index("<pad>"))
    dev_batches = get_batches(X_dev, y_dev, vocabs.index("<pad>"))
    p1_train = just_batch([p1[idx] for idx in indices_train], pad_idx=1000)
    p2_train = just_batch([p2[idx] for idx in indices_train], pad_idx=1000)
    p1_dev = just_batch([p1[idx] for idx in indices_dev], pad_idx=1000)
    p2_dev = just_batch([p2[idx] for idx in indices_dev], pad_idx=1000)

    p1_test = just_batch(test_p1, pad_idx=1000)
    p2_test = just_batch(test_p2, pad_idx=1000)
    test_batches = get_batches(test_sent, pad_idx=vocabs.index("<pad>"))


    data = {
        "vocab": vocabs,
        "tag_dict": tag_dict,
        "embeddings": embeddings,
        "train_batches": train_batches,
        "dev_batches": dev_batches,
        "true_dev_labels": true_dev_labels,
        "p1_train": p1_train,
        "p1_dev": p1_dev,
        "p2_train": p2_train,
        "p2_dev": p2_dev,
        "p1_test": p1_test,
        "p2_test": p2_test,
        "test_batches": test_batches
    }
    pickle.dump(data, open("data.pkl", "wb"))


if __name__ == '__main__':
        main()