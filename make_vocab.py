from preprocess import *
import nltk.tokenize as tokenizer
import json
import pickle


data_folder = 'data/'



def main():
    train_path = "SemEval2010_task8_training/TRAIN_FILE.TXT"
    train_file = open(data_folder + train_path, "r").readlines()
    params = json.load(open("params.json"))

    train_sent = list()
    char_labels = list()
    e1 = list()
    e2 = list()
    p1 = list()
    p2 = list()

    for i in range(8000):
        start = i * 4
        line = train_file[start]
        line = line.split("\t")[1].replace('"',"").replace("\n", "").replace("<", "").replace(">", "").replace("'", "")
        tokens = tokenizer.TreebankWordTokenizer().tokenize(line)
        e1_clause, e2_clause, e1_s, e1_e, e2_s, e2_e = get_e1_e2(tokens)
        e1.append(e1_clause)
        e2.append(e2_clause)
        train_sent.append(tokens)
        char_labels.append(train_file[start + 1].replace("\n", ""))
        pos_1, pos_2 = get_pos(tokens, e1_s, e1_e, e2_s, e2_e)
        p1.append(pos_1)
        p2.append(pos_2)


    vocabs, chars, params["max_char"], params["max_len"] = get_vocabs_chars(train_sent)
    print("There are", len(vocabs), "words in the dataset")
    train_sent = words_to_id(vocabs, train_sent)

    if params["pretrain"]:
        print("Loading GloVe pretrained vectors")
        embeddings = read_embedding(vocabs, 'embeddings/glove.300.txt')
    else:
        embeddings = None

    print("Converting tags to numbers")
    labels, tag_dict = tags_to_id(char_labels, set(char_labels))

    params["n_outputs"] = len(tag_dict.values())
    tag_dict = {idx: tag for tag, idx in tag_dict.items()}


    X = np.array(train_sent)
    y = np.array(labels)

    X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(X, y, range(len(train_sent)),
                                                                                     test_size=0.2, random_state=33)

    json.dump(params, open("params.json", "w"))
    true_dev_labels = [tag_dict[idx] for idx in y_dev]
    train_batches = get_batches(X_train, y_train, params["max_len"], vocabs.index("<pad>"))
    dev_batches = get_batches(X_dev, y_dev, params["max_len"], vocabs.index("<pad>"))
    p1_train = just_batch([p1[idx] for idx in indices_train], max_length=params["max_len"], pad_idx=1000)
    p2_train = just_batch([p2[idx] for idx in indices_train], max_length=params["max_len"], pad_idx=1000)
    p1_dev = just_batch([p1[idx] for idx in indices_dev], max_length=params["max_len"], pad_idx=1000)
    p2_dev = just_batch([p2[idx] for idx in indices_dev], max_length=params["max_len"], pad_idx=1000)


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
        "p2_dev": p2_dev
    }
    pickle.dump(data, open("data.pkl", "wb"))


if __name__ == '__main__':
        main()