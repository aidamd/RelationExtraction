
from collections import Counter
import operator
import numpy as np
import re
from sklearn.model_selection import train_test_split


def stopwords(corpus):
    stop_words = set(stopwords.words('english'))
    stop_words_exp = re.compile(r"({})\s+".format('|'.join(stop_words)))
    for idx in range(len(corpus)):
        corpus[idx] = [stop_words_exp.sub(' ', text.lower()) for text in corpus[idx]]
    return corpus

def get_vocabs_chars(data):
    # Gets the list of words and characters in the dataset
    all = [word for sent in data for word in sent]
    chars = list(set(char for word in all for char in word))
    dictionary = Counter([word for word in all])
    words, counts = zip(*sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
    vocab = list(words) + ["<unk>", "<pad>"]
    return vocab, chars, max(len(word) for word in dictionary.keys()), max(len(sent) for sent in data)

def tags_to_id(tags, unique_tags):
    # converts the NER tags to numbers
    dictionary = {tag: idx for idx, tag in enumerate(unique_tags)}
    for item in range(len(tags)):
        tags[item] = dictionary[tags[item]]
    return tags, dictionary

def words_to_id(vocab, data):
    # converts the words to their index according to generated vocab list
    dictionary = {word: idx for idx, word in enumerate(vocab)}
    for sent in range(len(data)):
        for word in range(len(data[sent])):
            try:
                data[sent][word] = dictionary[data[sent][word]]
            except Exception:
                data[sent][word] = dictionary["<unk>"]
                print(data[sent][word])
    return data

def read_embedding(vocab, file):
    # reads an embedding file and return a dictionary of word: vector
    with open(file, 'r') as file:
        vectors = dict()
        for line in file:
            tokens = line.split()
            vec = np.array(tokens[len(tokens) - 300:], dtype=np.float32)
            token = "".join(tokens[:len(tokens) - 300])
            vectors[token] = vec
    unk_embedding = np.random.rand(300) * 2. - 1.
    embedding = dict()
    for v in vocab:
        try:
            embedding[v] = vectors[v]
        except Exception:
            # if the word is not in the embeddings, use the random vector
            embedding[v] = unk_embedding
    return np.array(list(embedding.values()))

def just_batch(corpus, max_length=None, pad_idx=None):
    # changes the input list to a list of batches
    batches = []
    for idx in range(len(corpus) // 100):
        text_batch = corpus[idx * 100: min((idx + 1) * 100, len(corpus))]
        if pad_idx:
            text_batch = add_padding(text_batch, pad_idx, max_length)
        batches.append(np.array([np.array(line) for line in text_batch]))
    return batches

def get_batches(corpus, labels=None, max_length=None, pad_idx=None):
    # changes two lists of inputs to batches with paddings
    batches = []
    for idx in range(len(corpus) // 100):
        text_batch = corpus[idx * 100: min((idx + 1) * 100, len(corpus))]
        labels_batch = np.array(labels[idx * 100: min((idx + 1) * 100,
                                                         len(labels))] if labels is not None else [])

        if pad_idx:
            lengths = np.array([len(line) for line in text_batch])
            text_batch = add_padding(text_batch, pad_idx, max_length)
            #labels_batch = add_padding(labels_batch, -1, max_length)
        batches.append((np.array([np.array(line) for line in text_batch]), lengths, np.array([np.array(label) for label in labels_batch])))
    return batches


def add_padding(corpus, pad_idx, max_length):
    # adds padding to a dataset according to the padding token and max length
    for i in range(len(corpus)):
        corpus[i] = corpus[i][:min(len(corpus[i]), max_length) - 1]
        while len(corpus[i]) < max_length:
            corpus[i].append(pad_idx)
    return corpus

def clean(sentences):
    for sent in range(len(sentences)):
        for word in range(len(sentences[sent])):
            sentences[sent][word] = sentences[sent][word].lower()
    return sentences

def get_features(sent, idx, POS, POS2):
    # extracts features of a word in the sentence
    word = sent[idx][0]
    try:
        features = {
            "first": int(idx == 0),
            "last": int(idx == len(sent)),
            "upper": int(word.isupper()),
            "title": int(word.istitle()),
            "POS": POS[sent[idx][1]],
            "POS2": POS2[sent[idx][2]],
            "digit": int(word.isdigit()),
            "hasdigit": int(re.match(".*\\d+.*", word) is not None),
            "hasnonalpha": int(re.match(".*[^a-zA-Z].*", word) is not None),
            "padding": 0
        }
        vec = np.array(list(features.values()))
    except Exception:
        pass
        vec = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 0])
    return vec / sum(vec)

def ids_to_char(chars, vocab, data, max, sent_max):
    # generated the list of character ids
    new_data = list()
    lengths = list()
    padding = [len(chars)+1 for i in range(max)]
    for sent in range(len(data)):
        new_data.append(list())
        lengths.append(list())
        for word in range(len(data[sent])):
            my_word = data[sent][word][0]
            temp = []
            lengths[sent].append(min(len(my_word), max))
            for l in list(my_word):
                try:
                    temp.append(chars.index(l))
                except Exception:
                    temp.append(len(chars)+1)
            while len(temp) < max:
                temp.append(len(chars)+1)
            temp = temp[:max]
            new_data[sent].append(temp)
        while(len(new_data[sent]) < sent_max):
            new_data[sent].append(padding)
            lengths[sent].append(0)
        new_data[sent] = np.array([np.array(x) for x in new_data[sent]])
    return np.array([np.array(d) for d in new_data]), np.array([np.array(l) for l in lengths])


def get_e1_e2(tokens):
    e2_s = 0
    e2_e = 0
    e1_s = 0
    e1_e = 0
    e1_clause = ""
    e2_clause = ""
    for idx, tok in enumerate(tokens):
        if tok[:2] == "e1":
            e1_s = idx
            tokens[idx] = tokens[idx][2:]
        if tok[-3:] == "/e1":
            e1_e = idx
            tokens[idx] = tokens[idx][:-3]
            if e1_s != e1_e:
                e1_clause = " ".join(tokens[e1_s:e1_e + 1])
            else:
                e1_clause = tokens[idx]
        if tok[:2] == "e2":
            e2_s = idx
            tokens[idx] = tokens[idx][2:]
        if tok[-3:] == "/e2":
            e2_e = idx
            tokens[idx] = tokens[idx][:-3]
            if e2_s != e2_e:
                e2_clause = " ".join(tokens[e2_s:e2_e + 1])
            else:
                e2_clause = tokens[idx]
    return e1_clause, e2_clause, e1_s, e1_e, e2_s, e2_e


def get_pos(tokens, e1_s, e1_e, e2_s, e2_e):
    pos_1 = list()
    pos_2 = list()
    for idx in range(len(tokens)):
        if idx < e1_s:
            pos_1.append(idx - e1_s)
        if idx < e2_s:
            pos_2.append(idx - e2_s)
        if idx >= e1_s and idx <= e1_e:
            pos_1.append(0)
        if idx >= e2_s and idx <= e2_e:
            pos_2.append(0)
        if idx > e1_e:
            pos_1.append(idx - e1_e)
        if idx > e2_e:
            pos_2.append(idx - e2_e)
    return pos_1, pos_2
