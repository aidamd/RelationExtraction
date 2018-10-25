This code uses Python 3.6
Libraries:

- nltk 3.2.5
- numpy 1.15.2
- tensorflow 1.8
- sklearn 0.19.2

The params.json file contains the parameters of the program:
    attention_size: the size of hidden vectors that re generated to produce the attention values,
    num_layers : number of cell layers in the RNN
    model : can be LSTM, BiLSTM or CNN
    hidden_size : the hidden size of the neural cells
    n_outputs : number of labels, it is added to the params file by make_vocab.py
    filter_sizes : size of different filters used by CNN
    num_filters : number of filters used in CNN
    pretrain : use pretrained word embeddings or not
    embedding_size : size of embeddings
    learning_rate : learning rate of the model
    keep_ratio : keep ratio of cells
    epochs : max number of epochs
    max_char : added by make_vocab.py
    max_len : added by make_vocab.py


Pleas add 300d Glove word embeddings (Download from http://nlp.stanford.edu/data/glove.840B.300d.zip)
into embeddings folder and name it "glove.300.txt".

You can first run "python3 make_vocab.py" to read the train and test file, partition the train set to train and dev,
get position vectors for all sentences, convert sentences to their their Glove representation and also get the labels.

make_vocab.py generates the data.pkl that contains all the data. You can also skip this part and use the data.pkl file that is already in the repository



