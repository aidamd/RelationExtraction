import numpy as np
import tensorflow as tf
from utils.scorer import *
from tensorflow.contrib.layers import fully_connected
from preprocess import *


class LSTM():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=False, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocab), self.embedding_size])
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        self.embedding_placeholder = self.build_embedding()
        self.sequence_length = tf.placeholder(tf.int32, [None])

        # First_Pos shows the relative position of each word in the sentence regarding e1
        self.first_pos = tf.placeholder(tf.float32, shape=[None, None, 1], name="first_pos")

        # Second_Pos shows the relative position of each word in the sentence regarding e2
        self.second_pos = tf.placeholder(tf.float32, shape=[None, None, 1], name="second_pos")

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        #Assigning a place holder to learning rate, to reduce it later in the training process
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32)

        #Relation is th 9 possible types of relations
        self.relation = tf.placeholder(tf.int64, [None])

        #Direction is either 0 -> (e1, e2) oe 1 -> (e2, e1)
        self.direction = tf.placeholder(tf.int64, [None])
        self.max_len = tf.placeholder(tf.int32)

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        if self.model == "LSTM":
            rnn_outputs, state = tf.nn.dynamic_rnn(self.network, self.embed,
                                                dtype=tf.float32, sequence_length=self.sequence_length)
        elif self.model == "BiLSTM":
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.network, self.network, self.embed,
                                                                  dtype=tf.float32, sequence_length=self.sequence_length)
            fw_outputs, bw_outputs = bi_outputs
            fw_states, bw_states = bi_states
            rnn_outputs = tf.concat([fw_outputs, bw_outputs], 2)
            state = tf.concat([fw_states, bw_states], 2)

        # linear tranfer of the final state, tiled to be added to each hidden vector
        # shape: [batch_size, max_len, attention_size]
        hiddens = tf.tile(tf.reshape(fully_connected(state, self.attention_size), [-1, 1, self.attention_size]), [1, self.max_len, 1])

        # linear transfer of the hidden vectors
        # shape: [batch_size, max_len, attention_size]
        summary = fully_connected(rnn_outputs, self.attention_size)

        # linear transfer of the first_pos and second_pos
        # each of shape: [batch_size, max_len, attention_size]
        first = fully_connected(self.first_pos, self.attention_size)
        second = fully_connected(self.second_pos, self.attention_size)

        # sigmoid function on the linear transfer of the sum of hiddens, summary, first and second
        # the production is the vector of attentions, a value between 0 and 1 is assigned to each word
        # shape: [batch_size, max_len, 1]
        attention = tf.reshape(fully_connected(tf.add(tf.add(hiddens, summary), tf.add(first, second)), 1, activation_fn=tf.sigmoid), [-1, self.max_len, 1])

        # weighted sum of the hidden states, considering the attention values
        attentioned_states = tf.reduce_sum(attention * rnn_outputs, axis=1)

        # Multi-task section applies 2 different linear functions to the weighted sum to get relation and direction
        self.relation_predictions = fully_connected(attentioned_states, self.n_outputs, activation_fn=tf.sigmoid)
        self.direction_predictions = fully_connected(attentioned_states, 2, activation_fn=tf.sigmoid)

        self.xentropy_rel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relation,
                                                                               logits=self.relation_predictions)
        self.xentropy_dir = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.direction,
                                                                           logits=self.direction_predictions)
        self.loss_rel = tf.reduce_mean(self.xentropy_rel)
        self.loss_dir = tf.reduce_mean(self.xentropy_dir)

        # weighted sum of 2 loss functions, to give more priority to relation
        self.loss = (self.loss_dir + 4 * self.loss_rel) / 5

        self.predicted_rel= tf.argmax(self.relation_predictions, 1)
        self.predicted_dir = tf.argmax(self.direction_predictions, 1)

        self.accuracy_rel = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_rel, self.relation), tf.float32))
        self.accuracy_dir = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_dir, self.direction), tf.float32))
        self.accuracy = (self.accuracy_dir + self.accuracy_rel) / 2

        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss)


    def run_model(self, data):
        batches = data["train_batches"]
        dev_batches = data["dev_batches"]
        labels = data["true_dev_labels"]
        dic = data["tag_dict"]
        p1_train = data["p1_train"]
        p1_dev = data["p1_dev"]
        p2_train = data["p2_train"]
        p2_dev = data["p2_dev"]
        test_batches = data["test_batches"]
        test_p1 = data["p1_test"]
        test_p2 = data["p2_test"]
        dic = data["tag_dict"]

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()
            epoch = 1

            while True:
                ## Train
                if epoch == 100:
                    self.learning_rate = self.learning_rate * 0.1
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for idx in range(len(batches)):
                    X_batch, X_len, y_batch = batches[idx]
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: self.keep_ratio,
                                #self.labels: y_batch,
                                self.relation: y_batch[:, 0],
                                self.direction: y_batch[:, 1],
                                self.sequence_length: X_len,
                                self.learning_rate_placeholder: self.learning_rate,
                                self.first_pos: np.expand_dims(p1_train[idx], 2),
                                self.second_pos: np.expand_dims(p2_train[idx], 2),
                                self.max_len: X_batch.shape[1]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val= sess.run([self.training_op, self.loss], feed_dict=feed_dict)
                    acc_train += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## dev
                dev_predictions = list()
                for idx in range(len(dev_batches)):
                    X_batch, X_len, y_batch = dev_batches[idx]
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: 1,
                                #self.labels: y_batch,
                                self.relation: y_batch[:, 0],
                                self.direction: y_batch[:, 1],
                                self.sequence_length: X_len,
                                self.learning_rate_placeholder: self.learning_rate,
                                self.first_pos: np.expand_dims(p1_dev[idx], 2),
                                self.second_pos: np.expand_dims(p2_dev[idx], 2),
                                self.max_len: X_batch.shape[1]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    if not self.train:
                        _, loss_val = sess.run([self.training_op, self.loss], feed_dict=feed_dict)
                    dir = self.predicted_dir.eval(feed_dict=feed_dict)
                    rel = self.predicted_rel.eval(feed_dict=feed_dict)
                    dev_predictions.extend([true_label(rel[i], dir[i], dic) for i in range(len(dir))])
                evaluate(labels, dev_predictions)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)))
                if epoch == self.epochs:
                    break

            test_predictions = list()
            for idx in range(len(test_batches)):
                X_batch, X_len, _ = test_batches[idx]
                feed_dict = {self.train_inputs: X_batch,
                             self.keep_prob: 1,
                             self.sequence_length: X_len,
                             self.learning_rate_placeholder: self.learning_rate,
                             self.first_pos: np.expand_dims(test_p1[idx], 2),
                             self.second_pos: np.expand_dims(test_p2[idx], 2),
                             self.max_len: X_batch.shape[1]
                             }
                if self.pretrain:
                    feed_dict[self.embedding_placeholder] = self.my_embeddings
                dir = self.predicted_dir.eval(feed_dict=feed_dict)
                rel = self.predicted_rel.eval(feed_dict=feed_dict)
                test_predictions.extend([true_label(rel[i], dir[i], dic) for i in range(len(dir))])
            return test_predictions