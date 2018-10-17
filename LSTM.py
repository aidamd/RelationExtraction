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

        self.first_pos = tf.placeholder(tf.float32, shape=[None, None, 1], name="first_pos")
        self.second_pos = tf.placeholder(tf.float32, shape=[None, None, 1], name="second_pos")

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.relation = tf.placeholder(tf.int64, [None])
        self.direction = tf.placeholder(tf.int64, [None])
        self.max_len = tf.placeholder(tf.int32)

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)


        rnn_outputs, state = tf.nn.dynamic_rnn(self.network, self.embed,
                                                dtype=tf.float32, sequence_length=self.sequence_length)

        #self.max_len = int(self.train_inputs.get_shape()[1])
        #[batch_size * self.max_len * attention]
        hiddens = tf.tile(tf.reshape(fully_connected(state, self.attention_size), [-1, 1, self.attention_size]), [1, self.max_len, 1])
        summary = fully_connected(rnn_outputs, self.attention_size)
        first = fully_connected(self.first_pos, self.attention_size)
        second = fully_connected(self.second_pos, self.attention_size)

        #attention = fully_connected(hiddens + summary + first + second, self.self.max_len, activation_fn=tf.sigmoid)

        #[batch_size * self.max_len]
        attention = tf.reshape(fully_connected(tf.add(tf.add(hiddens, summary), tf.add(first, second)), 1, activation_fn=tf.sigmoid), [-1, self.max_len, 1])
        mask = tf.reshape(tf.sequence_mask(self.sequence_length, self.max_len), [-1, self.max_len])

        attentioned_states = tf.reduce_sum(attention * rnn_outputs, axis=1)

        self.relation_predictions = fully_connected(attentioned_states, self.n_outputs, activation_fn=tf.sigmoid)
        self.direction_predictions = fully_connected(attentioned_states, 2, activation_fn=tf.sigmoid)

        self.xentropy_rel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relation,
                                                                               logits=self.relation_predictions)
        self.xentropy_dir = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.direction,
                                                                           logits=self.direction_predictions)
        self.loss_rel = tf.reduce_mean(self.xentropy_rel)
        self.loss_dir = tf.reduce_mean(self.xentropy_dir)
        self.loss = (self.loss_dir + 3 * self.loss_rel) / 4

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
        test_batches = data["dev_batches"]
        labels = data["true_dev_labels"]
        dic = data["tag_dict"]
        p1_train = data["p1_train"]
        p1_test = data["p1_dev"]
        p2_train = data["p2_train"]
        p2_test = data["p2_dev"]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as self.sess:
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
                    _, loss_val= self.sess.run([self.training_op, self.loss], feed_dict=feed_dict)
                    acc_train += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                test_predictions = list()
                for idx in range(len(test_batches)):
                    X_batch, X_len, y_batch = test_batches[idx]
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: 1,
                                #self.labels: y_batch,
                                self.relation: y_batch[:, 0],
                                self.direction: y_batch[:, 1],
                                self.sequence_length: X_len,
                                self.learning_rate_placeholder: self.learning_rate,
                                self.first_pos: np.expand_dims(p1_test[idx], 2),
                                self.second_pos: np.expand_dims(p2_test[idx], 2),
                                self.max_len: X_batch.shape[1]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    dir = self.predicted_dir.eval(feed_dict=feed_dict)
                    rel = self.predicted_rel.eval(feed_dict=feed_dict)
                    #print()
                    test_predictions.extend([true_label(rel[i], dir[i], dic) for i in range(len(dir))])
                evaluate(labels, test_predictions)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)))
                if epoch == self.epochs:
                    test_predictions = np.transpose(test_predictions)
                    save_path = saver.save(self.sess, "/tmp/model.ckpt")
                    break
        print("Model is saved in: %s", save_path)
        return test_predictions

    def use_trained(self, data):
        tf.reset_default_graph()

        saver = tf.train.Saver()

        with tf.Session() as self.sess:
            saver.restore(self.sess, "/tmp/model.ckpt")
            print("Model restored")
