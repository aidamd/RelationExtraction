import numpy as np
import tensorflow as tf
from utils.scorer import *
from tensorflow.contrib.layers import fully_connected


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

        self.first_pos = tf.placeholder(tf.float32, shape=[None, self.max_len, 1], name="first_pos")
        self.second_pos = tf.placeholder(tf.float32, shape=[None, self.max_len, 1], name="second_pos")

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, self.max_len], name="inputs")
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int64, [None])

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)


        rnn_outputs, state = tf.nn.dynamic_rnn(self.network, self.embed,
                                                dtype=tf.float32, sequence_length=self.sequence_length)

        #[batch_size * max_len * attention]
        hiddens = tf.reshape(fully_connected(state, self.max_len * self.attention_size), [-1, self.max_len, self.attention_size])
        summary = fully_connected(rnn_outputs, self.attention_size)
        first = fully_connected(self.first_pos, self.attention_size)
        second = fully_connected(self.second_pos, self.attention_size)

        #attention = fully_connected(hiddens + summary + first + second, self.max_len, activation_fn=tf.sigmoid)

        #[batch_size * max_len]
        attention = tf.reshape(fully_connected(tf.add(tf.add(hiddens, summary), tf.add(first, second)), 1, activation_fn=tf.sigmoid), [-1, self.max_len, 1])
        mask = tf.reshape(tf.sequence_mask(self.sequence_length, self.max_len), [-1, self.max_len])

        attentioned_states = tf.reduce_sum(attention * rnn_outputs, axis=1)

        self.predictions = fully_connected(attentioned_states, self.n_outputs, activation_fn=tf.sigmoid)

        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                               logits=self.predictions)
        self.loss = tf.reduce_mean(self.xentropy)

        self.predicted_label= tf.argmax(self.predictions, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_label, self.labels), tf.float32))

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

        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            test_predictions = list()
            while True:
                ## Train
                #if epoch == 50:
                #    self.learning_rate = self.learning_rate * 0.1
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for idx in range(len(batches)):
                    X_batch, X_len, y_batch = batches[idx]
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: self.keep_ratio,
                                self.labels: y_batch,
                                self.sequence_length: X_len,
                                self.learning_rate_placeholder: self.learning_rate,
                                self.first_pos: np.expand_dims(p1_train[idx], 2),
                                self.second_pos: np.expand_dims(p2_train[idx], 2)
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val, predictions_= self.sess.run([self.training_op, self.loss, self.xentropy], feed_dict=feed_dict)
                    acc_train += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                acc_test = 0
                for idx in range(len(test_batches)):
                    X_batch, X_len, y_batch = test_batches[idx]
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: 1,
                                self.labels: y_batch,
                                self.sequence_length: X_len,
                                self.learning_rate_placeholder: self.learning_rate,
                                self.first_pos: np.expand_dims(p1_test[idx], 2),
                                self.second_pos: np.expand_dims(p2_test[idx], 2)
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    x= self.predicted_label.eval(feed_dict=feed_dict)
                    print(x)
                    test_predictions.extend([dic[i] for i in x])
                evaluate(labels, test_predictions)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))

                if epoch == self.epochs:
                    test_predictions = np.transpose(test_predictions)
                    break
        #save_path = saver.save(self.sess, "/tmp/model.ckpt")
        return test_predictions, acc_test / float(len(test_batches))