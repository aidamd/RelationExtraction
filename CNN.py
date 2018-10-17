import numpy as np
import tensorflow as tf
from utils.scorer import *
from tensorflow.contrib.layers import fully_connected
from preprocess import *


class CNN():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = np.expand_dims(my_embeddings, 2)

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
        embedding_placeholder_expanded = tf.expand_dims(embedding_placeholder, -1)
        return embedding_placeholder_expanded

    def build(self):
        self.embedding_placeholder = self.build_embedding()

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        pooled_outputs = list()
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.relation = tf.placeholder(tf.int64, [None])
        self.direction = tf.placeholder(tf.int64, [None])
        self.max_len = tf.placeholder(tf.int32)

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(self.embed, W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            #pooled = tf.nn.max_pool(relu, ksize=[1, self.max_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        self.relation_predictions = fully_connected(drop, self.n_outputs, activation_fn=tf.sigmoid)
        self.direction_predictions = fully_connected(drop, 2, activation_fn=tf.sigmoid)

        self.xentropy_rel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relation,
                                                                           logits=self.relation_predictions)
        self.xentropy_dir = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.direction,
                                                                           logits=self.direction_predictions)
        self.loss_rel = tf.reduce_mean(self.xentropy_rel)
        self.loss_dir = tf.reduce_mean(self.xentropy_dir)
        self.loss = (self.loss_dir + 3 * self.loss_rel) / 4

        self.predicted_rel = tf.argmax(self.relation_predictions, 1)
        self.predicted_dir = tf.argmax(self.direction_predictions, 1)

        self.accuracy_rel = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_rel, self.relation), tf.float32))
        self.accuracy_dir = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_dir, self.direction), tf.float32))
        self.accuracy = (self.accuracy_dir + self.accuracy_rel) / 2

        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss)


    def run_model(self, batches, test_batches, labels, dic):
        init = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                ## Train
                if epoch == 20:
                    self.learning_rate = self.learning_rate * 0.1
                if epoch == 50:
                    self.learning_rate = self.learning_rate * 0.5
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for (X_batch, X_len, y_batch) in batches:
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: self.keep_ratio,
                                 self.relation: y_batch[:, 0],
                                 self.direction: y_batch[:, 1],
                                 self.learning_rate_placeholder: self.learning_rate,
                                 self.max_len: X_batch.shape[1]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val= self.sess.run([self.training_op, self.loss], feed_dict=feed_dict)
                    acc_train += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                acc_test = 0
                test_predictions = list()
                for (X_batch, X_len, y_batch) in test_batches:
                    feed_dict = {self.train_inputs: X_batch,
                                self.keep_prob: 1,
                             self.relation: y_batch[:, 0],
                             self.direction: y_batch[:, 1],
                             self.learning_rate_placeholder: self.learning_rate,
                             self.max_len: X_batch.shape[1]}
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    dir = self.predicted_dir.eval(feed_dict=feed_dict)
                    rel = self.predicted_rel.eval(feed_dict=feed_dict)
                    test_predictions.extend([true_label(rel[i], dir[i], dic) for i in range(len(dir))])
                evaluate(labels, test_predictions)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))

                if epoch == self.epochs:
                    test_predictions = np.transpose(test_predictions)
                    break
        #save_path = saver.save(self.sess, "/tmp/model.ckpt")
        return test_predictions, acc_test / float(len(test_batches))