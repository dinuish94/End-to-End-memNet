import tensorflow as tf
import numpy as np
import random
import argparse
import sys
import os
import math

from utils import ProgressBar


class LSTM_rnn():

    def __init__(self, config, sess):

        self.state_size = 15
        self.num_classes = 20
        # self.num_classes = num_classes
        # self.ckpt_path = ckpt_path
        # self.model_name = model_name
        self.nwords = config.nwords
        self.max_words = config.max_words
        self.max_sentences = config.max_sentences
        self.init_mean = config.init_mean
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.anneal_epoch = config.anneal_epoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.max_grad_norm = config.max_grad_norm

        self.lin_start = config.lin_start
        self.show_progress = config.show_progress
        self.is_test = config.is_test

        self.global_step = tf.Variable(0, name='global_step')

        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.query = tf.placeholder(
            tf.int32, [None, self.max_words], name='input')
        self.time = tf.placeholder(
            tf.int32, [None, self.mem_size], name='time')
        self.target = tf.placeholder(
            tf.float32, [None, self.nwords], name='target')
        self.context = tf.placeholder(
            tf.int32, [None, self.mem_size, self.max_words], name='context')

        self.hid = []

        self.lr = None

        if self.lin_start:
            self.current_lr = 0.005
        else:
            self.current_lr = config.init_lr

        self.anneal_rate = config.anneal_rate
        self.loss = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

        # build graph ops
        def __graph__():
            # tf.reset_default_graph()
            # inputs
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
            #
            # embeddings
            embs = tf.get_variable('emb', [self.num_classes, self.state_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            #
            # initial hidden state
            init_state = tf.placeholder(
                shape=[2, None, self.state_size], dtype=tf.float32, name='initial_state')
            # initializer
            xav_init = tf.contrib.layers.xavier_initializer
            # params
            W = tf.get_variable(
                'W', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            U = tf.get_variable(
                'U', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.))
            ####
            # step - LSTM

            def step(prev, x):
                # gather previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)
                ####
                # GATES
                #
                #  input gate
                i = tf.sigmoid(tf.matmul(x, U[0]) + tf.matmul(st_1, W[0]))
                #  forget gate
                f = tf.sigmoid(tf.matmul(x, U[1]) + tf.matmul(st_1, W[1]))
                #  output gate
                o = tf.sigmoid(tf.matmul(x, U[2]) + tf.matmul(st_1, W[2]))
                #  gate weights
                g = tf.tanh(tf.matmul(x, U[3]) + tf.matmul(st_1, W[3]))
                ###
                # new internal cell state
                ct = ct_1*f + g*i
                # output state
                st = tf.tanh(ct)*o
                return tf.stack([st, ct])
            ###
            # here comes the scan operation; wake up!
            #   tf.scan(fn, elems, initializer)
            states = tf.scan(step,
                             tf.transpose(rnn_inputs, [1, 0, 2]),
                             initializer=init_state)
            #
            # predictions
            V = tf.get_variable('V', shape=[self.state_size, self.num_classes],
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[self.num_classes],
                                 initializer=tf.constant_initializer(0.))

            ####
            # get last state before reshape/transpose
            last_state = states[-1]

            ####
            # transpose
            states = tf.transpose(states, [1, 2, 0, 3])[0]
            #st_shp = tf.shape(states)
            # flatten states to 2d matrix for matmult with V
            #states_reshaped = tf.reshape(states, [st_shp[0] * st_shp[1], st_shp[2]])
            states_reshaped = tf.reshape(states, [-1, self.state_size])
            logits = tf.matmul(states_reshaped, V) + bo
            # predictions
            predictions = tf.nn.softmax(logits)
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=ys_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(
                learning_rate=0.1).minimize(loss)

            self.optim = tf.train.AdagradOptimizer(
                learning_rate=0.1).minimize(loss)
            #
            # expose symbols
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state
        #####
        # build graph
        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    ####
    # training
    def train(self, train_stories, train_questions):
        # training session
        # with tf.Session() as sess:

        N = int(math.ceil(len(train_questions) / self.batch_size))
        cost = 0

        if self.show_progress:
            bar = ProgressBar('Train', max=N)

        for idx in range(N):

            if self.show_progress:
                bar.next()

            if idx == N - 1:
                iterations = len(train_questions) - \
                    (N - 1) * self.batch_size
            else:
                iterations = self.batch_size

            query = np.ndarray(
                [iterations, self.max_words], dtype=np.int32)
            time = np.zeros([iterations, self.mem_size], dtype=np.int32)
            target = np.zeros([iterations, self.nwords], dtype=np.float32)
            context = np.ndarray(
                [iterations, self.mem_size, self.max_words], dtype=np.int32)

            for b in range(iterations):
                m = idx * self.batch_size + b

                curr_q = train_questions[m]
                q_text = curr_q['question']
                story_ind = curr_q['story_index']
                sent_ind = curr_q['sentence_index']
                answer = curr_q['answer'][0]

                curr_s = train_stories[story_ind]
                curr_c = curr_s[:sent_ind + 1]

                if len(curr_c) >= self.mem_size:
                    curr_c = curr_c[-self.mem_size:]

                    for t in range(self.mem_size):
                        time[b, t].fill(t)
                else:

                    for t in range(len(curr_c)):
                        time[b, t].fill(t)

                    while len(curr_c) < self.mem_size:
                        curr_c.append([0.] * self.max_words)

                query[b, :] = q_text
                target[b, answer] = 1
                context[b, :, :] = curr_c

            _, loss, self.step = self.sess.run([self.optim, self.loss, self.global_step],
                                            feed_dict={self.query: query, self.time: time,
                                                        self.target: target, self.context: context})
            cost += np.sum(loss)

            # return cost / len(train_questions)

            # training ends here;
            #  save checkpoint
            # saver = tf.train.Saver()
            # saver.save(sess, self.checkpoint_dir, global_step=idx)
        if self.show_progress:
                bar.finish()
