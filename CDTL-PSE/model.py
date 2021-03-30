# encoding = utf8
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import layers
from tensorflow.contrib.seq2seq.python.ops import *
import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob

class Model(object):
    def __init__(self, config):
        # add placeholders for the model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.config = config
            self.lr = config["lr"]
            self.char_dim = config["char_dim"]
            self.lstm_dim = config["lstm_dim"]
            self.seg_dim = config["seg_dim"]
            self.num_tags = config["num_tags"]
            self.num_tags_ner = config["num_tags_ner"]
            self.num_chars = config["num_chars"]
            self.num_chars_ner = config["num_chars_ner"]
            self.num_segs = 4
            self.num_layers = 2
            self.global_step = tf.Variable(0, trainable=False)
            self.global_step_ner = tf.Variable(0, trainable=False)
            self.best_dev_f1 = tf.Variable(0.0, trainable=False)
            self.best_test_f1 = tf.Variable(0.0, trainable=False)
            self.initializer = initializers.xavier_initializer()
            self.char_inputs = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name="ChatInputs")

            self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name="SegInputs")
            self.targets = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="Targets")
            # dropout keep prob
            self.dropout = tf.placeholder(dtype=tf.float32,
                                          name="Dropout")
            Add_Bilstm = True
            used = tf.sign(tf.abs(self.char_inputs))
            length = tf.reduce_sum(used, reduction_indices=1)
            self.lengths = tf.cast(length, tf.int32)
            self.batch_size = tf.shape(self.char_inputs)[0]
            self.num_steps = tf.shape(self.char_inputs)[-1]
            used_ner = tf.sign(tf.abs(self.char_inputs))
            length_ner = tf.reduce_sum(used_ner, reduction_indices=1)
            self.lengths_ner = tf.cast(length_ner, tf.int32)
            self.batch_size_ner = tf.shape(self.char_inputs)[0]
            self.num_steps_ner = tf.shape(self.char_inputs)[-1]

            # embeddings for chinese character and segmentation representation
            embedding_ner = self.embedding_layer_ner(self.char_inputs, self.seg_inputs, config)
            embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
            # apply dropout before feed to lstm layer
            lstm_inputs_ner = tf.nn.dropout(embedding_ner, self.dropout)
            lstm_inputs = tf.nn.dropout(embedding, self.dropout)
            # bi-directional lstm layer
            lstm_outputs_ner, lstm_cell_ner = self.biLSTM_layer_ner(lstm_inputs_ner, self.lstm_dim, self.lengths_ner)
            lstm_outputs_input ,lstm_cell= self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
            # adaptation layer
            if Add_Bilstm ==True:
                lstm_outputs = self.outadaptation_layer(lstm_outputs_input, self.lstm_dim, self.lengths)
                # logits for tags
                self.logits_ner = self.project_layer_ner(lstm_outputs_ner,False)
                self.logits = self.project_layer(lstm_outputs,False )
                # loss of the model
                self.loss_ner = self.loss_layer_ner(self.logits_ner, self.lengths_ner)
                self.loss = self.loss_layer(self.logits, self.lengths)
            else:
                # logits for tags
                self.logits_ner = self.project_layer_ner(lstm_outputs_ner,True)
                self.logits = self.project_layer(lstm_outputs_input,True)
                # loss of the model
                self.loss_ner = self.loss_layer_ner(self.logits_ner, self.lengths_ner)
                self.loss = self.loss_layer(self.logits, self.lengths)
            with tf.variable_scope("optimizer"):
                optimizer = self.config["optimizer"]
                if optimizer == "sgd":
                    self.opt = tf.train.GradientDescentOptimizer(self.lr)
                elif optimizer == "adam":
                    self.opt = tf.train.AdamOptimizer(self.lr)
                elif optimizer == "adgrad":
                    self.opt = tf.train.AdagradOptimizer(self.lr)
                else:
                    raise KeyError
                # apply grad clip to avoid gradient explosion
                grads_vars = self.opt.compute_gradients(self.loss)
                grads_vars_ner = self.opt.compute_gradients(self.loss_ner)
                self.train_op = self.opt.apply_gradients(grads_vars, self.global_step)
                self.train_op_ner = self.opt.apply_gradients(grads_vars_ner, self.global_step_ner)
            # saver of the model
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        embedding = []
        with tf.variable_scope("char_embedding",reuse=True if not name else name):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars_ner, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding",reuse=True):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed
    def embedding_layer_ner(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
        with tf.variable_scope("char_embedding"  if not name else name):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars_ner, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM",reuse=True if not name else name) as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction,reuse=True):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
            hidden_lstm= tf.concat(outputs, axis=2)


        return hidden_lstm,lstm_cell

    def outadaptation_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("outadaptation" if not name else name) as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction,reuse=True):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
            hidden_lstm= tf.concat(outputs, axis=2)
        return hidden_lstm

    def biLSTM_layer_ner(self, lstm_inputs_ner, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name) as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs_ner,
                dtype=tf.float32,
                sequence_length=lengths)
            hidden_lstm = tf.concat(outputs, axis=2)

        return hidden_lstm,lstm_cell

    def project_layer(self, lstm_outputs,argument, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden",reuse=True):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])

                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                if argument == True:
                    mask_0 = tf.zeros_like(hidden)
                    mask = tf.concat([hidden, mask_0, hidden], axis=1)
                    W = tf.get_variable("W", shape=[self.lstm_dim*3, self.num_tags],
                                        dtype=tf.float32, initializer=self.initializer)

                    b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
                    hidden = tf.nn.xw_plus_b(mask, W, b)
                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def project_layer_ner(self, lstm_outputs_ner,argument, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                output = tf.reshape(lstm_outputs_ner, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits_ner"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags_ner],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags_ner], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                if argument==True:
                    mask_0 = tf.zeros_like(hidden)
                    mask=tf.concat([hidden,hidden,mask_0],axis=1)
                    W = tf.get_variable("W", shape=[self.lstm_dim*3, self.num_tags],
                                        dtype=tf.float32, initializer=self.initializer)

                    b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
                    hidden = tf.nn.xw_plus_b(mask, W, b)
                pred_ner = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred_ner, [-1, self.num_steps_ner, self.num_tags_ner])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def loss_layer_ner(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss_ner"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size_ner, 1, self.num_tags_ner]), tf.zeros(shape=[self.batch_size_ner, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size_ner, self.num_steps_ner, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags_ner*tf.ones([self.batch_size_ner, 1]), tf.int32), self.targets], axis=-1)

            self.trans_ner = tf.get_variable(
                "transitions_ner",
                shape=[self.num_tags_ner + 1, self.num_tags_ner + 1],
                initializer=self.initializer)
            log_likelihood, self.trans_ner = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans_ner,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        #print('batch',batch)

        feed_dict = self.create_feed_dict(is_train, batch)
        #print('feed_dict',feed_dict)
        if is_train:
            global_step,loss,_ = sess.run(
                    [self.global_step, self.loss, self.train_op],
                    feed_dict)
            print('loss',loss)
            print('global_step',global_step)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits
    def create_feed_dict_ner(self, is_train, batch_ner):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch_ner

        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)

            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step_ner(self, sess, is_train, batch_ner):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict_ner(is_train, batch_ner)
        if is_train:
            global_step_ner, loss_ner,_ = sess.run(
                [self.global_step_ner, self.loss_ner, self.train_op_ner],
                feed_dict)
            print('loss',loss_ner)
            print('global_step',global_step_ner)
            return global_step_ner, loss_ner
        else:
            lengths, logits = sess.run([self.lengths_ner, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths
    def decode_ner(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags_ner +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths
    def precision(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        preds = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                pred=[]
                string = strings[i][:lengths[i]]
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                preds.append(pred)
        return preds
    def evaluate(self, sess, data_manager, id_to_tag,precision_loc,precision_per,precision_org):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        predds = []
        for i in range(len(precision_loc)):
            pred1 = precision_loc[i]
            pred2 = precision_per[i]
            pred3 = precision_org[i]
            preds = pred1
            for j in range(len(pred1)):
                value1 = pred1[j]
                value2 = pred2[j]
                value3 = pred3[j]
                if value1 == value2 == value3:
                    value = value1
                elif value1 == value2 and value2!= value3:
                    value = value1
                elif value1 == value3 and value2!= value3:
                    value = value1
                elif value2 == value3 and value1!= value3:
                    value = value2
                else:
                    value=value1
                preds[j] = value

            predds.append(preds)
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)

            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = predds[i]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results
    def evaluate_ner(self, sess, data_manager_ner, id_to_tag_ner):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans_ner.eval()
        for batch in data_manager_ner.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step_ner(sess, False, batch)
            batch_paths = self.decode_ner(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag_ner[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag_ner[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results
    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)


