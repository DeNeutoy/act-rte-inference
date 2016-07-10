

import tensorflow as tf
from AttentiveACTCell import AttentiveACTCell
from tensorflow.python.ops import array_ops, rnn, rnn_cell, seq2seq

class RTEModel(object):

    def __init__(self, config, is_training=False):

        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        self.vocab_size = config.vocab_size


        # placeholders for inputs
        self.premise = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.hypothesis = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, 3])

        self.initial_state = array_ops.zeros(
                array_ops.pack([self.batch_size, self.num_steps]),
                 dtype=tf.float32).set_shape([None, self.num_steps])

        embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32)


        # create lists of (batch,hidden_size) inputs for models
        premise_inputs = tf.nn.embedding_lookup(embedding, self.premise)
        premise_inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.config.num_steps, premise_inputs)]

        hypothesis_inputs = tf.nn.embedding_lookup(embedding, self.hypothesis)
        hypothesis_inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.config.num_steps, hypothesis_inputs)]


        with tf.variable_scope("gru_encoder"):
            encoder = rnn_cell.GRUCell(self.config.hidden_size)
            self.premise_cell = rnn_cell.MultiRNNCell([encoder]* self.num_layers)


        # run premise GRU over the sentence
        premise_outputs, premise_state = rnn.rnn(self.premise_cell, premise_inputs,dtype=tf.float32)

        with tf.variable_scope("gru_decoder"):
            decoder = rnn_cell.GRUCell(self.config.hidden_size)
            self.hyp_cell = rnn_cell.MultiRNNCell([decoder]* self.num_layers)


        with tf.variable_scope("ACT"):

            act = AttentiveACTCell(self.config.hidden_size, self.hyp_cell, config.epsilon,
                            max_computation = config.max_computation,
                            batch_size = self.batch_size,
                            encoder_outputs=premise_outputs)



        hyp_outputs, hyp_state = rnn.rnn(act,hypothesis_inputs,premise_state,tf.float32)

        premise_weight = tf.get_variable("premise_weight", [hidden_size, hidden_size])
        hyp_weight = tf.get_variable("hypothesis_weight", [hidden_size, hidden_size])

        output = tf.tanh(tf.matmul(premise_weight, premise_outputs[-1]) + tf.matmul(hyp_weight, hyp_outputs[-1]))

        # softmax over outputs to generate distribution over [neutral, entailment, contradiction]

        softmax_w = tf.get_variable("softmax_w", [hidden_size, 3])
        softmax_b = tf.get_variable("softmax_b", [3])
        self.logits = tf.matmul(output, softmax_w) + softmax_b   # dim (batch_size, 3)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])],
                3)

        self.cost = tf.reduce_mean(loss)


        if is_training:

            self.lr = tf.Variable(0.0, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.config.max_grad_norm)

            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))