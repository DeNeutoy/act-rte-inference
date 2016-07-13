

import tensorflow as tf
from tensorflow.python.ops import  rnn, rnn_cell, seq2seq

class IAAModel(object):

    """ Implements Iterative Alternating Attention for Machine Reading
        http://arxiv.org/pdf/1606.02245v3.pdf """

    def __init__(self, config, pretrained_embeddings=None,update_embeddings=True, is_training=False):

        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        self.vocab_size = config.vocab_size
        self.prem_steps = config.prem_steps
        self.hyp_steps = config.hyp_steps
        # placeholders for inputs
        self.premise = tf.placeholder(tf.int32, [batch_size, self.prem_steps])
        self.hypothesis = tf.placeholder(tf.int32, [batch_size, self.hyp_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, 3])


        if pretrained_embeddings is not None:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32,
                                        initializer=tf.constant_initializer(pretrained_embeddings), trainable=update_embeddings)
        else:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32)


        # create lists of (batch,hidden_size) inputs for models
        premise_inputs = tf.nn.embedding_lookup(embedding, self.premise)
        premise_inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.prem_steps, premise_inputs)]

        hypothesis_inputs = tf.nn.embedding_lookup(embedding, self.hypothesis)
        hypothesis_inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.hyp_steps, hypothesis_inputs)]


        with tf.variable_scope("gru_premise"):
            encoder = rnn_cell.GRUCell(self.config.hidden_size)
            self.premise_cell = rnn_cell.MultiRNNCell([encoder]* self.num_layers)


        # run GRUs over premise + hypothesis
        premise_outputs, premise_state = rnn.rnn(self.premise_cell, premise_inputs,dtype=tf.float32, scope="gru_premise")
        premise_outputs = tf.concat(1, [tf.expand_dims(x,1) for x in premise_outputs])

        with tf.variable_scope("gru_hypothesis"):
            decoder = rnn_cell.GRUCell(self.config.hidden_size)
            self.hyp_cell = rnn_cell.MultiRNNCell([decoder]* self.num_layers)

        hyp_outputs, hyp_state = rnn.rnn(self.hyp_cell,hypothesis_inputs,premise_state,tf.float32, scope= "gru_hypothesis")
        hyp_outputs = tf.concat(1, [tf.expand_dims(x,1) for x in hyp_outputs])


        with tf.variable_scope("gru_inference"):

            inference = rnn_cell.GRUCell(self.config.hidden_size)
            self.inference_cell = rnn_cell.MultiRNNCell([inference]*self.num_layers)
            self.inference_state = self.inference_cell.zero_state(self.batch_size, tf.float32)


        with tf.variable_scope("prediction"):
            prediction = self.do_inference_steps(self.inference_cell,self.inference_state,
                                             premise_outputs, hyp_outputs, 3)

        # softmax over outputs to generate distribution over [neutral, entailment, contradiction]

        softmax_w = tf.get_variable("softmax_w", [hidden_size, 3])
        softmax_b = tf.get_variable("softmax_b", [3])
        self.logits = tf.matmul(prediction, softmax_w) + softmax_b   # dim (batch_size, 3)

        _, targets = tf.nn.top_k(self.targets)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([batch_size])],
                3)
        self.cost = tf.reduce_mean(loss)

        _, logit_max_index = tf.nn.top_k(self.logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(logit_max_index, targets), tf.float32))


        if is_training:

            self.lr = tf.Variable(0.0, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.config.max_grad_norm)

            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def attention(self, query, attendees, scope):
        """Put attention masks on hidden using hidden_features and query."""

        attn_length = attendees.get_shape()[1].value
        attn_size = attendees.get_shape()[2].value

        with tf.variable_scope(scope):

            hidden = tf.reshape(attendees, [-1, attn_length, 1, attn_size])
            k = tf.get_variable("attention_W", [1,1,attn_size,attn_size])

            features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("attention_v", [attn_size])

            with tf.variable_scope("attention"):

                y = tf.nn.rnn_cell._linear(query, attn_size, True)

                y = tf.reshape(y, [-1, 1, 1, attn_size])
                # Attention mask is a softmax of v^T * tanh(...).
                s = tf.reduce_sum(v * tf.tanh(features + y), [2, 3])
                a = tf.nn.softmax(s)
                # Now calculate the attention-weighted vector d.
                d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,[1, 2])
                ds = tf.reshape(d, [-1, attn_size])

        return ds

    def do_inference_steps(self, cell, initial_state, premise, hypothesis, steps):

        state = initial_state

        for i in range(steps):

            if i > 0: tf.get_variable_scope().reuse_variables()

            hyp_attn = self.attention(state, hypothesis, "hyp_attn")

            state_for_premise = tf.concat(1, [state, hyp_attn])

            prem_attn = self.attention(state_for_premise, premise, "prem_attn")

            # feature representation to generate the "gate" to determine
            # how much of the inference we should carry through the cell
            state_for_gates = tf.concat(1, [state, hyp_attn ,prem_attn, prem_attn * hyp_attn])

            hyp_gate = self.gate_mechanism(state_for_gates, "hyp_gate")
            prem_gate = self.gate_mechanism(state_for_gates, "prem_gate")

            input = tf.concat(1, [hyp_gate * hyp_attn, prem_gate * prem_attn])

            output, state = cell(input,state)


        return state


    def gate_mechanism(self, gate_input, scope):

        with tf.variable_scope(scope):
            hidden1_w = tf.get_variable("hidden1_w", [4*self.hidden_size, 4*self.hidden_size])
            hidden1_b = tf.get_variable("hidden1_b", [4*self.hidden_size])

            hidden2_w = tf.get_variable("hidden2_w", [4*self.hidden_size, 2*self.hidden_size])
            hidden2_b = tf.get_variable("hidden2_b", [2*self.hidden_size])

            sigmoid_w = tf.get_variable("sigmoid_w", [2*self.hidden_size, self.hidden_size])
            sigmoid_b = tf.get_variable("sigmoid_b", [self.batch_size, self.hidden_size])

            hidden1 = tf.nn.relu(tf.matmul(gate_input, hidden1_w) + hidden1_b)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)

            gate_output = tf.nn.sigmoid(tf.matmul(hidden2, sigmoid_w) + sigmoid_b)

        return gate_output


