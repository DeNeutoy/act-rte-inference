

import tensorflow as tf
from tensorflow.python.ops import  rnn, rnn_cell, seq2seq
from embedding_utils import input_projection3D

class ACTDAAnalysisModel(object):


    """ Implements a Decomposable Attention model for NLI.
        http://arxiv.org/pdf/1606.01933v1.pdf """

    def __init__(self, config, pretrained_embeddings=None,update_embeddings=True, is_training=False):

        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        self.vocab_size = config.vocab_size
        self.prem_steps = config.prem_steps
        self.hyp_steps = config.hyp_steps
        self.is_training = is_training
        # placeholders for inputs
        self.premise = tf.placeholder(tf.int32, [batch_size, self.prem_steps])
        self.hypothesis = tf.placeholder(tf.int32, [batch_size, self.hyp_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, 3])

        if pretrained_embeddings is not None:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_size], dtype=tf.float32,
                                        trainable=update_embeddings)
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.config.embedding_size])
            self.embedding_init = embedding.assign(self.embedding_placeholder)
        else:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32)


        # create lists of (batch,step,hidden_size) inputs for models
        premise_inputs = tf.nn.embedding_lookup(embedding, self.premise)
        hypothesis_inputs = tf.nn.embedding_lookup(embedding, self.hypothesis)


        if pretrained_embeddings is not None:
            with tf.variable_scope("input_projection"):
                premise_inputs = input_projection3D(premise_inputs, self.hidden_size)
            with tf.variable_scope("input_projection", reuse=True):
                hypothesis_inputs = input_projection3D(hypothesis_inputs, self.hidden_size)

        # run FF networks over inputs
        with tf.variable_scope("FF"):
            prem_attn = self.feed_forward_attention(premise_inputs)
        with tf.variable_scope("FF", reuse=True):
            hyp_attn = self.feed_forward_attention(hypothesis_inputs)

        # get activations, shape: (batch, prem_steps, hyp_steps )
        dot = tf.batch_matmul(prem_attn, hyp_attn, adj_y=True)

        hypothesis_softmax = tf.reshape(dot, [batch_size*self.prem_steps, -1,]) #(300,10)
        hypothesis_softmax = tf.expand_dims(tf.nn.softmax(hypothesis_softmax),2)

        dot = tf.transpose(dot, [0,2,1])

        premise_softmax = tf.reshape(dot, [batch_size*self.hyp_steps, -1]) #(200,15)
        premise_softmax = tf.expand_dims(tf.nn.softmax(premise_softmax),2)

        # this is very ugly: we make a copy of the original input for each of the steps
        # in the opposite sentence, multiply with softmax weights, sum and reshape.
        alphas = tf.reduce_sum(premise_softmax *
                               tf.tile(premise_inputs, [self.hyp_steps, 1, 1]), [1])
        betas = tf.reduce_sum(hypothesis_softmax *
                              tf.tile(hypothesis_inputs, [self.prem_steps, 1 ,1 ]), [1])

        # this is (batch, hyp_steps, hidden dim )
        alphas = [tf.squeeze(x,[1]) for x in
                  tf.split(1,self.hyp_steps,tf.reshape(alphas, [batch_size, -1, self.hidden_size]))]
        # this is (batch, prem_steps, hidden dim)
        betas = [tf.squeeze(x, [1]) for x in
                 tf.split(1, self.prem_steps,tf.reshape(betas, [batch_size, -1 , self.hidden_size]))]

        # list of original premise vecs to go with betas
        prem_list = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.prem_steps, premise_inputs)]

        # list of original hypothesis vecs to go with alphas
        hyp_list = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.hyp_steps, hypothesis_inputs)]

        beta_concat_prems = []
        alpha_concat_hyps = []
        for input, rep in zip(prem_list, betas):
            beta_concat_prems.append(tf.concat(1,[input,rep]))

        for input, rep in zip(hyp_list, alphas):
            alpha_concat_hyps.append(tf.concat(1, [input, rep]))

        prem_comparison_vecs = tf.concat(1, [tf.expand_dims(x,1) for x in beta_concat_prems])
        hyp_comparison_vecs = tf.concat(1, [tf.expand_dims(x,1) for x in alpha_concat_hyps])


        with tf.variable_scope("gru_inference"):
            inference = rnn_cell.GRUCell(self.config.inference_size)
            self.inference_cell = rnn_cell.MultiRNNCell([inference]*self.num_layers)
            self.inference_state = self.inference_cell.zero_state(self.batch_size, tf.float32)


        with tf.variable_scope("inference"):
            final_representation, remainders, self.iterations = self.do_inference_steps(self.inference_state,prem_comparison_vecs, hyp_comparison_vecs)


        # softmax over outputs to generate distribution over [neutral, entailment, contradiction]
        softmax_w = tf.get_variable("softmax_w", [self.config.inference_size, 3])
        softmax_b = tf.get_variable("softmax_b", [3])
        self.logits = tf.matmul(final_representation, softmax_w) + softmax_b   # dim (batch_size, 3)

        _, targets = tf.nn.top_k(self.targets)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([self.batch_size])],
                3)
        self.cost = tf.reduce_mean(loss)

        _, logit_max_index = tf.nn.top_k(self.logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(logit_max_index, targets), tf.float32))

        self.per_step_accs, self.per_step_dists = self.evaluate_representation()

        if is_training:

            self.lr = tf.Variable(self.config.learning_rate, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.config.max_grad_norm)

            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def evaluate_representation(self):
        # note this assumes batch size == 1, normal for analysis.
        tf.get_variable_scope().reuse_variables()
        softmax_w = tf.get_variable("softmax_w", [self.config.inference_size, 3])
        softmax_b = tf.get_variable("softmax_b", [3])

        #per_step_accs = []

        def evaluate(act_step):

            logits = tf.matmul(act_step, softmax_w) + softmax_b
            #_, targets = tf.nn.top_k(self.targets)

            #_, logit_max_index = tf.nn.top_k(logits)
            #step_acc = tf.reduce_mean(tf.cast(tf.equal(logit_max_index, targets), tf.float32))

            return tf.nn.softmax(logits)

        def evaluate_acc(act_step):

            logits = tf.matmul(act_step, softmax_w) + softmax_b
            _, targets = tf.nn.top_k(self.targets)

            _, logit_max_index = tf.nn.top_k(logits)
            step_acc = tf.reduce_mean(tf.cast(tf.equal(logit_max_index, targets), tf.float32))

            return step_acc


        per_step_dists = tf.map_fn(evaluate, self.incremental_states)
        per_step_accs = tf.map_fn(evaluate_acc, self.incremental_states)
        return per_step_accs, per_step_dists

    def do_inference_steps(self, initial_state, premise, hypothesis):


        self.one_minus_eps = tf.constant(1.0 - self.config.eps, tf.float32,[self.batch_size])
        self.N = tf.constant(self.config.max_computation, tf.float32,[self.batch_size])


        prob = tf.constant(0.0,tf.float32,[self.batch_size], name="prob")
        prob_compare = tf.constant(0.0,tf.float32,[self.batch_size], name="prob_compare")
        counter = tf.constant(0.0, tf.float32,[self.batch_size], name="counter")
        i = tf.constant(0, tf.int32, name="index")
        acc_states = tf.zeros_like(initial_state, tf.float32, name="state_accumulator")
        batch_mask = tf.constant(True, tf.bool,[self.batch_size])

        # Tensor arrays to collect information about the run:
        array_probs = tf.TensorArray(tf.float32,0, dynamic_size=True)
        premise_attention = tf.TensorArray(tf.float32,0, dynamic_size=True)
        hypothesis_attention = tf.TensorArray(tf.float32,0, dynamic_size=True)
        incremental_states = tf.TensorArray(tf.float32,0, dynamic_size=True)


        # While loop stops when this predicate is FALSE.
        # Ie all (probability < 1-eps AND counter < N) are false.

        pred = lambda i ,incremental_states, array_probs, premise_attention, hypothesis_attention, batch_mask,prob_compare,prob,\
                      counter,state,premise, hypothesis ,acc_state:\
            tf.reduce_any(
                tf.logical_and(
                    tf.less(prob_compare,self.one_minus_eps),
                    tf.less(counter,self.N)))
                # only stop if all of the batch have passed either threshold

            # Do while loop iterations until predicate above is false.
        i,incremental_states, array_probs,premise_attention,hypothesis_attention,_,_,remainders,iterations,_,_,_,state = \
            tf.while_loop(pred,self.inference_step,
            [i,incremental_states, array_probs, premise_attention, hypothesis_attention,
             batch_mask,prob_compare,prob,
             counter,initial_state,premise, hypothesis, acc_states])

        self.ACTPROB = array_probs.pack()
        self.ACTPREMISEATTN = premise_attention.pack()
        self.ACTHYPOTHESISATTN = hypothesis_attention.pack()
        self.incremental_states = incremental_states.pack()

        return state, remainders, iterations

    def inference_step(self, i, incremental_states, array_probs, premise_attention, hypothesis_attention,
        batch_mask, prob_compare,prob,counter, state, premise, hypothesis, acc_states):

        if self.config.keep_prob < 1.0 and self.is_training:
            premise = tf.nn.dropout(premise, self.config.keep_prob)
            hypothesis = tf.nn.dropout(hypothesis,self.config.keep_prob)

        hyp_weights, hyp_attn = self.attention(state, hypothesis, "hyp_attn")
        state_for_premise = tf.concat(1, [state, hyp_attn])
        prem_weights, prem_attn = self.attention(state_for_premise, premise, "prem_attn")
        state_for_gates = tf.concat(1, [state, hyp_attn ,prem_attn, prem_attn * hyp_attn])

        hyp_gate = self.gate_mechanism(state_for_gates, "hyp_gate")
        prem_gate = self.gate_mechanism(state_for_gates, "prem_gate")

        input = tf.concat(1, [hyp_gate * hyp_attn, prem_gate * prem_attn])
        output, new_state = self.inference_cell(input,state)
        incremental_states = incremental_states.write(i, new_state)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear(new_state, 1, True)))


        new_batch_mask = tf.logical_and(tf.less(prob + p,self.one_minus_eps),batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        prob += p * new_float_mask
        prob_compare += p * tf.cast(batch_mask, tf.float32)



        def use_remainder():

            remainder = tf.constant(1.0, tf.float32,[self.batch_size]) - prob
            remainder_expanded = tf.expand_dims(remainder,1)
            tiled_remainder = tf.tile(remainder_expanded,[1,self.config.inference_size])

            ap = array_probs.write(i, remainder)
            ha= hypothesis_attention.write(i, hyp_weights)
            pa = premise_attention.write(i, prem_weights)
            acc_state = (new_state * tiled_remainder) + acc_states
            return ap,ha,pa,acc_state

        def normal():

            p_expanded = tf.expand_dims(p * new_float_mask,1)
            tiled_p = tf.tile(p_expanded,[1,self.config.inference_size])

            ap= array_probs.write(i, p*new_float_mask)
            pa = premise_attention.write(i, prem_weights)
            ha= hypothesis_attention.write(i, hyp_weights)
            acc_state = (new_state * tiled_p) + acc_states
            return ap,ha,pa,acc_state


        counter += tf.constant(1.0,tf.float32,[self.batch_size]) * new_float_mask
        counter_condition = tf.less(counter,self.N)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask,counter_condition))

        array_probs, hypothesis_attention,\
        premise_attention,acc_state = tf.cond(condition, normal, use_remainder)
        i += 1

        return (i, incremental_states, array_probs, premise_attention, hypothesis_attention,
                new_batch_mask, prob_compare,prob,counter, new_state,
                premise, hypothesis, acc_state)


    def gate_mechanism(self, gate_input, scope):

        with tf.variable_scope(scope):

            size = gate_input.get_shape()[1].value
            out_size = 2*self.config.hidden_size

            hidden1_w = tf.get_variable("hidden1_w", [size, size])
            hidden1_b = tf.get_variable("hidden1_b", [size])

            hidden2_w = tf.get_variable("hidden2_w", [size, size])
            hidden2_b = tf.get_variable("hidden2_b", [size])

            sigmoid_w = tf.get_variable("sigmoid_w", [size, out_size])
            sigmoid_b = tf.get_variable("sigmoid_b", [out_size])

            if self.config.keep_prob < 1.0 and self.is_training:
                gate_input = tf.nn.dropout(gate_input, self.config.keep_prob)

            hidden1 = tf.nn.relu(tf.matmul(gate_input, hidden1_w) + hidden1_b)

            if self.config.keep_prob < 1.0 and self.is_training:
                hidden1 = tf.nn.dropout(hidden1, self.config.keep_prob)

            hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)

            gate_output = tf.nn.sigmoid(tf.matmul(hidden2, sigmoid_w) + sigmoid_b)

        return gate_output


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

        return a ,ds

    def feed_forward_attention(self, attendees):
        "Sends 3D tensor through two 2D convolutions with 2 different features"

        attn_length = attendees.get_shape()[1].value
        attn_size = attendees.get_shape()[2].value


        hidden = tf.reshape(attendees, [-1, attn_length, 1, attn_size])
        k1 = tf.get_variable("W1", [1,1,attn_size,attn_size])
        k2 = tf.get_variable("W2", [1,1,attn_size,attn_size])
        b1 = tf.get_variable("b1", [attn_size])
        b2 = tf.get_variable("b2", [attn_size])

        if self.config.keep_prob < 1.0 and self.is_training:
            hidden = tf.nn.dropout(hidden, self.config.keep_prob)

        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hidden, k1, [1, 1, 1, 1], "SAME"),b1))

        if self.config.keep_prob < 1.0 and self.is_training:
            features = tf.nn.dropout(features, self.config.keep_prob)

        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(features, k2, [1,1,1,1], "SAME"), b2))

        return tf.squeeze(features)



    def feedforward_network(self, input):

        hidden_dim = input.get_shape()[1].value

        hidden1_w = tf.get_variable("hidden1_w", [hidden_dim, hidden_dim])
        hidden1_b = tf.get_variable("hidden1_b", [hidden_dim])

        hidden2_w = tf.get_variable("hidden2_w", [hidden_dim, hidden_dim])
        hidden2_b = tf.get_variable("hidden2_b", [hidden_dim])

        if self.config.keep_prob < 1.0 and self.is_training:
            input = tf.nn.dropout(input, self.config.keep_prob)

        hidden1 = tf.nn.relu(tf.matmul(input, hidden1_w) + hidden1_b)

        if self.config.keep_prob < 1.0 and self.is_training:
            hidden1 = tf.nn.dropout(hidden1, self.config.keep_prob)

        gate_output = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)

        return gate_output

