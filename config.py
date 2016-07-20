from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.001
  max_grad_norm = 5     # changed from 5
  num_layers = 2
  hidden_size = 200 # should be 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 0.8
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  embedding_reg = None

class IAAConfig(object):

  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  encoder_size = 128
  inference_size = 512
  hidden_size = 384
  max_epoch = 4
  max_max_epoch = 16
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000
  bidirectional = False

  embedding_size = 300
  embedding_reg = 0.0001
  train_embeddings = True
  use_embeddings = False

class DAConfig(object):

  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 16
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000

  embedding_size = 300
  embedding_reg = None
  train_embeddings = False
  use_embeddings = True

class AdaptiveIAAConfig(object):

  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  encoder_size = 128
  inference_size = 512
  hidden_size = 384
  max_epoch = 4
  max_max_epoch = 16
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000
  bidirectional = False

  embedding_size = 300
  embedding_reg = 0.0001
  train_embeddings = True
  use_embeddings = False

  eps = 0.01
  max_computation = 20
  step_penalty = 0.00001



  """Debugging code for models"""
  # import sys
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     m = self.lstm_model
#     n = self.rnn_model
#     o = self.linear_model
#     from numpy.random import randint
#     x = randint(0,10,[batch_size, num_steps])
#     y = randint(0,10,[batch_size, num_steps])
#
#     out = sess.run(self.train_op,
#                    {m.input_data: x,
#                     m.targets: y,
#                     m.initial_state: self.lstm_model.initial_state.eval(),
#                     n.input_data: x,
#                     n.targets: y,
#                     n.initial_state: self.rnn_model.initial_state.eval(),
#                     o.input_data: x,
#                     o.targets: y,
#                     o.initial_state: self.linear_model.initial_state.eval()})
#     print(out)
# sys.exit()

