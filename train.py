from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from datetime import datetime
import ACTconfig as cf
import snli_reader
import tensorflow as tf
from epoch import run_epoch
from Vocab import Vocab
from IAAModel import IAAModel
import saveload
import argparse


def get_config(conf):

    if conf == "small":
        return cf.SmallConfig
    elif conf == "medium":
        return cf.MediumConfig
    elif conf == "large":
        return cf.LargeConfig

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.0005
  max_grad_norm = 5     # changed from 5
  num_layers = 2
  prem_steps = 20
  hyp_steps = 20
  hidden_size = 200 # should be 200
  max_epoch = 4
  max_max_epoch = 13
  lr_decay = 0.5
  batch_size = 20 #changed from 20
  vocab_size = 10000


def main(unused_args):

    config = get_config(args.model_size)
    eval_config = get_config(args.model_size)
    saved_model_path = args.model_path
    vocab_path = args.vocab_path
    weights_dir = args.weights_dir
    verbose = args.verbose
    debug = args.debug


    if weights_dir is not None:
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

    vocab = Vocab(vocab_path, args.data_path,max_vocab_size=20000)
    config.vocab_size = vocab.size()
    eval_config.vocab_size = vocab.size()

    train, val, test = "snli_1.0_train.jsonl","snli_1.0_dev.jsonl","snli_1.0_test.jsonl"

    if debug:
        buckets = []
        raw_data = snli_reader.load_data(args.data_path,train, val, test, vocab, False,
                            max_records=10000,buckets=buckets, max_len=(20,20), batch_size=config.batch_size)
    else:
        buckets = [(20,15),(30,45)]
        raw_data = snli_reader.load_data(args.data_path,train, val, test, vocab, False,
                            max_records=1000000, buckets=buckets, max_len=(60,60), batch_size=config.batch_size)

    train_data, val_data, test_data = raw_data

    # tuples of (data, stats), where:
    # data = list(bucketed data)
    # bucketed data = tuple()

    # dictionaries of bucket id : list(tuple(sentences, target))
    # where the sentences will be of length buckets[bucket_id]
    train_buckets = {x:v for x,v in enumerate(train_data[0])}
    val_buckets = {x:v for x,v in enumerate(val_data[0])}
    test_buckets = {x:v for x,v in enumerate(test_data[0])}


    print("loading models into memory")
    with tf.Graph().as_default(), tf.Session() as session:
        initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("all_models_with_buckets"):
            m, m_val, m_test = [], [], []
            for i in range(len(train_buckets)):

                # correct config for this bucket
                config.prem_steps, config.hyp_steps = buckets[i]
                eval_config.prem_steps, eval_config.hyp_steps = buckets[i]

                with tf.variable_scope('model', reuse= True if i > 0 else None, initializer=initialiser):

                    m.append(IAAModel(config, is_training=True))

                    #### Reload Model ####
                    if saved_model_path is not None:
                        saveload.main(saved_model_path, session)

                with tf.variable_scope('model', reuse=True):
                    m_val.append(IAAModel(config, is_training=False))
                    m_test.append(IAAModel(eval_config,is_training=False))

                tf.initialize_all_variables().run()

            print("beginning training")
            for i in range(config.max_max_epoch):
                #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                #session.run(tf.assign(m[model].lr, config.learning_rate * lr_decay))
                print("Epoch {}, Training data")
                train_loss, train_acc = run_epoch(session, m, train_buckets,training=True, verbose=True)
                print ("Epoch {}, Validation data")
                valid_loss, valid_acc = run_epoch(session, m_val, val_buckets,training=False)

                print("Epoch: {} Train Loss: {} Train Acc: {}".format(i + 1, train_loss, train_acc))
                print("Epoch: {} Valid Loss: {} Valid Acc: {}".format(i + 1, valid_loss, valid_acc))

                #######    Model Hooks    ########
                if weights_dir is not None:
                    date = "{:%m.%d.%H.%M}".format(datetime.now())
                    saveload.main(weights_dir + "/Epoch_{:02}Train_{:0.3f}Val_{:0.3f}date{}.pkl"
                                  .format(i+1,train_acc,valid_acc, date), session)


            test_loss, test_acc = run_epoch(session, m_test, test_buckets, training=False)
            if verbose:
                print("Test Accuracy: {}".format(test_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size")
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument("--weights_dir")
    parser.add_argument("--verbose")
    parser.add_argument("--debug")
    parser.add_argument("--vocab_path")

    from sys import argv

    args = parser.parse_args()
    main(argv)
