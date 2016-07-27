from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from datetime import datetime
import config as CONFIG
import snli_reader
import tensorflow as tf
from epoch import extra_epoch
from Vocab import Vocab
from IAAModel import IAAModel
from DAModel import DAModel
from AdaptiveIAAModel import AdaptiveIAAModel
from AdaptiveReader import AdaptiveReader
from embedding_utils import import_embeddings
import saveload
import argparse
import numpy as np
from collections import defaultdict

def get_config_and_model(conf):

    if conf == "DAModel":
        return DAModel, CONFIG.DAConfig()
    elif conf == "IAAModel":
        return IAAModel, CONFIG.IAAConfig()
    elif conf == "AdaptiveIAAModel":
        return AdaptiveIAAModel, CONFIG.AdaptiveIAAConfig()
    elif conf == "AdaptiveReader":
        return  AdaptiveReader, CONFIG.IAAConfig()

def main(unused_args):
    saved_model_path = args.model_path
    vocab_path = args.vocab_path
    weights_dir = args.weights_dir
    verbose = args.verbose
    debug = args.debug
    embeddings = args.embedding_path

    MODEL, config = get_config_and_model(args.model)
    _, eval_config = get_config_and_model(args.model)



    if weights_dir is not None:
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)



    vocab = Vocab(vocab_path, args.data_path,max_vocab_size=40000)

    config.vocab_size = vocab.size()
    eval_config.vocab_size = vocab.size()

    if args.grid_search:

        config.hidden_size = int(args.hidden_size)
        config.learning_rate = float(args.learning_rate)
        config.eps = float(args.eps)
        config.step_penalty = float(args.step_penalty)
        config.keep_prob = float(args.keep_prob)
        eval_config.hidden_size = int(args.hidden_size)
        eval_config.learning_rate = float(args.learning_rate)
        eval_config.eps = float(args.eps)
        eval_config.step_penalty = float(args.step_penalty)
        eval_config.keep_prob = float(args.keep_prob)

    train, val, test = "snli_1.0_train.jsonl","snli_1.0_dev.jsonl","snli_1.0_test.jsonl"

    # Dataset Stats(we ignore bad labels)
    #                   train   val    test
    # max hypothesis:   82      55     30
    # max premise:      62      59     57
    # bad labels        785     158    176

    if debug:
        buckets = [(15,10)]
        raw_data = snli_reader.load_data(args.data_path,train, val, test, vocab, False,
                            max_records=100,buckets=buckets, batch_size=config.batch_size)
    else:
        buckets = [(10,5),(20,10),(30,20),(40,30),(50,40),(82,62)]
        raw_data = snli_reader.load_data(args.data_path,train, val, test, vocab, False,
                            max_records=None, buckets=buckets, batch_size=config.batch_size)

    train_data, val_data, test_data, stats = raw_data
    print(stats)

    # data = list(bucketed data)
    # bucketed data = list(tuple(dict(hypothesis,premise), target))
    # where the sentences will be of length buckets[bucket_id]

    # bucket_id : data
    train_buckets = {x:v for x,v in enumerate(train_data)}
    val_buckets = {x:v for x,v in enumerate(val_data)}
    test_buckets = {x:v for x,v in enumerate(test_data)}

    if config.use_embeddings:
        print("loading embeddings from {}".format(embeddings))
        vocab_dict = import_embeddings(embeddings)

        config.embedding_size = len(vocab_dict["the"])
        eval_config.embedding_size = config.embedding_size

        embedding_var = np.random.normal(0.0, config.init_scale, [config.vocab_size, config.embedding_size])
        no_embeddings = 0
        for word in vocab.token_id.keys():
            try:
                embedding_var[vocab.token_id[word],:] = vocab_dict[word]
            except KeyError:
                no_embeddings +=1
                continue
        print("num embeddings with no value:{}".format(no_embeddings))
    else:
        embedding_var = None

    print("loading models into memory")
    trainingStats = defaultdict(list)
    trainingStats["config"].append(config)
    trainingStats["eval_config"].append(eval_config)
    with tf.Graph().as_default(), tf.Session() as session:
        initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # generate a model per bucket which share parameters
        # store in list where index corresponds to the id of the batch
        with tf.variable_scope("all_models_with_buckets"):
            models, models_val, models_test = [], [], []

            for i in range(len(train_buckets)):
                # correct config for this bucket
                config.prem_steps, config.hyp_steps = buckets[i]
                eval_config.prem_steps, eval_config.hyp_steps = buckets[i]

                with tf.variable_scope('model', reuse= True if i > 0 else None, initializer=initialiser):

                    models.append(MODEL(config, pretrained_embeddings=embedding_var,
                                        update_embeddings=config.train_embeddings, is_training=True))


                with tf.variable_scope('model', reuse=True):
                    models_val.append(MODEL(config, pretrained_embeddings=embedding_var,
                                            update_embeddings=config.train_embeddings, is_training=False))
                    models_test.append(MODEL(eval_config, pretrained_embeddings=embedding_var,
                                             update_embeddings=config.train_embeddings, is_training=False))

                tf.initialize_all_variables().run()

                if config.use_embeddings:
                    session.run([models[0].embedding_init],feed_dict={models[0].embedding_placeholder:embedding_var})


                        #### Reload Model ####
                if saved_model_path is not None:
                    saveload.main(saved_model_path, session)
                    try:
                        trainingStats = pickle.load(open(os.path.join(weights_dir,"stats.pkl"), "rb"))
                    except:
                        print("unable to rejoin original statistics - ignore if not continuing training.")

            epochs = [i for i in range(config.max_max_epoch)]
            epochs = epochs[len(trainingStats["epoch"]):]
            print("beginning training")

            for i in epochs:
                #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                #session.run(tf.assign(m[model].lr, config.learning_rate * lr_decay))
                print("Epoch {}, Training data".format(i + 1))
                train_loss, train_acc, train_mean, train_var = extra_epoch(session, models, train_buckets,training=True, verbose=True)
                print ("Epoch {}, Validation data".format(i + 1))
                valid_loss, valid_acc, val_mean, val_var = extra_epoch(session, models_val, val_buckets,training=False)

                trainingStats["train_loss"].append(train_loss)
                trainingStats["train_acc"].append(train_acc)
                trainingStats["train_step_mean"].append(train_mean)
                trainingStats["train_step_var"].append(train_var)
                trainingStats["val_step_mean"].append(val_mean)
                trainingStats["val_step_var"].append(val_var)
                trainingStats["val_loss"].append(valid_loss)
                trainingStats["val_acc"].append(valid_acc)
                trainingStats["epoch"].append(i)

                # if trainingStats["val_acc"][i-1] >= trainingStats["val_acc"][i]:
                #     print("decaying learning rate")
                #     trainingStats["lr_decay"].append(i)
                #     current_lr = session.run(models[0].lr)
                #     session.run(tf.assign(models[0].lr, config.lr_decay * current_lr))

                print("Epoch: {} Train Loss: {} Train Acc: {}".format(i + 1, train_loss, train_acc))
                print("Epoch: {} Valid Loss: {} Valid Acc: {}".format(i + 1, valid_loss, valid_acc))

                #######    Model Hooks    ########
                if weights_dir is not None:
                    date = "{:%m.%d.%H.%M}".format(datetime.now())
                    file ="/Epoch_{:02}Train_{:0.3f}Val_{:0.3f}date{}.pkl".format(i+1,train_acc,valid_acc, date)
                    saveload.main(weights_dir + file, session)
                    trainingStats["epoch_files"].append(file)
                    pickle.dump(trainingStats,open(os.path.join(weights_dir, "stats.pkl"), "wb"))

            # load weights with best validation performance:

            best_epoch = np.argmax(trainingStats["val_acc"])
            file = trainingStats["epoch_files"][best_epoch]
            saveload.main(weights_dir + file, session)

            test_loss, test_acc, test_mean, test_var = extra_epoch(session, models_test, test_buckets, training=False)
            date = "{:%m.%d.%H.%M}".format(datetime.now())

            trainingStats["test_loss"].append(test_loss)
            trainingStats["test_acc"].append(test_acc)
            trainingStats["test_step_mean"].append(test_mean)
            trainingStats["test_step_var"].append(test_var)
            file = "/BestEpoch_{}FinalTestAcc_{:0.5f}date{}.pkl".format(best_epoch,test_acc, date)
            trainingStats["test_file"].append(file)
            saveload.main(weights_dir + file, session)

            pickle.dump(trainingStats,open(os.path.join(weights_dir, "stats.pkl"), "wb"))

            if verbose:
                print("Test Accuracy: {}".format(test_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument("--weights_dir")
    parser.add_argument("--verbose")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--grid_search", action='store_true', default=False)
    parser.add_argument("--vocab_path")
    parser.add_argument("--embedding_path")
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--step_penalty", type=float)
    parser.add_argument("--keep_prob", type=float)


    from sys import argv

    args = parser.parse_args()
    main(argv)
