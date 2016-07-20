import time
import numpy as np
from random import shuffle
from itertools import chain
import tensorflow as tf

def bucket_shuffle(dict_data):
    # zip each data tuple with it's bucket id.
    # return as a randomly shuffled iterator.
    id_to_data =[]
    for x, data in dict_data.items():
        id_to_data += list(zip([x]*len(data), data))

    shuffle(id_to_data)

    return len(id_to_data), iter(id_to_data)


def run_epoch(session, models, data, training, verbose=False):
    """Runs the model on the given data."""
    #epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    accuracy = 0.0
    first_acc = 0.0
    epoch_size, id_to_data = bucket_shuffle(data)

    for step, (id,(x, y)) in enumerate(id_to_data):
        m = models[id]
        assert x["premise"].shape == (m.premise.get_shape())
        assert x["hypothesis"].shape == (m.hypothesis.get_shape())

        if training:
            eval_op = m.train_op
        else:
            eval_op = tf.no_op()

        batch_acc, cost, _ = session.run([m.accuracy, m.cost, eval_op], feed_dict={m.premise: x["premise"],
                                      m.hypothesis: x["hypothesis"],
                                      m.targets: y})

        costs += cost
        iters += 1
        accuracy += batch_acc
        #if verbose and step % (epoch_size // 10) == 10:
        print("%.3f acc: %.3f loss: %.3f speed: %.0f examples/s" %
              (step * 1.0 / epoch_size,
               accuracy / iters,
               costs / iters,
               iters * m.batch_size / (time.time() - start_time)))


    return (costs / iters), (accuracy / iters)


def extra_epoch(session, models, data, training, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    accuracy = 0.0
    avg_steps = 0.0
    var_steps = 0.0
    epoch_size, id_to_data = bucket_shuffle(data)

    for step, (id,(x, y)) in enumerate(id_to_data):
        m = models[id]
        assert x["premise"].shape == (m.premise.get_shape())
        assert x["hypothesis"].shape == (m.hypothesis.get_shape())

        if training:
            eval_op = m.train_op
        else:
            eval_op = tf.no_op()

        batch_acc, cost,act_steps, _ = session.run([m.accuracy, m.cost, m.iterations, eval_op], feed_dict={m.premise: x["premise"],
                                      m.hypothesis: x["hypothesis"],
                                      m.targets: y})

        avg_steps += np.mean(act_steps)
        var_steps += np.sum(np.square(act_steps))
        costs += cost
        iters += 1
        accuracy += batch_acc
        #if verbose and step % (epoch_size // 10) == 10:
        print("%.3f acc: %.3f loss: %.3f speed: %.0f examples/s" %
              (step * 1.0 / epoch_size,
               accuracy / iters,
               costs / iters,
               iters * m.batch_size / (time.time() - start_time)))

    variance = var_steps/iters*m.batch_size - np.square(avg_steps/iters)
    return (costs / iters), (accuracy / iters), (avg_steps/iters), variance