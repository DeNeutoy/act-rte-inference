

import tensorflow as tf
import numpy as np
import threading




# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

def MyLoop(coord, session, i):
  while not coord.should_stop():

    x,y, _ = session.run([W,b,train])
    print(x,y)
    i +=1
    if i > 200:
      coord.request_stop()

# Main code: create a coordinator.
coord = tf.train.Coordinator()
session = tf.Session()
session.run(tf.initialize_all_variables())
# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,session, 0.0)) for x in range(5)]

# Start the threads and wait for all of them to stop.
for t in threads: t.start()
coord.join(threads)

W,b  = session.run([W, b])

print("final")
print(W, b)








# x = tf.TensorArray(tf.float32,0, dynamic_size=True)
#
#
#
# def while_func(i, x):
#
#     x = x.write(i, tf.constant(3.0, shape=[20,10]))
#     i +=1
#     return (i, x)
#
# i = tf.constant(0)
#
# index, array = tf.while_loop(lambda i, x: tf.less(i,10),while_func,[i,x])
#
# packed = array.pack()
#
# with tf.Session() as sess:
#
#     res = sess.run([packed])
#
#     print(res)
