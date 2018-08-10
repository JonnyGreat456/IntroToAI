def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# Loading MNIST Data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
"""
    mnist: class that stores the training, validation, and test sets as NumPy arrays, also
           has a function for iterating through data mini-batches
"""

import tensorflow as tf

sess = tf.InteractiveSession() # allows you to interleave operations which build a computational graph
                               # with ones that run the graph

# Building Softmax Regression Model

x = tf.placeholder(tf.float32, shape = [None, 784]) # input images
"""None - first dimension of x can be of any size; 784 - dimensionality of 28x28 pixel image"""
y_ = tf.placeholder(tf.float32, shape = [None, 10]) # output targets
"""10 - each row of y is a one-hot 10-D vector indicating which digit class (0-9) the corresponding image belongs to"""

W = tf.Variable(tf.zeros([784,10])) # matrix of weights to be tweaked according to model training
b = tf.Variable(tf.zeros([10])) # biases for each of the ten classes

sess.run(tf.global_variables_initializer()) # initializing all variables

y = tf.matmul(x, W) + b # regression model (multiply inputs by weights and add bias to output predictions)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
"""
    nn.softmax_cross_entropy_with_logits applies softmax function to the model's prediction and sums acros all classes
    reduce_mean takes the average over the sums
"""

# Training the Model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
"""uses a SGD optimizer to minimize the cross_entropy function with a learning rate of 0.5"""

for _ in range(1000):
    batch = mnist.train.next_batch(100) # takes 100 training examples from mnist and trains on those
    train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

# Evaluating the Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
