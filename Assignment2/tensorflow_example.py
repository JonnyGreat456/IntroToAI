import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32) # weights
b = tf.Variable([-.3], tf.float32) # bias
# Model input and output
x = tf.placeholder(tf.float32) # use for inputs
linear_model = W * x + b
y = tf.placeholder(tf.float32) # use for targets
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) # makes a Stochastic Gradient Descent Optimizer with learning rate = 0.01
train = optimizer.minimize(loss) # minimizes loss function
# training data
x_train = [1,2,3,4] # training input
y_train = [0,-1,-2,-3] # training targets
# training loop
init = tf.global_variables_initializer() # initializes all tf.___ tensors to initial values
sess = tf.Session() # creates a Session object
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train}) # for 1000 iterations, tweak independent variables
  # in order to minimize the loss function

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
