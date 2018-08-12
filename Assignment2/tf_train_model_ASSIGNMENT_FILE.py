import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class build_train:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   # DO NOT EDIT
        self.save_dir = self.rootPath + '/tf_model'                             # DO NOT EDIT

    def build_train_network(self, network):

        ############### MNIST DATA #########################################
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # DO NOT EDIT
        ############### END OF MNIST DATA ##################################

        ############### CONSTRUCT NEURAL NETWORK MODEL HERE ################

        # MODEL
        # INPUT MUST BE 784 array in order be able to train on MNIST
        # INPUT PLACEHOLDERS MUST BE NAME AS name='ph_x' AND name='ph_y_'
        '''
        Follow following format for defining placeholders:
        x = tf.placeholder(data_type, array_shape, name='ph_x')
        y_ = tf.placeholder(data_type, array_shape, name='ph_y_')
        '''
        # OUTPUT VECTOR y MUST BE LENGTH 10, EACH OUTPUT NEURON CORRESPONDS TO A DIGIT 0-9
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            # returns a tensor of the specified shape filled with random truncated normal values,
            # with given standard deviation of the non-truncated normal distribution
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape = shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        x = tf.placeholder(tf.float32, [None, 784], name = 'ph_x')
        y_ = tf.placeholder(tf.float32, [None, 10], name = 'ph_y_')

        # First Convolutional Layer

        W_conv1 = weight_variable([5, 5, 1, 32]) # convolution computes 32 features for each 5x5
                                                 # patch, with 1 input channel
        b_conv1 = bias_variable([32]) # bias for each output channel (32 features)

        x_image = tf.reshape(x, [-1, 28, 28, 1]) # 28x28 image, last 1 denotes 1 color channel
        print(x_image)

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Second Convolutional Layer

        W_conv2 = weight_variable([5, 5, 32, 64]) # computes 64 features for each 5x5
                                                  # patch, with 32 input channels (from W_conv1)
        b_conv2 = bias_variable([64]) # bias for each output channel (64 features)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Densely Connected Layer
        # Image size has been reduced to 7x7, add a fully connected layer with 1024 neurons
        # to allow processing on the entire image.

        W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 7x7 "image" with 64 features
                                                    # (output channels from h_pool2)
        b_fc1 = bias_variable([1024]) # bias each output channel from W_fc1

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout Step

        #keep_prob = tf.placeholder(tf.float32)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout Layer

        W_fc2 = weight_variable([1024, 10]) # 1024 inputs, 10 outputs
        b_fc2 = bias_variable([10])

	#y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name = 'op_y')

        # Training and Evaluating the Model

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='op_loss')

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1), name='op_correct')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='op_accuracy')
        ############# END OF NEURAL NETWORK MODEL ##########################

        ############# CONSTRUCT TRAINING FUNCTION ##########################

        # TRAINING FUNCTION SHOULD USE YOUR LOSS FUNCTION TO OPTIMIZE THE MODEL PARAMETERS

        #train_step = tf.train.GradientDescentOptimizer(0.45).minimize(cross_entropy, name='op_train')
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='op_train')


        ############# END OF TRAINING FUNCTION #############################


        ############# CONSTRUCT TRAINING SESSION ###########################
        saver = tf.train.Saver()                                            # DO NOT EDIT
        sess = tf.InteractiveSession()                                      # DO NOT EDIT
        sess.run(tf.global_variables_initializer())                         # DO NOT EDIT
        acc_train = []
        acc_valid = []
        acc_test = []
        for i in range(0, 10001):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})#, keep_prob: 1.0})
                acc_train.append(train_accuracy)
                print("step %d, training accuracy %g" % (i, train_accuracy))

                batch = mnist.validation.next_batch(100)
                validation_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})#, keep_prob: 1.0})
                acc_valid.append(validation_accuracy)
                print("step %d, validation accuracy %g" % (i, validation_accuracy))

                batch = mnist.test.next_batch(100)
                test_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})#, keep_prob: 1.0})
                acc_test.append(test_accuracy)
                print("step %d, testing accuracy %g" % (i, test_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1]})#, keep_prob: 0.5})

        #print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        ############# END OF TRAINING SESSION ##############################

        ############# SAVE MODEL ###########################################

        saver.save(sess, save_path=self.save_dir, global_step=network)      # DO NOT EDIT
        print('Model Saved')                                                # DO NOT EDIT
        sess.close()                                                        # DO NOT EDIT
        ############# END OF SAVE MODEL ####################################

        ############# OUTPUT ACCURACY PLOT ################################
        x = range(0, 10100, 100)
        plt.figure(1)
        ax = plt.subplot()
        ax.plot(x, acc_train, 'r-', label = 'Training')
        ax.plot(x, acc_valid, 'b-', label = 'Validation')
        ax.plot(x, acc_test, 'g-', label = 'Testing')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        ax.set_title('Plot of Accuracy over 10000 Iterations')
        ax.grid(True)
        ax.legend(loc = 'lower right', fancybox = True, shadow = True)
        plt.show()

############# END OF ACCURACY PLOT ################################
