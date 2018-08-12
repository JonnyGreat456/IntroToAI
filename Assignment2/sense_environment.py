import numpy as np
import sys
import random
import os
import tensorflow as tf
from PIL import Image









#############################################################################
#############################################################################
#############################################################################
#############################################################################
class sensor:
	def __init__(self):
		self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   # DO NOT EDIT
		self.save_dir = self.rootPath + '/Model/tf_model'                       # DO NOT EDIT

	def predict(self, input_x, model_version=0):
		out = ""
		if len(input_x) != (28*28):
			print "The input image or input array is shaped incorrectly. Expecting a 28x28 image."
		for i in xrange(0,28):
			out = out+"\n"
			for j in xrange(0,28):
				if input_x[(i*28)+j]>0.5:
					out = out + "1"
				else:
					out = out + "0"
		print "Input image array: \n", out

		os.chdir(self.rootPath)
		print os.getcwd()
		filename_x = "text.txt"
		file_ = open(filename_x, "r")

		filename = self.save_dir + '-' + str(model_version) + '.meta'           # DO NOT EDIT
		filename2 = self.save_dir + '-' + str(model_version)                    # DO NOT EDIT
		print('Opening Model: ' + str(filename))                                # DO NOT EDIT
		saver = tf.train.import_meta_graph(filename)                            # DO NOT EDIT
		sess = tf.InteractiveSession()                                          # DO NOT EDIT

		os.chdir('.ros')

		# INITIALIZE GLOBAL VARIABLES
		sess.run(tf.global_variables_initializer())                             # DO NOT EDIT

		# RESTORE MODEL LOSS OP AND INPUT PLACEHOLDERS
		saver.restore(sess, filename2)                                          # DO NOT EDIT
		graph = tf.get_default_graph()                                          # DO NOT EDIT


		# accuracy = graph.get_tensor_by_name('op_accuracy:0')
		x = graph.get_tensor_by_name('ph_x:0')
		# y_ = graph.get_tensor_by_name('ph_y_:0')
		y = graph.get_tensor_by_name('op_y:0')
		output = tf.arg_max(y,1)

		input_x = np.reshape(input_x, (-1, 784))

		print('Classifying Image...')
		output = sess.run(output, feed_dict={x: input_x})
		sess.close()

		print output

		return output[0]

def predict(input_x):
	predicter = sensor()
	return predicter.predict(input_x)
#############################################################################
#############################################################################
#############################################################################
#############################################################################















if len(sys.argv) < 1:
	print "The script should be passed the full path to the image location"
filename = sys.argv[1]
# full_image = Image.open('$PRACSYS_PATH/prx_output/images/_0.jpg')
full_image = Image.open(filename)
size = 28,28
image = full_image.resize(size, Image.ANTIALIAS)
width, height = image.size
pixels = image.load()
print width, height
fill = 1
array = [[fill for x in range(width)] for y in range(height)]

for y in range(height):
    for x in range(width):
        r, g, b = pixels[x,y]
        lum = 255-((r+g+b)/3)
        array[y][x] = float(lum/255)

image_array = []
for arr in array:
    for ar in arr:
    	image_array.append(ar)
im_array = np.array(image_array)
print image_array
print im_array
out = predict(im_array)

outfile = "/".join(filename.split("/")[:-1])+"/predict.ion"
outf = open(outfile, 'w')
outf.write(str(out))
