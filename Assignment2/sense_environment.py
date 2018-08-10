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
def predict(input_x):
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

	filename = 'prx_core_ws/src/prx_core/sensing/tf_model-0.meta'
        filename2 = 'prx_core_ws/src/prx_core/sensing/tf_model-0'
        print('Opening Model: ' + str(filename))
        saver = tf.train.import_meta_graph(filename)
        sess = tf.InteractiveSession()

        # INITIALIZE GLOBAL VARIABLES
        sess.run(tf.global_variables_initializer())

        # RESTORE MODEL LOSS OP AND INPUT PLACEHOLDERS
        saver.restore(sess, filename2)
        graph = tf.get_default_graph()


        accuracy = graph.get_tensor_by_name('op_accuracy:0')
        x = graph.get_tensor_by_name('ph_x:0')
        y_ = graph.get_tensor_by_name('ph_y_:0')

	y = graph.get_tensor_by_name('op_y')


	prediction = sess.run(y_, feed_dict={x: input_x})
	print prediction
	# prediction = int(random.random()*9.9) #Current prediction is random
	return prediction
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
