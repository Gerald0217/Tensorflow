import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

#Save to file
#remember to define the same dtype and shape when restore


# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

# init = tf.initialize_all_variables()

# saver = tf.train.Saver()

# with tf.Session() as sess:
# 	sess.run(init)
# 	save_path = saver.save(sess, 'my_net/save_net.ckpt')
# 	print("Saver to path:", save_path)

#restore variables
#redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

#not need init step 
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, "my_net/save_net.ckpt")
	print("weights", sess.run(W))
	print("biases", sess.run(b))
