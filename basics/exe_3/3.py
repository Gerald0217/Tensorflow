import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

input1 = tf.placeholder(tf.float32)#一般情况下TensorFlow只能处理32的数据

input2= tf.placeholder(tf.float32)#一般情况下TensorFlow只能处理32的数据

output = tf.multiply(input1, input2) #两个变量的一个惩罚运算

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1:[7], input2:[2]}))

