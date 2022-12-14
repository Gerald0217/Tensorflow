import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, input_size, output_size, activation_function=None):
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):

			Weights = tf.Variable(tf.random_normal([input_size, output_size]), name = 'W')
		with tf.name_scope('biases'):	
			biases =  tf.Variable(tf.zeros([1, output_size]) + 0.1, name='b')
		with tf.name_scope('Wx_plus_b'):	
			Wx_plus_b = tf.matmul(inputs, Weights) + biases

		if  activation_function==None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None, 1], name='x_data')
	ys = tf.placeholder(tf.float32,[None, 1], name = 'y_data')

l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function = None)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys), reduction_indices=[1]))
with tf.name_scope('train_step'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()  #防止画了图就停止了，这行代码可以让程序继续运行下去
plt.show()

for i in range(5000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i%50==0:
		# print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:x_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
		plt.pause(0.1)
plt.pause(0)

