import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np 
import matplotlib.pyplot as plt

def add_layer(inputs, input_size, output_size, n_layer, active_function=None):
	with tf.name_scope('layer'):
		layer_name = 'layer%s'%n_layer
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([input_size, output_size]), name='W')
			tf.summary.histogram(layer_name+'/Weights', Weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, output_size])+0.1, name = 'b')
			tf.summary.histogram(layer_name+'/biases', biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if active_function==None:
		outputs = Wx_plus_b
		tf.summary.histogram(layer_name+'/outputs', outputs)
	else:
		outputs = active_function(Wx_plus_b)
		tf.summary.histogram(layer_name+'/outputs', outputs)

	return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_data')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_data')

l1 = add_layer(xs, 1, 10, 1, active_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, active_function=None)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
	tf.summary.scalar('loss', loss) #查看损失函数loss的变化

with tf.name_scope('train'):
	train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
merged =tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
	sess.run(train, feed_dict={xs:x_data, ys:y_data}) #每一步的训练都sess.run()
	if i%50==0:
		result =sess.run(merged, feed_dict={xs:x_data, ys:y_data})
		writer.add_summary(result, i)
		print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_vale = sess.run(prediction, feed_dict = {xs:x_data})
		lines = ax.plot(x_data, prediction_vale, 'r-', lw=5)w2
		plt.pause(0.1)

plt.pause(0)