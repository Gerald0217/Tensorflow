import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.10)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	#stride[1, x_movement, y_movement, 1]
	#Must have stride[0] = 

	return tf.nn.conv2d(x, W, strides=[1, 1, 1,1], padding = 'SAME')

def max_pool_2x2(x):
	#'卷积层会选择性的丢失一些数据， 池化层的目的是提高数据的准确度'
	#stride[1, x_movement, y_movement, 1]
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
	

#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#这里的-1位置是表示有多少这样的图片，28,28是表示每张图片的像素点，1表示的是图片的颜色是黑白的
# print(x_image.shape) #[n_shape, 28, 28, 1]

#conv1 layer 
W_conv1 = weight_variable([5, 5, 1, 32]) #pach:5x5, in size:1, out size:32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                          #output size 14*14*32


#conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64]) #pach:5x5, in size:1, out size:32
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                          #output size 7*7*64

##function1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #[n_samples, 7, 7, 64]变换为[n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##function2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) #交叉熵作为损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #换了优化器

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.25})
	if i%50==0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))









