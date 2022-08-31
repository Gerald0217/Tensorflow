#激活函数：为了解决现实生活中不能用线性函数来概括的问题#
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import numpy as np
import matplotlib.pyplot as plt 

#定义了一个神经层的函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
	'可视化隐藏层'
	layer_name = 'layer%s'%n_layer
	with tf.name_scope('layer_name'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]),  name='W')
			tf.summary.histogram(layer_name+'/weights', Weights) #查看权重的变化
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='b') #ML中推荐的baises是不为0的，所以加上0.1
			tf.summary.histogram(layer_name+'/biases', biases) #查看偏差的变化
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b) 
		tf.summary.histogram(layer_name+'/outputs', outputs) #查看输出的变化

	return outputs

#输入层有多少个data就有多少个神经元，输出层和输入层一样
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] #-1到1之间间隔均匀的300个数据,300行1列的矩阵
noise = np.random.normal(0, 0.05, x_data.shape) #从正态分布中绘制随机样本，0是均值，0.05是方差。高斯噪声
y_data = np.square(x_data)-0.5 + noise

#下面是为了可视化输入层
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input') #name是表示变量的名字
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#隐藏层的神经元个数可以自己假设了,这里假设了有10个神经元，所以outsize是10
#n_layer=1表示第一层神经层，n_layer=2表示第二层神经层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu) #这里的1指的是只有一个x_data,10指的是神经元的个数
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None) #这里的10是神经元的个数，1表示输出数据的个数

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))#reduction_indices=[1]表示按照行来运算
	tf.summary.scalar('loss', loss) #查看损失函数loss的变化

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()

#把可视了的神经网络图放在程序所在的文件夹里新建一个logs文件夹
writer = tf.summary.FileWriter("logs/", sess.graph) 
#查看logs文件夹里的图片，第一步：cmd+R打开终端，切换到程序所在的目录，第二步：运行tensorboard --logdir logs/,第三步：复制最后的网址进行查看
sess.run(init) #激活initialize

fig = plt.figure() #生成一个图片框
ax = fig.add_subplot(1, 1, 1) #图像的编号
ax.scatter(x_data, y_data) #这个是真实值的图像
plt.ion() #show了以后不暂停，还是继续往下运行住程序
plt.show()

for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data}) #传入数据进行一步一步的训练
	if i%50==0:
		result =sess.run(merged, feed_dict={xs:x_data, ys:y_data})
		writer.add_summary(result, i)
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		# print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		prediction_value = sess.run(prediction, feed_dict={xs:x_data})#因为prediction和xs有关
		lines = ax.plot(x_data, prediction_value, 'r-', lw = 5) #这里使用曲线进行拟合
		plt.pause(0.1)

plt.pause(0) #防止画完图之后图像消失