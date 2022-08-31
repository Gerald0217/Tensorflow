import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3 #要达到的目的是要使的weigh接近0.1，bias接近0.3
   
#create tensorflow structure start#
Weights = tf.Variable(tf.random.uniform([1],-1,1)) #定义初始值为随机的，[1]表示的是结构，后面的两个数表示的是范围
biases = tf.Variable(tf.zeros([1])) #定义b偏差为0

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))#显示真实值与预测值得差别
optimizer = tf.train.GradientDescentOptimizer(0.5) #建立一个优化器，优化器有很多，这里选择了一个最简单的，0.5表示学习效率(<1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() #初始化所有的参数结构变量
#create tensorflow structure end#
sess = tf.Session()
sess.run(init) #激活了整个神经网络结构，非常的重要

for step in range(201):
	sess.run(train)
	if step%20 == 0:
		print(step, sess.run(Weights), sess.run(biases))