import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.5*x_data + 0.3

#create tensorflow structure start#
Weights = tf.Variable(tf.random.uniform([1],-1,1)) #初始化权重
biases = tf.Variable(tf.zeros([1])) #初始化偏差

y_predict = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y_predict-y_data)) #定义损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5) #选择训练优化器
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() #初始化tensorflow结构中的所有参数
#create tensorflow structure end#
sess = tf.Session() #执行命令的东西
sess.run(init)

for step in range(200):
	sess.run(train)
	if step%20==0:
		print(step, sess.run(Weights), sess.run(biases))
