import tensorflow.compat .v1 as tf 
tf.disable_v2_behavior()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#load data 

digits = load_digits()
X = digits.data #加载0-9的图片数据
y = digits.target #
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size])+0.1)

	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

	#考虑到过拟合的问题，一般是对Wx_plus_b进行选择性的dropout

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	tf.summary.histogram(n_layer+'/outputs', outputs)

	return outputs

#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

#add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function = tf.nn.tanh) #使用这个激活函数可以可以避免有些变量不能进行梯度下降
prediction = add_layer(l1, 50, 10, 'l2', activa去tion_function = tf.nn.softmax)


#the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) #用交叉熵来表示损失函数loss
tf.summary.scalar('loss', cross_entropy) #查看cross_entropy的变化
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.summary.merge_all()

#summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
sess.run(init)

for i in range(500):
	sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.25})#丢弃掉75%的节点，这个比例比较好。一般情况下不宜太大和太小，这是两端进行变化的
	if i%50==0:
		train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
		test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})

		train_writer.add_summary(train_result, i)
		test_writer.add_summary(test_result, i)