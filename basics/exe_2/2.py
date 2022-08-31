import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

matrix1 = tf.constant([[3, 3]]) #一个1行2列的矩阵
matrix2 = tf.constant([[2],[2]]) #一个2行1列的矩阵

product = tf.matmul(matrix1, matrix2) #matrix multiply,numpy中的 np.dot(m1, m2)矩阵相乘

#下面是session的两种打开方式
#method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#method 2
#会自动的关闭sess
with tf.Session() as sess:
	reult2 = sess.run(product)
	print(result2)
