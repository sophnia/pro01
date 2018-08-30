import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy=np.loadtxt('data-01-test-score.csv',delimiter=',',dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]
print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.matmul(x,w)+b


