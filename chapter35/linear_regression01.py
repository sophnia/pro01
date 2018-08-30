import tensorflow as tf
#生成随机种子数  让每次随机的数值是一样的
tf.set_random_seed(777)

x_train=[1,2,3]
y_train=[1,2,3]

#随机正太分布随机数
W=tf.Variable(tf.random_normal([1]),name='weight')#初始化
b=tf.Variable(tf.random_normal([1]),name='bias')

#预测值
hypothesis=x_train*W+b

#损失函数
cost=tf.reduce_mean(tf.square(hypothesis-y_train))

#最小值
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#进行图像会话

sess=tf.Session()
#对变量进行初始化
sess.run(tf.global_variables_initializer())

#拟合的线
for step in range(2001):
    sess.run(train)
    if step%20==0:
        #cost是损失函数，当损失函数最小的时候，找到w（theta1---theta n）和b(theta0)
        #损失函数通过梯度下降进行的
        print(step,sess.run(cost),sess.run(W),sess.run(b))
