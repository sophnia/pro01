import tensorflow as tf
#生成随机种子数
tf.set_random_seed(777)

X=[1,2,3]
Y=[1,2,3]

W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=X*W+b

#损失函数
cost=tf.reduce_mean(tf.square(hypothesis-Y))


#梯度下降
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

ses=tf.Session()
ses.run(tf.global_variables_initializer())

for step in range(2001):
    cost1,w_,b_,_=ses.run([cost,W,b,train],
        feed_dict={X,Y})
    if step%20==0:
        print(step,cost1,W,b)


