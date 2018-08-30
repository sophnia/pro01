import numpy as np
import matplotlib.pyplot as plt

def scal(x):
    x_mean=np.mean(x,0)#求平均值（x,0）0代表列
    x_std=np.std(x,0)#求均方差
    x-=x_mean
    x/=x_std
    x = np.c_[np.ones(m), x]###
    return x

def cost(x,y,theta):
    J=1.0/(2*m)*(np.sum(np.square(x.dot(theta)-y)))#不要忘记两个括号间的乘号
    return J

def gradient(x,y,theta=[[0],[0],[0]],alpha=0.01,iterms=1500):
    J_histories=np.ones(iterms)#注意循环的次数
    for i in range(iterms):
        J_histories[i]=cost(x,y,theta)
        qd=(1.0/m)*(x.T.dot(x.dot(theta)-y))###注意公式
        theta=theta-alpha*qd
    return J_histories,theta

data=np.loadtxt('ex1data2.txt',delimiter=',')#注意文件后缀名
m=len(data)
x=data[:,:2]#从头打印前两行
a=scal(x)   #特征缩放以后的x输入
y=data[:,2].reshape(-1,1)#(无论多少行都转化成1列)

J_histories,theta=gradient(a,y)
plt.plot(J_histories)
plt.show()


pred_y=a.dot(theta)
print(y.shape)
error=np.hstack((y,pred_y,y-pred_y))
print(error)

plt.plot(y,pred_y,'rx')
plt.show()

