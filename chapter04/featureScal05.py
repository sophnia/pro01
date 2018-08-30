#特征缩放

import numpy as np
import matplotlib.pyplot as plt

#特征缩放

def scal(x):
    x_mean=np.mean(x,0)
    x_std=np.std(x,0)
    x-=x_mean
    x_std/=x_std
    x=np.c_[np.ones(m),x]
    return x

#代价函数
def cost(x,y,theta):
    J=1.0/(2*m)*(np.sum(np.square(x.dot(theta)-y)))
    return J

#梯度下降

def gradient(x,y,theta=[[0],[0],[0]],alpha=0.01,iterms=15000):
    J_histories=np.ones(iterms)
    for i in range(iterms):
        J_histories[i]=cost(x,y,theta)
        qd=(1.0/m)*(x.T.dot(x.dot(theta)-y))
        theta=theta-alpha*qd
    return J_histories,theta

#读取数据
data=np.loadtxt('ex1data2.txt',delimiter=',')
m=len(data)
x=data[:,:2]
y=data[:,2].reshape(-1,1)
a=scal(x)

J_histories,theta=gradient(a,y)
plt.plot(J_histories)
plt.show()

pred_y=a.dot(theta)
plt.plot(y)
plt.show()


