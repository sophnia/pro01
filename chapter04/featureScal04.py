import numpy as np
import matplotlib.pyplot as plt

def featurescal(x):
    x_mean=np.mean(x,0)#求平均值（x,0）0代表列
    x_std=np.std(x,0)#求均方差
    x-=x_mean
    x/=x_std
    x=np.c_[np.ones(m),x]
    return x

def cost(x,y,theta):
    J=1.0/(2*m)*np.sum(np.square(x.dot(theta)-y))
    return J

#梯度下降                   theta0+两个特征
def gradient(x,y,theta=[[0],[0],[0]],alpha=0.01,iterms=1500):
    J_histories=np.ones(iterms)
    for i in range(iterms):
        J_histories[i]=cost(x,y,theta)
        qd=(1.0/m)*(x.T.dot(x.dot(theta)-y))
        theta=theta-alpha*qd
    return J_histories,theta

#读取数据
data=np.loadtxt('ex1data2.txt',delimiter=',')
m=len(data)
x=data[:,:2]#从头打印前两行
y=data[:,2].reshape(-1,1)#(无论多少行都转化成1列)

a=featurescal(x)#特征缩放以后的x输入
J_histories,theta=gradient(a,y)
plt.plot(J_histories)
plt.show()

y_pred=a.dot(theta)
plt.plot(y,y_pred,'rx')
plt.show()

print('真实值                  预测值                     误差')
error=np.hstack((y,y_pred,y-y_pred))
print(error)

