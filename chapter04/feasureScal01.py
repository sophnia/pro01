import numpy as np
import matplotlib.pyplot as plt

#特征缩放
def featureScal(x):
    x_mean=np.mean(x,0) #求平均值（x,0）0代表列
    x_std=np.std(x,0)
    x-=x_mean
    x/=x_std
    x=np.c_[np.ones((m,1)),x]
    return x

#代价函数
def cost(X,y,theta):
    J=(1.0)/(2*m)*(np.sum(np.square(X.dot(theta)-y)))
    return J

#梯度下降                   theta0+两个特征
def gradient(X,y,theta=[[0],[0],[0]],alpha=0.01,iterms=100000):
    J_histories=np.ones(iterms)
    for i in range(iterms):
        J_histories[i]=cost(X,y,theta)
        qd=(1.0/m)*(X.T.dot(X.dot(theta)-y))
        theta=theta-qd*alpha
    return J_histories,theta

#读取数据
data=np.loadtxt('ex1data2.txt',delimiter=',')
m=data.shape[0]
x=data[:,:2]
y=data[:,2].reshape(m,1)

X=featureScal(x)
J_histories,theta=gradient(X,y)

#根据模型得出预测值
pred_y=X.dot(theta)
error=np.hstack((y,pred_y,pred_y-y))
print("真实值  预测值   误差")
print(error)

#画图
plt.plot(y,pred_y,'rx')
plt.show()

plt.plot(J_histories)
plt.show()


