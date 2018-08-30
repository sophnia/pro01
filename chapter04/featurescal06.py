import numpy as np
import matplotlib.pyplot as plt

def fea_scal(x):
    x_mean=np.mean(x,0)
    x_td=np.std(x,0)
    x-=x_mean
    x/=x_td
    x=np.c_[np.ones(m),x]
    return x

#代价函数
def cost(x,y,theta):
#   J = 1.0 / (2 * m) * (np.sum(np.square(x.dot(theta) - y)))
    J = 1.0/ (2 * m) * (np.sum(np.square(x.dot(theta)- y)))
    return J

#梯度下降
def gradient(x,y,theta=[[0],[0],[0]],alpha=0.01,iterms=1500):
    J_histories=np.ones(iterms)#注意是iterms
    for i in range(iterms):
        J_histories[i]=cost(x,y,theta)
        qd=(1.0/m)*(x.T.dot(x.dot(theta)-y))
        theta=theta-qd*alpha
    return J_histories,theta

#读取数据
data=np.loadtxt('ex1data2.txt',delimiter=',')
m=len(data)
X=data[:,:2]
y=data[:,2].reshape(-1,1)
#特征缩放后的输出
x=fea_scal(X)

J_histories,theta=gradient(x,y)
plt.plot(J_histories)
plt.show()


pred_y=x.dot(theta)
print(y.shape)
error=np.hstack((y,pred_y,y-pred_y))
print(error)

plt.plot(y,pred_y,'rx')
plt.show()




