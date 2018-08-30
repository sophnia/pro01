import numpy as np
import matplotlib.pyplot as plt


def djhs(theta):
    J=1.0/(2*m)*(x.dot(theta)-y).T.dot(x.dot(theta)-y)
    return J

def tdxj(x,y,theta=[[1],[1]],alpha=0.01,iterms=1500):
    J_histories=np.ones(iterms)
    for i in range (iterms):#(有括号)
        J_histories[i]=djhs(theta)
        qd=1.0/m*x.T.dot(x.dot(theta)-y)
        theta=theta-alpha*qd
    return theta,J_histories


#读取文件
file=np.loadtxt("ex1data1.txt",delimiter=',')
m=len(file)

#读取x,y数据 np_c[两个矩阵的链接]
x=np.c_[np.ones(m),file[:,0]]
y=np.c_[file[:,1]]

plt.plot(x[:,1],y,'bx')
plt.show()

theta,J_histories=tdxj(x,y)
print(theta)
plt.plot(theta)
plt.show()

xx=np.arange(50)
yy=theta[0]+theta[1]*xx
plt.plot(xx,yy,'r.')
plt.show()

