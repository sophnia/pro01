import numpy as np
import matplotlib.pyplot as plt

def djhs(theta):
    J=1.0/(2*m)*(x.dot(theta)-y).T.dot(x.dot(theta)-y)
    return J

def tdxj(x,y,theta=[[1],[1]],alpha=0.01,iterms=1500):
    J_histories=np.ones(iterms)
    for i in range(iterms):
        J_histories[i]=djhs(theta)
        qd=1.0/m*(x.T.dot(x.dot(theta)-y))
        theta=theta-alpha*qd
    return theta,J_histories

file=np.loadtxt("ex1data1.txt",delimiter=',')
m=len(file)
x=np.c_[np.ones(m),file[:,0]]
y=np.c_[file[:,1]]
plt.plot(x[:,1],y,'rx')
plt.show()

theta,J_histories=tdxj(x,y)
print(theta)

plt.plot(J_histories)
plt.show()

yy=theta[0]+theta[1]*x
plt.plot(x,yy)
plt.show()
print(yy)
