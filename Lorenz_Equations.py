import numpy as np
import matplotlib.pyplot as plt

#Function derivatives definition dx/dt=f, dy/dt=g and dz/dt=p
f=lambda x,y : -a*x+a*y 
g=lambda x,y,z : r*x-y-x*z
p=lambda x,y,z : -b*z+x*y
#Parameter setting
Dt=0.03125
a=10
b=8/3
r=28
itend=1000
x1=np.zeros((itend,1))
y1=np.zeros((itend,1))
z1=np.zeros((itend,1))
x1[0]=5
y1[0]=5
z1[0]=5
def RK(x,y,z,f,g,p,Dt,a,b,r,itend):
    #Fourth order RK solver
    for it in range(1,itend):
        k1f=f(x[it-1],y[it-1])
        k1g=g(x[it-1],y[it-1],z[it-1])
        k1p=p(x[it-1],y[it-1],z[it-1])
        xhalf=x[it-1]+k1f*Dt/2
        yhalf=y[it-1]+k1g*Dt/2
        zhalf=z[it-1]+k1p*Dt/2
       
        k2f=f(xhalf,yhalf)
        k2g=g(xhalf,yhalf,zhalf)
        k2p=p(xhalf,yhalf,zhalf)
        xhalf=x[it-1]+k2f*Dt/2
        yhalf=y[it-1]+k2g*Dt/2
        zhalf=z[it-1]+k2p*Dt/2
        
        k3f=f(xhalf,yhalf)
        k3g=g(xhalf,yhalf,zhalf)
        k3p=p(xhalf,yhalf,zhalf)
        xwhole=x[it-1]+k3f*Dt
        ywhole=y[it-1]+k3g*Dt
        zwhole=z[it-1]+k3p*Dt
        
        k4f=f(xwhole,ywhole)
        k4g=g(xwhole,ywhole,zwhole)
        k4p=p(xwhole,ywhole,zwhole)
        
        x[it]=x[it-1]+1/6*(k1f+2*k2f+2*k3f+k4f)*Dt
        y[it]=y[it-1]+1/6*(k1g+2*k2g+2*k3g+k4g)*Dt
        z[it]=z[it-1]+1/6*(k1p+2*k2p+2*k3p+k4p)*Dt
    return [x,y,z]

[x1,y1,z1]=RK(x1,y1,z1,f,g,p,Dt,a,b,r,itend)

x2=np.zeros((itend,1))
y2=np.zeros((itend,1))
z2=np.zeros((itend,1))
x2[0]=5.001
y2[0]=5
z2[0]=5

[x2,y2,z2]=RK(x2,y2,z2,f,g,p,Dt,a,b,r,itend)
plt.figure(1)
plt.plot(x1,y1,x2,y2)

plt.figure(2)
plt.plot(np.arange(1,itend+1),x1,np.arange(1,itend+1),x2)
plt.show()