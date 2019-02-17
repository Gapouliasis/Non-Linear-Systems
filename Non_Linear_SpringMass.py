import numpy as np
import matplotlib.pyplot as plt

#Function derivatives definition dx/dt=f and dv/dt=g
f=lambda v : v
g=lambda x,v,t : -c*v-k*x-a*x**3+F*np.cos(w*t)
#Parameter setting
Dt=0.1
k=0
c=0.1
a=1
F=10
w=1
itend=1000
x1=np.zeros((itend,1))
v1=np.zeros((itend,1))
x1[0]=0.03
v1[0]=0
def RK(x,v,f,g,Dt,k,c,a,itend):
    #Fourth order RK solver
    for it in range(1,itend):
        k1f=f(v[it-1])
        k1g=g(x[it-1],v[it-1],it*Dt)
        xhalf=x[it-1]+k1f*Dt/2
        vhalf=v[it-1]+k1g*Dt/2
       
        k2f=f(vhalf)
        k2g=g(xhalf,vhalf,it*Dt)
        xhalf=x[it-1]+k2f*Dt/2
        vhalf=v[it-1]+k2g*Dt/2
        
        k3f=f(vhalf)
        k3g=g(xhalf,vhalf,it*Dt)
        xwhole=x[it-1]+k3f*Dt
        vwhole=v[it-1]+k3g*Dt
        
        k4f=f(vwhole)
        k4g=g(xwhole,vwhole,it*Dt)
        
        x[it]=x[it-1]+1/6*(k1f+2*k2f+2*k3f+k4f)*Dt
        v[it]=v[it-1]+1/6*(k1g+2*k2g+2*k3g+k4g)*Dt
    return [x,v]

[x1,v1]=RK(x1,v1,f,g,Dt,k,c,a,itend)

x2=np.zeros((itend,1))
v2=np.zeros((itend,1))
x2[0]=0.03001
v2[0]=0

[x2,v2]=RK(x2,v2,f,g,Dt,k,c,a,itend)
plt.figure(1)
plt.plot(x1,v1,x2,v2)
plt.figure(2)
plt.plot(np.arange(1,itend+1),x1,np.arange(1,itend+1),x2)
plt.show()
