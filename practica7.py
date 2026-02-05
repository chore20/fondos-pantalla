import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Math, Latex, Markdown
sp.init_printing()
import diferencialesOTP as diff

def f1(u,t):
    g = np.zeros((1,1))
    g = -(1+t)*u/t + np.exp(-t)*np.sin(2*t)
    return g

def f_ex(T1):
    z = np.exp(-T1)*(np.exp(1)+np.cos(2)/2)/T1-np.exp(-T1)*np.cos(2*T1)
    return z


a = 1
b = 10
c = np.array([1])
d = 40
tex = np.linspace(a,b,20)
uex = f_ex(tex)

#t,u = diff.Euler(f1,1,10,1,40)
#t,u = diff.RungeKutta01O2(f1,a,b,c,d)
#t,u = diff.AdamBashford2(f1, diff.RungeKutta01O2,a,b,1,d)
t,u = diff.AdamBashford4(f1, diff.RUNGEKUTTA4, a, b, 1, d)

plt.plot(t,u[:],'r.--')
plt.plot(tex,uex[:],'g-')
plt.show()