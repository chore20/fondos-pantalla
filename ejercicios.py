import raaices_ecuas as nw 
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import random as ra
import raicesOptimizadas as rop

#Funciones normales 
def f1(x):
    return x*np.e**x-1

def f2(x):
    return np.e**x-x-2

def f3(x):
    return 3*x-np.cos(x)

def f4(x):
    return x*np.log(x)-1

def f5(x):
    return x+np.sqrt(x)-x**2
#funciones derivadas
def f1_prime(x):
    return np.e**x + x * np.e**x

def f2_prime(x):
    return np.e**x - 1

def f3_prime(x):
    return 3 + np.sin(x)

def f4_prime(x):
    return np.log(x) + 1

def f5_prime(x):
    return 1 - (1 / (2 * np.sqrt(x))) - 2 * x

def grafica(c):
    x=np.linspace(c,5,200,-200)
    y=x+np.sqrt(x)-1-x**2
    plt.plot(x,y,x,np.zeros(len(x)),'r')
    plt.grid(True)  
    plt.show()

#print(nw.biseccion2(f5, -0.5, 2, 10**(-8), 100))
#grafica(-5)
print(nw.Newton(f5, f5_prime, 1, 10**-8, 100))
#print(nw.secante(f5, -0.5, 1, 10**-8, 100))

def matriz1(x):
    x1=np.zeros((2,3))
    x1[0]=x[0]**3-3*x[0]*x[1]**2-1
    x1[1]=3*x[0]**2*x[1]-x[1]**3
    return x1

def matriz1_d(x):
   x1=np.zeros((2, 2))
   x1[0]=x[0]**2-9*x[1]**2
   x1[1]=6*x[0]-3*x[1]**2
   return x1

#print(matriz1([1,1]))

# nw.newton_e(matriz1, matriz1_d, np.array([[-1],[1]]), 10**-6, 100)
def f7(x):
    return x**3-2*x**2-x+2
def df7(x):
    return 3*x**2- 4*x-1

#print(nw.secante(f7, -0.5, 1, 10**-8, 100))
#print(nw.newton_e(f7, df7, 1, 10**-8, 100))
#rop.Newton(f7, df7, -5, 10**-8, 100)
rop.secante(f7, -2, 3, 10**-8, 100)