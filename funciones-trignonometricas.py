import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import random as ra
import pandas as pd

def grafico(a):
    x=np.linspace(-a,a,1000)
    f=-(np.sin(x))/x
    plt.plot(x,f,'g',0,1/2,'ro')
    plt.show()

def f1(x):
    return (np.sin(x))/x 

def repeticion(u):
    for i in range(u,16):
        print(f1(10**(-i)))

def raices(a, x):
    return(np.sqrt(x-a) - np.sqrt(x))

def repe(u):
    for i in range(u, 16):
        print(raices(1.001, 1.002)**(-i))

#repe(1)

# Valores de x cercanos a 0
x_values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])

# Expresión original
y_original = (np.exp(x_values) - 1) / x_values**2

# Expresión modificada usando la expansión de Taylor
y_taylor_approx = (1 / x_values) + 0.5

# Crear tabla de resultados
table = pd.DataFrame({
    'x': x_values,
    'y_original': y_original,
    'y_taylor_approx': y_taylor_approx
})

table
