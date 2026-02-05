import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint # para modelos mas grandes o complejos
from IPython.display import display, Math, Latex, Markdown
sp.init_printing()

'''
IMPORTANTE!!!
para poner una funcion hay que definirla literalmente como una funcion (def). Tiene que ser de columnas varias
dependiendo cuantas funciones diferenciales tengamos:
(x = np.zeros((filas, columnas)))
Dentro de esta funsion deben de estar las ecuaciones dentro de la misma variable:

def funsion(u,t):
    x = np.zeros((filas, columnas))
    x[0,0] = ecuacion1
    x[1,0] = ecuacion2
    return x

Esto depende si son varias ecuaciones, si no es asi no es necesario: 

def funsion(u,t):
    x = ecuacion1
    return x
'''


def Euler(f, a, b, ua, n):
    """
    Implementa el método de Euler para resolver ecuaciones diferenciales ordinarias.
    
    Parámetros:
    f (función): La función que define la EDO dy/dt = f(y,t)
    a (float): Tiempo inicial
    b (float): Tiempo final
    ua (float): Valor inicial y(a)
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U los valores aproximados de la solución
    """
    T = np.linspace(a, b, n+1)
    U = np.zeros(len(T))
    U[0] = ua
    h = T[1]-T[0]
    for i in range(n):
        U[i+1] = U[i] + h*f(U[i], T[i])
    return T, U

def TaylorExp(ua, a, b, n):
    """
    Implementa el método de Taylor para la ecuación exponencial.
    
    Parámetros:
    ua (float): Valor inicial
    a (float): Tiempo inicial
    b (float): Tiempo final
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    U = np.zeros(len(T))
    U[0] = ua
    h = T[1]-T[0]
    for i in range(n):
        U[i+1] = U[i] + h*U[i] + h**2/2*U[i]
    return T, U

def RungeKutta01O2(f, a, b, ua, n):
    """
    Implementa el método de Runge-Kutta de orden 2 (punto medio).
    
    Parámetros:
    f (función): La función que define la EDO dy/dt = f(y,t)
    a (float): Tiempo inicial
    b (float): Tiempo final
    ua (float): Valor inicial y(a)
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    U = np.zeros(len(T))
    U[0] = ua
    h = T[1]-T[0]
    for i in range(n):
        K1 = h*f(U[i], T[i])
        K2 = h*f(U[i]+K1/2, T[i]+h/2)
        U[i+1] = U[i] + K2
    return T, U

def EULER(f, a, b, Ua, n):
    """
    Implementa el método de Euler para sistemas de EDOs.
    
    Parámetros:
    f (función): La función que define el sistema de EDOs
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (array): Vector de valores iniciales
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U es una matriz con las soluciones
    """
    T = np.linspace(a, b, n+1)
    m = len(Ua)
    U = np.zeros((m, n+1))
    for k in range(m):
        U[k,0] = Ua[k]
    h = T[1]-T[0]
    for i in range(n):
        for k in range(m):
            U[k,i+1] = U[k,i] + h*f(U[:,i], T[i])[k]
    return T, U

def RUNGEKUTTA01O2(f, a, b, Ua, n):
    """
    Implementa el método de Runge-Kutta de orden 2 para sistemas de EDOs.
    
    Parámetros:
    f (función): La función que define el sistema de EDOs
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (array): Vector de valores iniciales
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U es una matriz con las soluciones
    """
    T = np.linspace(a, b, n+1)
    m = len(Ua)
    U = np.zeros((m, n+1))
    for k in range(m):
        U[k,0] = Ua[k]
    h = T[1]-T[0]
    for i in range(n):
        K1 = h*f(U[:,i], T[i])
        K2 = h*f(np.array(U[:,i]).reshape((m,1))+np.array(K1)/2, T[i]+h/2)
        for k in range(m):
            U[k,i+1] = U[k,i] + K2[k]
    return T, U

def RUNGEKUTTA4(f, a, b, Ua, n):
    """
    Implementa el método de Runge-Kutta de orden 4 para sistemas de EDOs.
    
    Parámetros:
    f (función): La función que define el sistema de EDOs
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (array): Vector de valores iniciales
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, U) donde T son los puntos de tiempo y U es una matriz con las soluciones
    """
    T = np.linspace(a, b, n+1)
    m = len(Ua)
    U = np.zeros((m, n+1))
    for k in range(m):
        U[k,0] = Ua[k]
    h = T[1]-T[0]
    for i in range(n):
        K1 = h*f(U[:,i], T[i])
        K2 = h*f(np.array(U[:,i]).reshape((m,1))+np.array(K1)/2, T[i]+h/2)
        K3 = h*f(np.array(U[:,i]).reshape((m,1))+np.array(K2)/2, T[i]+h/2)
        K4 = h*f(np.array(U[:,i]).reshape((m,1))+np.array(K3), T[i]+h)
        for k in range(m):
            U[k,i+1] = U[k,i] + 1/6*(K1[k]+2*K2[k]+2*K3[k]+K4[k])
    return T, U

def Multipaso2(f, fRK, a, b, Ua, n):
    """
    Implementa un método multipaso de orden 2.
    
    Parámetros:
    f (función): La función que define la EDO
    fRK (función): Método de Runge-Kutta para inicialización
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (float): Valor inicial
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, X) donde T son los puntos de tiempo y X los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    h = (b-a)/n
    X = np.zeros(n+1)
    X[0] = Ua
    _, aux = fRK(f, T[0], T[1], Ua, 1)
    X[1] = aux[-1]
    for i in range(2, n+1):
        X[i] = X[i-2] + 2*h*f(X[i-1], T[i-1])
    return T, X

def AdamBashford2(f, fRK, a, b, ua, n):
    """
    Implementa el método de Adams-Bashforth de orden 2.
    
    Parámetros:
    f (función): La función que define la EDO
    fRK (función): Método de Runge-Kutta para inicialización
    a (float): Tiempo inicial
    b (float): Tiempo final
    ua (float): Valor inicial
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, X) donde T son los puntos de tiempo y X los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    h = (b-a)/n
    X = np.zeros(n+1)
    X[0] = ua
    _, aux = fRK(f, T[0], T[1], ua, 1)
    X[1] = aux[-1]
    for i in range(2, n+1):
        X[i] = X[i-1] + 3/2*h*f(X[i-1], T[i-1]) - 1/2*h*f(X[i-2], T[i-2])
    return T, X

def AdamBashford3(f, fRK, a, b, Ua, n):
    """
    Implementa el método de Adams-Bashforth de orden 3.
    
    Parámetros:
    f (función): La función que define la EDO
    fRK (función): Método de Runge-Kutta para inicialización
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (float): Valor inicial
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, X) donde T son los puntos de tiempo y X los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    h = (b-a)/n
    X = np.zeros(n+1)
    X[0] = Ua
    _, aux = fRK(f, T[0], T[2], Ua, 2)
    X[1] = aux[1]
    X[2] = aux[2]
    for i in range(3, n+1):
        X[i] = X[i-1] + 23/12*h*f(X[i-1], T[i-1]) - 16/12*h*f(X[i-2], T[i-2]) + 5/12*h*f(X[i-3], T[i-3])
    return T, X

def AdamBashford4(f, fRK, a, b, Ua, n):
    """
    Implementa el método de Adams-Bashforth de orden 4.
    
    Parámetros:
    f (función): La función que define la EDO
    fRK (función): Método de Runge-Kutta para inicialización
    a (float): Tiempo inicial
    b (float): Tiempo final
    Ua (float): Valor inicial
    n (int): Número de pasos
    
    Retorna:
    tuple: (T, X) donde T son los puntos de tiempo y X los valores aproximados
    """
    T = np.linspace(a, b, n+1)
    h = (b-a)/n
    X = np.zeros(n+1)
    X[0] = Ua
    _, aux = fRK(f, T[0], T[3], Ua, 3)
    X[1] = aux[-3]
    X[2] = aux[-2]
    X[3] = aux[-1]
    for i in range(4, n+1):
        X[i] = X[i-1] + 55/24*h*f(X[i-1], T[i-1]) - 59/24*h*f(X[i-2], T[i-2]) + 37/24*h*f(X[i-3], T[i-3]) - 9/24*h*f(X[i-4], T[i-4])
    return T, X

