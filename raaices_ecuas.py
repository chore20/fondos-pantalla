import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import random as ra


def biseccion1(f,a,b,tol):
    iter=0
    if f(a)==0:
        return a,f(a),iter
    elif f(b)==0:
        return b,f(b),iter
    else:
        if np.sign(f(a))*np.sign(f(b))<0:
            c=(a+b)/2
            while np.abs(f(c))>tol:
                c=(a+b)/2
                iter=iter+1
                print("%d %.8f %.8f %.8f %.8g" % (iter,a,b,c,f(c)))
                if np.sign(f(a))*np.sign(f(c))<0:
                    b=c
                else:
                    a=c
            return c,f(c),iter
        else:
            print('Error: El producto de la evaluación de los puntos iniciales no es inferior a 0.')



def biseccion2(f, a, b, tol=1e-6, max_iter=100):
    """
    Método de Bisección para encontrar una raíz de la ecuación f(x) = 0.
    :param f: Función para la cual se busca la raíz.
    :param a: Extremo izquierdo del intervalo.
    :param b: Extremo derecho del intervalo.
    :param tol: Tolerancia para la precisión de la raíz encontrada.
    :param max_iter: Número máximo de iteraciones.
    :return: Aproximación de la raíz y el número de iteraciones realizadas.
    """

    # Verificar que la función tenga signos opuestos en los extremos
    if f(a) * f(b) > 0:
        raise ValueError("La función debe tener signos opuestos en los extremos a y b.")

    iteraciones = 0

    while (b - a) / 2.0 > tol and iteraciones < max_iter:
        # Punto medio del intervalo
        c = (a + b) / 2.0

        # Verificar si c es una raíz o si hemos alcanzado la tolerancia deseada
        if f(c) == 0:
            return c, iteraciones

        # Actualizar el intervalo según el signo de f(c)
        if f(a) * f(c) < 0:
            b = c  # La raíz está en el intervalo [a, c]
        else:
            a = c  # La raíz está en el intervalo [c, b]

        iteraciones += 1

    # La raíz aproximada es el punto medio del último intervalo
    raiz_aproximada = (a + b) / 2.0
    return raiz_aproximada, iteraciones

def EstimacionBisec(a,b,tol):
    return np.ceil(np.log2(b-a)-np.log2(tol))

# metodos de newton y newton modificado

def Newton(f,df,x0,tol,maxiter):
    x=x0
    iter=0
    while (np.abs(f(x))>tol)&(iter<maxiter):
        iter=iter+1
        x=x-f(x)/df(x)
        print('%d %.8f %.16f' % (iter,x,f(x)))
    return x,f(x),iter

def newton_modificado(f,df,df2,x,TOL,maxit):
  iter = 0
  while (np.abs(f(x)) > TOL) & (iter < maxit):
    x = x - (f(x)*df(x))/(df(x)**2-f(x)*df2(x))
    iter = iter + 1
  return x, f(x), iter

def secante(f,x0,x1,tol,maxiter):
    iter=0
    while (np.abs(f(x1))>tol)&(iter<maxiter):
        iter=iter+1
        aux=x1
        x1=x1-(x1-x0)*f(x1)/(f(x1)-f(x0))
        x0=aux
        print('%d %.8f %.8f %.g' % (iter,x0,x1,f(x1)))
    return x1,f(x1),iter

def newton_e(f,df,x0,tol,maxiter):
    x=x0
    iter=0
    while (np.linalg.norm(f(x))>tol)&(iter<maxiter):
        iter=iter+1
        x=x-np.dot(np.linalg.inv(df(x)),f(x))
        print(x)
    return x,np.linalg.norm(f(x)),iter

