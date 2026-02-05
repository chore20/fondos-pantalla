import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import random as ra


def biseccion1(f, a, b, tol):
    """
    Método de Bisección para encontrar una raíz de la ecuación f(x) = 0.
    
    Parámetros:
    - f: Función para la cual se busca la raíz.
    - a: Extremo izquierdo del intervalo.
    - b: Extremo derecho del intervalo.
    - tol: Tolerancia para determinar la precisión de la raíz encontrada.
    
    Retorna:
    - c: Aproximación de la raíz.
    - f(c): Valor de la función en la raíz encontrada.
    - iter: Número de iteraciones realizadas.
    """
    iter = 0

    # Verificar si los extremos ya son una raíz
    if f(a) == 0:
        return a, f(a), iter
    elif f(b) == 0:
        return b, f(b), iter

    # Verificar que f(a) y f(b) tengan signos opuestos
    if np.sign(f(a)) * np.sign(f(b)) < 0:
        c = (a + b) / 2
        while np.abs(f(c)) > tol:
            c = (a + b) / 2
            iter += 1
            print("%d %.8f %.8f %.8f %.8g" % (iter, a, b, c, f(c)))
            if np.sign(f(a)) * np.sign(f(c)) < 0:
                b = c
            else:
                a = c
        return c, f(c), iter
    else:
        print('Error: La función no tiene signos opuestos en a y b.')


def biseccion2(f, a, b, tol=1e-6, max_iter=100):
    """
    Método de Bisección para encontrar una raíz de la ecuación f(x) = 0.
    
    Parámetros:
    - f: Función para la cual se busca la raíz.
    - a: Extremo izquierdo del intervalo.
    - b: Extremo derecho del intervalo.
    - tol: Tolerancia para la precisión de la raíz (por defecto 1e-6).
    - max_iter: Número máximo de iteraciones (por defecto 100).
    
    Retorna:
    - raiz_aproximada: Aproximación de la raíz encontrada.
    - iteraciones: Número de iteraciones realizadas.
    """
    # Verificar que la función tenga signos opuestos en los extremos
    if f(a) * f(b) > 0:
        raise ValueError("La función debe tener signos opuestos en los extremos a y b.")

    iteraciones = 0

    while (b - a) / 2.0 > tol and iteraciones < max_iter:
        c = (a + b) / 2.0
        error = (b-a)/2.0
        if f(c) == 0:  # Verificar si c es raíz exacta
            return c, iteraciones

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        iteraciones += 1

    raiz_aproximada = (a + b) / 2.0
    return raiz_aproximada, iteraciones, error


def EstimacionBisec(a, b, tol):
    """
    Calcula una estimación del número máximo de iteraciones necesarias para 
    el método de Bisección.

    Parámetros:
    - a: Extremo izquierdo del intervalo.
    - b: Extremo derecho del intervalo.
    - tol: Tolerancia requerida.

    Retorna:
    - Número estimado de iteraciones.
    """
    return np.ceil(np.log2(b - a) - np.log2(tol))


def Newton(f, df, x0, tol, maxiter):
    """
    Método de Newton para encontrar una raíz de la ecuación f(x) = 0.

    Parámetros:
    - f: Función para la cual se busca la raíz.
    - df: Derivada de la función f.
    - x0: Valor inicial de la iteración.
    - tol: Tolerancia para determinar la precisión de la raíz.
    - maxiter: Número máximo de iteraciones.

    Retorna:
    - x: Aproximación de la raíz.
    - f(x): Valor de la función en la raíz encontrada.
    - iter: Número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    error = np.inf  

    while (error > tol) and (iter < maxiter):
        iter += 1
        x_new = x - f(x) / df(x)  
        error = np.abs(x_new - x) 
        x = x_new
        print('%d %.8f %.16f %.8g' % (iter, x, f(x), error))  

    return x, f(x), iter, error

def newton_modificado(f, df, df2, x, tol, maxiter):
    """
    Método de Newton modificado que utiliza derivadas de primer y segundo orden.

    Parámetros:
    - f: Función para la cual se busca la raíz.
    - df: Derivada de la función f.
    - df2: Segunda derivada de la función f.
    - x: Valor inicial de la iteración.
    - tol: Tolerancia requerida.
    - maxiter: Número máximo de iteraciones.

    Retorna:
    - x: Aproximación de la raíz.
    - f(x): Valor de la función en la raíz encontrada.
    - iter: Número de iteraciones realizadas.
    """
    iter = 0
    while (np.abs(f(x)) > tol) and (iter < maxiter):
        x = x - (f(x) * df(x)) / (df(x)**2 - f(x) * df2(x))  # Fórmula modificada
        iter += 1
    return x, f(x), iter


def secante(f, x0, x1, tol, maxiter):
    """
    Método de la Secante para encontrar una raíz de la ecuación f(x) = 0.

    Parámetros:
    - f: Función para la cual se busca la raíz.
    - x0: Primer valor inicial.
    - x1: Segundo valor inicial.
    - tol: Tolerancia requerida.
    - maxiter: Número máximo de iteraciones.

    Retorna:
    - x1: Aproximación de la raíz.
    - f(x1): Valor de la función en la raíz encontrada.
    - iter: Número de iteraciones realizadas.
    """
    iter = 0
    while (np.abs(f(x1)) > tol) and (iter < maxiter):
        iter += 1
        aux = x1
        x1 = x1 - (x1 - x0) * f(x1) / (f(x1) - f(x0))  # Fórmula de la Secante
        x0 = aux
        print('%d %.8f %.8f %.g' % (iter, x0, x1, f(x1)))
        error = np.abs(x1 - x0)
    return x1, f(x1), iter, error


def newton_e(f, df, x0, tol, maxiter):
    """
    Método de Newton extendido para sistemas de ecuaciones no lineales.

    Parámetros:
    - f: Función vectorial para la cual se busca la raíz.
    - df: Jacobiano de la función f.
    - x0: Valor inicial de la iteración.
    - tol: Tolerancia requerida.
    - maxiter: Número máximo de iteraciones.

    Retorna:
    - x: Aproximación del vector solución.
    - np.linalg.norm(f(x)): Norma del error final.
    - iter: Número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    while (np.linalg.norm(f(x)) > tol) and (iter < maxiter):
        iter += 1
        x = x - np.dot(np.linalg.inv(df(x)), f(x))  # Actualización de Newton
        print(x)
    return x, np.linalg.norm(f(x)), iter
