import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def Householder(A):
    """
    Genera una matriz de transformación de Householder H.
    
    Parámetros:
    A : ndarray
        Submatriz o vector a triangularizar.
        
    Retorna:
    H : ndarray
        Matriz de transformación de Householder.
    """
    n = len(A)  # Número de filas de la matriz A
    z = A[:, 0]  # Primera columna de la matriz A

    # Cálculo de alfa para dirigir z hacia una dirección específica
    alfa = -np.sign(z[0]) * np.linalg.norm(z)  # Magnitud con el signo adecuado
    
    # Creación del vector e1 (vector canónico unitario en la primera dirección)
    e1 = np.zeros((n, 1))
    e1[0] = 1  # Primer componente es 1

    # Calcular el vector v que apunta hacia alfa * e1
    v = alfa * e1

    # Calcular el vector u que define la reflexión
    u = z.reshape((n, 1)) - v  # Diferencia entre z y alfa * e1
    u = u / np.linalg.norm(u)  # Normalización de u

    # Construcción de la matriz de Householder H
    H = np.eye(n) - 2 * u.dot(np.transpose(u))  # H = I - 2 * u * u^T
    
    return H  # Matriz de transformación de Householder

def descQR(A):
    """
    Realiza la descomposición QR de una matriz cuadrada A utilizando transformaciones de Householder.
    
    Parámetros:
    A : ndarray
        Matriz cuadrada a descomponer.
        
    Retorna:
    Q : ndarray
        Matriz ortogonal de la descomposición QR.
    R : ndarray
        Matriz triangular superior de la descomposición QR.
    """
    n = A.shape[0]  # Dimensión de la matriz A
    R = A.copy()  # Inicializamos R como una copia de A
    Q = np.eye(n)  # Inicializamos Q como la matriz identidad

    for k in range(n - 1):
        # Seleccionamos la submatriz a triangularizar
        Ak = R[k:, k:]  # Submatriz desde la fila y columna k

        # Calculamos la matriz de Householder para la submatriz Ak
        Hk = Householder(Ak)

        # Expandimos Hk al tamaño original de A
        Qk = np.eye(n)
        Qk[k:, k:] = Hk  # Inserción de Hk en la posición correspondiente

        # Actualizamos Q y R aplicando Qk
        Q = Qk.dot(Q)  # Acumulamos las transformaciones de Householder
        R = Qk.dot(R)  # Actualizamos la matriz R hacia la forma triangular superior

    return np.transpose(Q), R  # Retornamos Q transpuesta (porque Q se acumula como Q^T) y R

def AlgQR(A, tol, maxiter):
    """
    Encuentra los autovalores de una matriz cuadrada A utilizando el algoritmo QR iterativo.
    
    Parámetros:
    A : ndarray
        Matriz cuadrada de entrada.
    tol : float
        Tolerancia para el criterio de convergencia (valores pequeños fuera de la diagonal).
    maxiter : int
        Número máximo de iteraciones.
        
    Retorna:
    autovalores : ndarray
        Autovalores de la matriz A.
    A_final : ndarray
        Matriz A transformada a una forma casi diagonal.
    iter : int
        Número de iteraciones realizadas.
    """
    iter = 0  # Inicializamos el contador de iteraciones

    # Continuamos hasta que la parte triangular inferior sea suficientemente pequeña o se alcance el máximo de iteraciones
    while (np.linalg.norm(np.tril(A, -1)) > tol) and (iter < maxiter):
        Q, R = descQR(A)  # Descomponemos A en Q y R usando descQR
        A = R.dot(Q)  # Calculamos la matriz A para la siguiente iteración
        iter += 1  # Incrementamos el contador de iteraciones

    # Retornamos los autovalores (diagonal de A final), la matriz casi diagonal y las iteraciones realizadas
    return np.diag(A), A, iter
