import numpy as np
import sympy as sp
from IPython.display import display
import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
import pandas as pd
from IPython.display import display

sp.init_printing()

def factorizacion_lu_sp(A):
    """Realiza la factorización LU de una matriz A sin permutaciones.

    Parámetros:
    A: matriz cuadrada que se desea factorizar.

    Retorno:
    L: matriz triangular inferior.
    U: matriz triangular superior.
    """
    n = A.shape[0]  # número de filas
    L = np.eye(n)  # Matriz L inicializada como la identidad
    U = A.copy()   # U se inicializa como copia de A

    for k in range(n):
        # Eliminación hacia adelante en las filas de U
        for i in range(k + 1, n):
            mik = U[i, k] / U[k, k]  # multiplicador
            U[i, k:] = U[i, k:] - mik * U[k, k:]  # Actualizar fila i de U
            L[i, k] = mik  # Actualizar matriz L
    return L, U

def factorizacion_lu_parcial(A):
    """Realiza la factorización LU de una matriz A con permutaciones.

    Parámetros:
    A: matriz cuadrada que se desea factorizar.

    Retorno:
    P: matriz de permutación.
    L: matriz triangular inferior.
    U: matriz triangular superior.
    """
    n = A.shape[0]  # dimensión de la matriz
    P = np.eye(n)  # Matriz de permutación (inicialmente identidad)
    L = np.eye(n)  # Matriz triangular inferior (inicialmente identidad)
    U = A.copy()   # U empieza como una copia de A

    for k in range(n):
        # Encontrar el índice del pivote: máximo absoluto en la columna k
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k  # Índice global del pivote
        if k != pivot_row:
            # Intercambiar filas en U y P
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # Intercambiar filas correspondientes en L
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Eliminación de Gauss
        for i in range(k + 1, n):
            mik = U[i, k] / U[k, k]  # El multiplicador de eliminación
            U[i, k:] = U[i, k:] - mik * U[k, k:]  # Actualizar fila i de U
            L[i, k] = mik  # Guardar el multiplicador en L

    return P, L, U

def sustitución_progresiva(L, b):
    """Resuelve el sistema Ly = b usando sustitución hacia adelante.

    Parámetros:
    L: matriz triangular inferior.
    b: vector de términos independientes.

    Retorno:
    y: vector solución.
    """
    n = len(b)
    y = np.zeros_like(b)  # Vector de soluciones, inicializado en ceros
    for i in range(n):
        suma = np.dot(L[i, :i], y[:i])  # Suma de términos previos
        y[i] = (b[i] - suma) / L[i, i]  # Resolver para y[i]
    return y

def sustitucion_regresiva(U, y):
    """Resuelve el sistema Ux = y usando sustitución hacia atrás.

    Parámetros:
    U: matriz triangular superior.
    y: vector solución del sistema anterior.

    Retorno:
    x: vector solución.
    """
    n = len(y)
    x = np.zeros_like(y)  # Vector de soluciones, inicializado en ceros
    for i in range(n - 1, -1, -1):
        suma = np.dot(U[i, i + 1:], x[i + 1:])  # Suma de términos resueltos
        x[i] = (y[i] - suma) / U[i, i]  # Resolver para x[i]
    return x

def resuelve_sistema_lineal(A, b):
    """Resuelve el sistema Ax = b utilizando la factorización LU con permutación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.

    Retorno:
    x: vector solución.
    """
    P, L, U = factorizacion_lu_parcial(A)
    y = sustitución_progresiva(L, P.dot(b))  # Ly = Pb
    x = sustitucion_regresiva(U, y)  # Ux
    return x

def resuelve_sistema_lineal_sin_permutacion(A, b):
    """Resuelve el sistema Ax = b utilizando la factorización LU sin permutación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.

    Retorno:
    x: vector solución.
    """
    L, U = factorizacion_lu_sp(A)
    y = sustitución_progresiva(L, b)  # Ly = b
    x = sustitucion_regresiva(U, y)  # Ux
    return x

def Determinante(A):
    """Calcula el determinante de la matriz A utilizando la factorización LU.

    Parámetros:
    A: matriz cuadrada.

    Retorno:
    Determinante de A.
    """
    _, U = factorizacion_lu_sp(A)
    return np.prod(np.diag(U))

def inversa_por_gauss_jordan(A):
    """Calcula la inversa de la matriz A utilizando el método de Gauss-Jordan.

    Parámetros:
    A: matriz cuadrada.

    Retorno:
    A_inv: inversa de A.
    """
    n = A.shape[0]
    A_augmented = np.hstack([A, np.eye(n)])  # Matriz aumentada [A | I]

    for i in range(n):
        max_row = np.argmax(np.abs(A_augmented[i:n, i])) + i
        if A_augmented[max_row, i] == 0:
            raise ValueError("La matriz no es invertible.")

        A_augmented[[i, max_row]] = A_augmented[[max_row, i]]  # Intercambiar filas
        A_augmented[i] = A_augmented[i] / A_augmented[i, i]  # Hacer 1 el pivote

        for j in range(n):
            if j != i:
                A_augmented[j] = A_augmented[j] - A_augmented[j, i] * A_augmented[i]  # Hacer ceros en la columna

    A_inv = A_augmented[:, n:]  # Extraer la parte de la matriz aumentada que corresponde a la inversa
    return A_inv

def metodo_jacobi(A, b, tol=1e-8, max_iter=1000):
    """Resuelve el sistema Ax = b utilizando el método de Jacobi.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    tol: tolerancia para la convergencia (por defecto 1e-8).
    max_iter: número máximo de iteraciones (por defecto 1000).

    Retorno:
    x_new: vector solución.
    """
    n = len(b)
    x = np.zeros_like(b)  # Vector de soluciones, inicializado en ceros
    x_new = np.zeros_like(b)  # Vector para almacenar los valores de la siguiente iteración

    for k in range(max_iter):
        for i in range(n):
            sum_ax = np.dot(A[i, :], x)  # Suma A_ij * x_j
            sum_ax -= A[i, i] * x[i]  # Restar A_ii * x_i
            x_new[i] = (b[i] - sum_ax) / A[i, i]  # Calcular nuevo valor

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:  # Comprobar convergencia
            print(f"Convergió en la iteración {k + 1}")
            return x_new

        x = x_new.copy()  # Actualizar el valor de x

    print("Alcanzó el número máximo de iteraciones sin converger.")
    return x_new

# Métodos de relajación
def RichardsonSOR(A, b, x0, tol, maxiter, w):
    """Método de Richardson con Sobrerrelajación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    x0: vector inicial.
    tol: tolerancia para la convergencia.
    maxiter: número máximo de iteraciones.
    w: factor de relajación.

    Retorno:
    x: vector solución.
    residuo: residuo del sistema.
    iter: número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    N = np.eye(n) - w * A
    while (np.linalg.norm(np.dot(A, x) - b) > tol) & (iter < maxiter):
        x = w * b + np.dot(N, x)
        iter += 1
    return x, np.dot(A, x) - b, iter

def JacobiSOR(A, b, x0, tol, maxiter, w):
    """Método de Jacobi con Sobrerrelajación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    x0: vector inicial.
    tol: tolerancia para la convergencia.
    maxiter: número máximo de iteraciones.
    w: factor de relajación.

    Retorno:
    x: vector solución.
    residuo: residuo del sistema.
    iter: número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    M = np.diag(np.diag(A))
    N = M - w * A
    invM = np.diag(1 / np.diag(A))
    while (np.linalg.norm(np.dot(A, x) - b) > tol) & (iter < maxiter):
        x = np.dot(invM, w * b + np.dot(N, x))
        iter += 1
    return x, np.dot(A, x) - b, iter

def JacobiSORi(A, b, x0, tol, maxiter, w):
    """Método de Jacobi con Sobrerrelajación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    x0: vector inicial.
    tol: tolerancia para la convergencia.
    maxiter: número máximo de iteraciones.
    w: factor de relajación.

    Retorno:
    x: vector solución.
    residuo: residuo del sistema.
    iter: número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    M = np.diag(np.diag(A))
    N = M - w * A
    invM = np.diag(1 / np.diag(A))
    
    while (np.linalg.norm(np.dot(A, x) - b) > tol) & (iter < maxiter):
        x = np.dot(invM, w * b + np.dot(N, x))
        iter += 1

    # Calcular residuo final
    residuo = np.linalg.norm(np.dot(A, x) - b)

    # Determinar si converge
    if residuo <= tol:
        print("El método Jacobi con Sobrerrelajación converge en", iter, "iteraciones.")
    else:
        print("El método Jacobi con Sobrerrelajación no converge después de", iter, "iteraciones.")

    return x, residuo, iter


def GaussSeidelSOR(A, b, x0, tol, maxiter, w):
    """Método de Gauss-Seidel con Sobrerrelajación.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    x0: vector inicial.
    tol: tolerancia para la convergencia.
    maxiter: número máximo de iteraciones.
    w: factor de relajación.

    Retorno:
    x: vector solución.
    residuo: residuo del sistema.
    iter: número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    M = np.tril(A)
    N = M - w * A
    invM = np.linalg.inv(M)
    while (np.linalg.norm(np.dot(A, x) - b) > tol) & (iter < maxiter):
        x = np.dot(invM, w * b + np.dot(N, x))
        iter += 1
    return x, np.dot(A, x) - b, iter

def errores(A, b):
    """Evalúa la convergencia de los métodos iterativos.

    Parámetros:
    A: matriz de coeficientes.
    b: vector de términos independientes.
    """
    # Método de Jacobi
    d = np.diag(A)
    D = np.diag(d)
    N = D - A
    Tj = np.linalg.inv(D).dot(N)  # D^{-1}N
    print("Tj=", Tj)
    vj, _ = np.linalg.eig(Tj)
    print("Jacobi: max autovalor", np.max(np.abs(vj)))

    # Método de Gauss-Seidel
    M = np.tril(A)
    N = M - A
    Tgs = np.linalg.inv(M).dot(N)  # M^{-1}N
    print("Tgs=", Tgs)
    vgs, _ = np.linalg.eig(Tgs)
    print("Gauss-Seidel: max autovalor", np.max(np.abs(vgs)))

    # Método de Richardson
    n = A.shape[0]
    N = np.eye(n) - A
    Tr = N
    print('Tr=', Tr)
    vr, _ = np.linalg.eig(Tr)
    print("Richardson: max autovalor", np.max(np.abs(vr)))

import numpy as np

def calcula_error_sistema(A, b, solucion_aproximada):
    """
    Calcula el error relativo de la solución aproximada para el sistema Ax = b.

    Parámetros:
    - A: Matriz de coeficientes del sistema.
    - b: Vector de términos independientes.
    - solucion_aproximada: Solución aproximada del sistema.

    Retorno:
    - error_relativo: Error relativo de la solución aproximada.
    """
    b_aprox = np.dot(A, solucion_aproximada)  # Calcula Ax para la solución aproximada
    error_absoluto = np.linalg.norm(b - b_aprox)  # Norma del residuo
    error_relativo = error_absoluto / np.linalg.norm(b)  # Norma relativa
    return error_relativo

def compara_metodos(A, b):
    """
    Compara el error relativo entre la solución de resuelve_sistema_lineal y numpy.linalg.solve.

    Parámetros:
    - A: Matriz de coeficientes del sistema.
    - b: Vector de términos independientes.

    Retorno:
    - resultados: Diccionario con los errores relativos y soluciones.
    """
    # Solución con resuelve_sistema_lineal
    solucion_nuestra = resuelve_sistema_lineal(A, b)
    error_nuestra = calcula_error_sistema(A, b, solucion_nuestra)

    # Solución con numpy.linalg.solve (solución exacta de referencia)
    solucion_numpy = np.linalg.solve(A, b)
    error_numpy = calcula_error_sistema(A, b, solucion_numpy)

    # Comparar las dos soluciones
    diferencia_soluciones = np.linalg.norm(solucion_nuestra - solucion_numpy)

    resultados = {
        "Solución nuestra": solucion_nuestra,
        "Error relativo nuestra": error_nuestra,
        "Solución numpy": solucion_numpy,
        "Error relativo numpy": error_numpy,
        "Diferencia entre soluciones": diferencia_soluciones,
    }
    return resultados

import numpy as np

def generar_matriz_no_cuadrada(filas, columnas):
    """
    Genera una matriz aleatoria no cuadrada y, opcionalmente,
    su equivalente definida positiva al calcular A * A^T.

    Parámetros:
    - filas (int): Número de filas de la matriz.
    - columnas (int): Número de columnas de la matriz.

    Retorna:
    - matriz (ndarray): Matriz aleatoria no cuadrada de dimensiones (filas, columnas).
    - matriz_definida_positiva (ndarray): Matriz definida positiva de tamaño (filas, filas).
    """
    # Generar una matriz aleatoria no cuadrada
    A = np.random.rand(filas, columnas)
    # Calcular la matriz definida positiva A * A^T
    matriz_definida_positiva = np.dot(A, A.T)
    # Asegurar que sea estrictamente definida positiva
    matriz_definida_positiva += np.eye(filas) * 1e-6
    return A, matriz_definida_positiva


def generar_matriz_definida_positiva(tamano):
    # Generar una matriz aleatoria
    A = np.random.rand(tamano, tamano)
    # Multiplicar la matriz por su transpuesta para asegurar que sea definida positiva
    matriz_definida_positiva = np.dot(A, A.T)
    print(matriz_definida_positiva.shape)
    return matriz_definida_positiva 
