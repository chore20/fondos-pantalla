import numpy as np
import sympy as sp
from IPython.display import display, Math, Latex, Markdown
import pprint
import scipy
import scipy.linalg  # Biblioteca de álgebra lineal de SciPy
sp.init_printing()  # Inicialización para imprimir expresiones matemáticas en formato legible

# ----------------------------------------------------------------------------------------------------------------
# FUNCIONES DE FACTORIZACIÓN LU
# ----------------------------------------------------------------------------------------------------------------

def factorizacion_lu_sp(A):
    """
    Realiza la factorización LU de una matriz A sin permutación de filas.
    
    Parámetros:
        A: np.array
            Matriz cuadrada a descomponer.
    
    Retorna:
        L: np.array
            Matriz triangular inferior con 1's en la diagonal.
        U: np.array
            Matriz triangular superior.
    """
    n = A.shape[0]  # Dimensión de la matriz
    L = np.eye(n)   # Matriz L inicializada como la identidad
    U = A.copy()    # U se inicializa como una copia de A

    for k in range(n):
        for i in range(k + 1, n):
            mik = U[i, k] / U[k, k]  # Calculamos el multiplicador
            U[i, k:] = U[i, k:] - mik * U[k, k:]  # Actualizamos la fila i de U
            L[i, k] = mik  # Almacenamos el multiplicador en L

    return L, U

def factorizacion_lu_parcial(A):
    """
    Realiza la factorización LU de una matriz A con permutación parcial de filas.
    
    Parámetros:
        A: np.array
            Matriz cuadrada a descomponer.
    
    Retorna:
        P: np.array
            Matriz de permutación.
        L: np.array
            Matriz triangular inferior con 1's en la diagonal.
        U: np.array
            Matriz triangular superior.
    """
    n = A.shape[0]
    P = np.eye(n)  # Matriz de permutación inicializada como identidad
    L = np.eye(n)  # Matriz L inicializada como identidad
    U = A.copy()   # U se inicializa como una copia de A

    for k in range(n):
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k  # Índice del pivote
        if k != pivot_row:
            # Intercambiar filas en U, P y L
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        for i in range(k + 1, n):
            mik = U[i, k] / U[k, k]  # Multiplicador de eliminación
            U[i, k:] = U[i, k:] - mik * U[k, k:]
            L[i, k] = mik  # Guardar multiplicador en L

    return P, L, U

# ----------------------------------------------------------------------------------------------------------------
# FUNCIONES DE SUSTITUCIÓN
# ----------------------------------------------------------------------------------------------------------------

def sustitución_progresiva(L, b):
    """
    Realiza sustitución hacia adelante para resolver Ly = b.
    
    Parámetros:
        L: np.array
            Matriz triangular inferior.
        b: np.array
            Vector constante.
    
    Retorna:
        y: np.array
            Solución del sistema Ly = b.
    """
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        suma = np.dot(L[i, :i], y[:i])  # Suma de términos previos
        y[i] = (b[i] - suma) / L[i, i]
    return y

def sustitucion_regresiva(U, y):
    """
    Realiza sustitución hacia atrás para resolver Ux = y.
    
    Parámetros:
        U: np.array
            Matriz triangular superior.
        y: np.array
            Vector constante.
    
    Retorna:
        x: np.array
            Solución del sistema Ux = y.
    """
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        suma = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - suma) / U[i, i]
    return x

# ----------------------------------------------------------------------------------------------------------------
# RESOLUCIÓN DE SISTEMAS LINEALES
# ----------------------------------------------------------------------------------------------------------------

def resuelve_sistema_lineal(A, b):
    """
    Resuelve un sistema lineal Ax = b utilizando factorización LU con permutación parcial.
    
    Parámetros:
        A: np.array
            Matriz de coeficientes.
        b: np.array
            Vector constante.
    
    Retorna:
        x: np.array
            Solución del sistema Ax = b.
    """
    P, L, U = factorizacion_lu_parcial(A)
    y = sustitución_progresiva(L, P.dot(b))  # Ly = Pb
    x = sustitucion_regresiva(U, y)         # Ux = y
    return x

def resuelve_sistema_lineal_sin_permutacion(A, b):
    """
    Resuelve un sistema lineal Ax = b utilizando factorización LU sin permutación.
    
    Parámetros:
        A: np.array
            Matriz de coeficientes.
        b: np.array
            Vector constante.
    
    Retorna:
        x: np.array
            Solución del sistema Ax = b.
    """
    L, U = factorizacion_lu_sp(A)
    y = sustitución_progresiva(L, b)  # Ly = b
    x = sustitucion_regresiva(U, y)  # Ux = y
    return x

# ----------------------------------------------------------------------------------------------------------------
# CÁLCULO DEL DETERMINANTE E INVERSA
# ----------------------------------------------------------------------------------------------------------------

def Determinante(A):
    """
    Calcula el determinante de una matriz A usando factorización LU.
    
    Parámetros:
        A: np.array
            Matriz cuadrada.
    
    Retorna:
        determinante: float
            Determinante de la matriz A.
    """
    _, U = factorizacion_lu_sp(A)
    return np.prod(np.diag(U))  # Producto de los elementos diagonales de U

def inversa_por_gauss_jordan(A):
    """
    Calcula la inversa de una matriz A utilizando el método de Gauss-Jordan.
    
    Parámetros:
        A: np.array
            Matriz cuadrada.
    
    Retorna:
        A_inv: np.array
            Inversa de la matriz A.
    """
    n = A.shape[0]
    A_augmented = np.hstack([A, np.eye(n)])  # Matriz aumentada [A | I]

    for i in range(n):
        max_row = np.argmax(np.abs(A_augmented[i:n, i])) + i
        if A_augmented[max_row, i] == 0:
            raise ValueError("La matriz no es invertible.")
        A_augmented[[i, max_row]] = A_augmented[[max_row, i]]
        A_augmented[i] /= A_augmented[i, i]
        for j in range(n):
            if j != i:
                A_augmented[j] -= A_augmented[j, i] * A_augmented[i]

    return A_augmented[:, n:]  # Parte derecha corresponde a la inversa

# ----------------------------------------------------------------------------------------------------------------
# MÉTODOS ITERATIVOS
# ----------------------------------------------------------------------------------------------------------------

def RichardsonSOR(A, b, x0, tol, maxiter, w):
    """
    Método de Richardson con sobre-relajación (SOR) para resolver Ax = b.
    """
    x = x0
    n = len(A)
    N = np.eye(n) - w * A
    iter_count = 0
    while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
        x = w * b + np.dot(N, x)
        iter_count += 1
    return x, np.dot(A, x) - b, iter_count


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

# Similar explicación para `JacobiSOR` y `GaussSeidelSOR`.
import matplotlib.pyplot as plt

def graficar_metodos(A, b, x0_1, x0_2, tol, maxiter, w):
    """
    Grafica la evolución del error de convergencia para los métodos Richardson, Jacobi y Gauss-Seidel.
    
    Parámetros:
    - A: Matriz del sistema.
    - b: Vector de términos independientes.
    - x0_1: Vector inicial de prueba 1.
    - x0_2: Vector inicial de prueba 2.
    - tol: Tolerancia del error de convergencia.
    - maxiter: Número máximo de iteraciones.
    - w: Factor de relajación (omega).
    """
    metodos = [
        ("Richardson", RichardsonSOR),
        ("Jacobi", JacobiSOR),
        ("Gauss-Seidel", GaussSeidelSOR)
    ]
    puntos_iniciales = [x0_1, x0_2]
    puntos_labels = ["x0 = -100", "x0 = 100"]
    
    plt.figure(figsize=(12, 8))
    
    for (nombre_metodo, metodo), x0, label in zip(metodos * 2, puntos_iniciales * 3, puntos_labels * 3):
        errores = []
        iteraciones = []
        x = x0
        iter_count = 0
        n = len(A)
        
        # Configuración del método específico
        if nombre_metodo == "Richardson":
            N = np.eye(n) - w * A
            while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
                x = w * b + np.dot(N, x)
                errores.append(np.linalg.norm(np.dot(A, x) - b))
                iteraciones.append(iter_count)
                iter_count += 1
        elif nombre_metodo == "Jacobi":
            M = np.diag(np.diag(A))
            N = M - w * A
            invM = np.diag(1 / np.diag(A))
            while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
                x = np.dot(invM, w * b + np.dot(N, x))
                errores.append(np.linalg.norm(np.dot(A, x) - b))
                iteraciones.append(iter_count)
                iter_count += 1
        elif nombre_metodo == "Gauss-Seidel":
            M = np.tril(A)
            N = M - w * A
            invM = np.linalg.inv(M)
            while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
                x = np.dot(invM, w * b + np.dot(N, x))
                errores.append(np.linalg.norm(np.dot(A, x) - b))
                iteraciones.append(iter_count)
                iter_count += 1
        
        # Graficar el error vs iteraciones
        plt.plot(iteraciones, errores, label=f"{nombre_metodo} ({label})")
    
    # Configuración de la gráfica
    plt.yscale("log")  # Escala logarítmica para visualizar mejor el error
    plt.xlabel("Número de iteraciones")
    plt.ylabel("Error de convergencia (norma)")
    plt.title("Evolución del error de convergencia para diferentes métodos")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------
# FIN DEL CÓDIGO
# ----------------------------------------------------------------------------------------------------------------
