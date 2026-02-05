import pprint
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import numpy as np
import math


def cholesky1(A):
    
    L = scipy.linalg.cholesky(A, lower=True)
    U = scipy.linalg.cholesky(A, lower=False)

    print("A:")
    pprint.pprint(A)

    print( "L:")
    pprint.pprint(L)

    print( "U:")
    pprint.pprint(U)


def metodoCholesky(A, n):
    """
    Método de Cholesky para descomponer una matriz simétrica y definida positiva 
    en el producto de una matriz triangular inferior (G) y su traspuesta (G^T).

    Parámetros:
    - A: Matriz cuadrada de coeficientes del sistema. Debe ser simétrica y definida positiva.
    - b: Vector de términos independientes. Aunque no se utiliza directamente en esta función, 
         podría ser útil en etapas posteriores para resolver el sistema.
    - n: Dimensión de la matriz cuadrada A (n x n).

    Retorna:
    - G: Matriz triangular inferior resultante de la descomposición de Cholesky.
    - Gt: Matriz traspuesta de G.
    - Si la matriz no es definida positiva, retorna ["NULL", "NULL"].
    """

    # Inicialización de una matriz nula G de dimensiones n x n.
    G = [[0.0] * n] * n

    # Realizamos la descomposición de Cholesky en la matriz A.
    for i in range(n):
        suma = A[i][i]  # Calculamos el valor diagonal actual
        for k in range(i):
            suma -= A[k][i] ** 2  # Restamos los cuadrados de los elementos de la fila superior
        
        if suma < 0:  # Si el valor diagonal es negativo, no es definida positiva
            return ["NULL", "NULL"]
        
        A[i][i] = math.sqrt(suma)  # Calculamos la raíz cuadrada del elemento diagonal
        
        # Llenamos los elementos fuera de la diagonal en la misma fila
        for j in range(i + 1, n):
            suma = A[i][j]
            for k in range(i):
                suma -= A[k][i] * A[k][j]  # Restamos los productos correspondientes
            A[i][j] = suma / A[i][i]  # Dividimos por el valor diagonal correspondiente

    # Transformamos la matriz A en G (matriz triangular inferior)
    for j in range(n):
        for i in range(n):
            if i > j:
                A[i][j] = 0.0  # Forzamos los valores superiores a la diagonal a ser 0

    # Gt es simplemente la traspuesta de G
    Gt = A
    G = np.transpose(Gt)

    # Imprimir resultados
    print('\nMatriz G:')
    print(G)

    print('\nMatriz G transpuesta:')
    print(Gt)

    # Retornamos la matriz G y su traspuesta Gt
    return G, Gt

def factorizacion_cholesky(A):
    m, n = np.shape(A)
    if m != n:
        print("La matriz no es cuadrada.")
        return None

    if A.dtype == complex:
        C = np.zeros((n, n), complex)
    else:
        C = np.zeros((n, n), float)

    for i in range(n):
        C[i, i] = np.sqrt(A[i, i] - sum(abs(C[i, :i])**2))
        if abs(C[i, i]) < 1e-15:
            print("No existe la factorización de Cholesky.")
            return None
        else:
            for j in range(i+1, n):
                C[j, i] = (A[i, j] - sum(C[i, :i]*C[j, :i]))/C[i, i]
    
    return C

matrizA = np.array([[4, 6, 8, 10], [6, 25, 24, 31],[8, 24, 29, 38],[10, 31, 38, 54]])

#metodoCholesky(matrizA, 4)

tamano = 5

def generar_matriz_definida_positiva(tamano):
    # Generar una matriz aleatoria
    A = np.random.rand(tamano, tamano)
    # Multiplicar la matriz por su transpuesta para asegurar que sea definida positiva
    matriz_definida_positiva = np.dot(A, A.T)
    print(matriz_definida_positiva.shape)
    return matriz_definida_positiva 

#matriz3 = generar_matriz_definida_positiva(tamano)

#pprint.pprint(matriz3)

#metodoCholesky(matriz3, tamano)

def resolverSistemaCholesky(A, b):
    """
    Resuelve el sistema Ax = b utilizando la descomposición de Cholesky.
    Parámetros:
    - A: Matriz cuadrada (numpy array) simétrica y definida positiva.
    - b: Vector columna (numpy array) de términos independientes.
    Retorna:
    - x: Solución del sistema Ax = b.
    """
    # Descomposición de Cholesky
    G = factorizacion_cholesky(A)
    Gt = G.T
    # Resolución de los sistemas triangulares
    y = np.linalg.solve(G, b)  # Resolver Gy = b
    x = np.linalg.solve(Gt, y)  # Resolver G^T x = y
    return x

matrizA2= np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
matrizB2= np.array([1, 2, 3])

#print(resolverSistemaCholesky(matrizA2, matrizB2))

A2 = np.array([[1, 2, 3, 7],[2, 3, 4, 8],[3, 4, 5, 9],[4,5,6,10]])
B2 = np.array([9, 12, 15, 17])
sol = resolverSistemaCholesky(A2, B2)
pprint.pprint(sol)