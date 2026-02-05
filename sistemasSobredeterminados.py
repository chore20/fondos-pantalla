import numpy as np
import sympy as sp
from IPython.display import display
import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
import matplotlib.pyplot as plt
import math as ma
import pandas as pd
from fractions import Fraction
from decimal import Decimal
import MatricesOptimizadas as mop

def Regresion(df,variable, objetivo, grado):
    aux=df[[variable,objetivo]]
    aux=aux.sort_values(variable)
    x=aux[[variable]]
    y=aux[[objetivo]]
    x=np.array(x)
    y=np.array(y)
    n=grado
    A=x**n
    for i in range(n):
        A=np.concatenate([A,x**(n-1-i)],axis=1)
    AA=A.T.dot(A)
    Ab=A.T.dot(y)
    sol=np.linalg.solve(AA,Ab)
    yaprox=A.dot(sol)
    res=yaprox-y
    return x,y,sol,yaprox,np.linalg.norm(res)

#---------------------------------------------------------------#

# Datos
sexo = [1, 1, 1, 2, 2, 1]  # 1: Varón, 2: Mujer
edad = [73, 69, 68, 64, 72, 71]
peso = [60, 77, 96, 82, 87, 92]
pas = [130, 155, 158, 134, 150, 144]

datos = pd.DataFrame({'sexo': sexo, 'edad': edad, 'peso': peso, 'pas': pas})
def inciso1(A,b):
    # 1. Modelo de regresión lineal: pas = β0 + β1*sexo + β2*edad + β3*peso
    # Construcción de la matriz de diseño A y el vector b
    A = np.column_stack([np.ones(len(sexo)), datos['sexo'], datos['edad'], datos['peso']])
    b = np.array(datos['pas']).reshape(-1, 1)

    # 2. Resolver el sistema sobredeterminado por mínimos cuadrados
    # Normal ecuación: A.T * A * β = A.T * b
    AT_A = A.T @ A
    AT_b = A.T @ b
    beta = np.linalg.solve(AT_A, AT_b)

    # 3. Calcular los valores ajustados y los errores
    pas_pred = A @ beta  # Valores ajustados
    errores = b - pas_pred  # Residuos

    # 4. Descomposición QR y solución de mínimos cuadrados
    Q, R = np.linalg.qr(A)  # Descomposición QR
    beta_qr = np.linalg.solve(R, Q.T @ b)  # Resolver usando QR

    # 5. Imprimir resultados
    print("Coeficientes (β) usando ecuaciones normales:\n", beta.flatten())
    print("Coeficientes (β) usando descomposición QR:\n", beta_qr.flatten())
    print("Errores para cada dato:\n", errores.flatten())

    # 6. Gráfico de comparación
    plt.scatter(range(len(pas)), pas, color='red', label='PAS Observado')
    plt.plot(range(len(pas)), pas_pred, color='blue', label='PAS Ajustado')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Presión Arterial Sistólica (PAS)')
    plt.legend()
    plt.title('Modelo de Regresión Lineal')
    plt.show()

#--------------------------------------------------------------

# Generar matriz A (4x3)
    A = np.random.rand(4, 3)

def inciso2(A):
    
    def householder_qr(A):
        """
        Calcula la descomposición QR de una matriz A usando las transformaciones de Householder.

        Parámetros:
        - A (ndarray): Matriz de entrada (m x n).

        Retorna:
        - Q (ndarray): Matriz ortogonal (m x m).
        - R (ndarray): Matriz triangular superior (m x n).
        """
        m, n = A.shape
        Q = np.eye(m)  # Matriz identidad de tamaño m (para construir Q)
        R = A.copy()   # Copia de A (para convertir a forma triangular superior)
    
        for i in range(n):  # Iterar sobre las columnas
            # Seleccionar el vector x de la columna i desde la fila i hacia abajo
            x = R[i:, i]
            # Crear el vector e1 (primera base canónica)
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
            # Calcular el vector de Householder v
            v = x - e1
            v = v / np.linalg.norm(v)
            # Crear la matriz de Householder H
            H = np.eye(m)
            H[i:, i:] -= 2.0 * np.outer(v, v)
            # Aplicar H a R (H * R)
            R = H @ R
            # Acumular H en Q (Q * H.T)
            Q = Q @ H.T
    
        return Q, R

    # Aplicar Householder manualmente
    Q_manual, R_manual = householder_qr(A)

    # Comparar con numpy.linalg.qr()
    Q_numpy, R_numpy = np.linalg.qr(A)

    # Verificar los resultados
    print("Matriz original A:\n", A)
    print("\nQ calculado manualmente:\n", Q_manual)
    print("\nR calculado manualmente:\n", R_manual)
    print("\nQ calculado con numpy:\n", Q_numpy)
    print("\nR calculado con numpy:\n", R_numpy)

    # Comprobaciones:
    print("\nVerificación de ortogonalidad (Q.T @ Q ≈ I):\n", np.allclose(Q_manual.T @ Q_manual, np.eye(Q_manual.shape[0])))
    print("\nVerificación de reconstrucción (Q @ R ≈ A):\n", np.allclose(Q_manual @ R_manual, A))

#----------------------------------------------
# Paso 1: Generar la matriz B
B = np.random.uniform(-2, 3, (10, 10))
def inciso3(B):
      # Paso 2: Calcular A = B^T B
    A = B.T @ B

    def qr_algorithm(A, tol=1e-10, max_iter=1000):
        """
        Calcula los autovalores de una matriz simétrica usando el algoritmo de descomposición QR.

        Parámetros:
        - A (ndarray): Matriz simétrica (n x n).
        - tol (float): Tolerancia para la convergencia de los autovalores.
        - max_iter (int): Número máximo de iteraciones.

        Retorna:
        - autovalores (ndarray): Vector con los autovalores de A.
        """
        n = A.shape[0]
        Ak = A.copy()  # Inicializar Ak como la matriz A
        for _ in range(max_iter):
            Q, R = np.linalg.qr(Ak)  # Descomposición QR
            Ak_next = R @ Q          # Iteración Ak+1 = R * Q
            if np.allclose(Ak, Ak_next, atol=tol):  # Verificar convergencia
                break
            Ak = Ak_next
        autovalores = np.diag(Ak)  # Los autovalores están en la diagonal de Ak
        return np.sort(autovalores)  # Retornar en orden ascendente

    # Paso 3: Calcular autovalores usando el algoritmo QR
    autovalores_qr = qr_algorithm(A)

    # Paso 4: Calcular autovalores usando np.linalg.eig()
    autovalores_numpy = np.linalg.eigvals(A)

    # Comparar resultados
    print("Autovalores calculados con QR:\n", autovalores_qr)
    print("\nAutovalores calculados con numpy.linalg.eig():\n", np.sort(autovalores_numpy))

    # Verificar si los resultados coinciden
    print("\n¿Los autovalores coinciden?:", np.allclose(autovalores_qr, np.sort(autovalores_numpy)))

#-------------------------------------------------------
autovalores_dados = [1, 4, 6, 7, 10]

def inciso4(autovalores):
    
    def generar_matriz_con_autovalores(autovalores):
        """
        Genera una matriz no diagonal ni triangular con un conjunto de autovalores dado.

        Parámetros:
        - autovalores (list): Lista de autovalores deseados.

        Retorna:
        - A (ndarray): Matriz generada con los autovalores dados.
        """
        n = len(autovalores)
        # Crear matriz diagonal con los autovalores
        Lambda = np.diag(autovalores)
    
        # Generar una matriz P aleatoria e invertible
        P = np.random.rand(n, n)
        while np.linalg.det(P) == 0:  # Asegurar que P sea invertible
            P = np.random.rand(n, n)
    
        # Calcular A = P Λ P⁻¹
        A = P @ Lambda @ np.linalg.inv(P)
        return A

    def qr_algorithm(A, tol=1e-10, max_iter=1000):
        """
        Calcula los autovalores de una matriz simétrica usando el algoritmo de descomposición QR.

        Parámetros:
        - A (ndarray): Matriz cuadrada.
        - tol (float): Tolerancia para la convergencia de los autovalores.
        - max_iter (int): Número máximo de iteraciones.

        Retorna:
        - autovalores (ndarray): Vector con los autovalores de A.
        """
        n = A.shape[0]
        Ak = A.copy()  # Inicializar Ak como la matriz A
        for _ in range(max_iter):
            Q, R = np.linalg.qr(Ak)  # Descomposición QR
            Ak_next = R @ Q          # Iteración Ak+1 = R * Q
            if np.allclose(Ak, Ak_next, atol=tol):  # Verificar convergencia
                break
            Ak = Ak_next
        autovalores = np.diag(Ak)  # Los autovalores están en la diagonal de Ak
        return np.sort(autovalores)  # Retornar en orden ascendente

    # Paso 1: Generar matriz con autovalores dados
    A = generar_matriz_con_autovalores(autovalores_dados)
    # Paso 2: Calcular autovalores con el algoritmo QR
    autovalores_qr = qr_algorithm(A)

    # Paso 3: Calcular autovalores con numpy.linalg.eig()
    autovalores_numpy = np.linalg.eigvals(A)

    # Resultados
    print("Matriz generada (A):\n", A)
    print("\nAutovalores calculados con QR:\n", autovalores_qr)
    print("\nAutovalores calculados con numpy.linalg.eig():\n", np.sort(autovalores_numpy))
    print("\n¿Los autovalores coinciden?:", np.allclose(autovalores_qr, np.sort(autovalores_numpy)))

#inciso4(autovalores_dados)