import numpy as np
import sympy as sp
from IPython.display import display, Math, Latex, Markdown
import pprint
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
sp.init_printing()

def factorizacion_lu_sp(A):
    n = A.shape[0] #numero de filas
    L = np.eye(n)  # Matriz L inicializada como la identidad
    U = A.copy()   # U se inicializa como copia de A

    for k in range(n):
        # Realizamos la eliminación hacia adelante en las filas de U
        # Etapa k: columna k
        for i in range(k+1, n):
            # Calculamos el multiplicador para la fila i
            mik = U[i, k] / U[k, k] # multiplicador mik

            # Actualizamos la fila i de U con el pivote Ukk
            U[i, k:] = U[i, k:] - mik * U[k, k:]

            # Actualizamos la matriz L (debajo de la diagonal)
            L[i, k] = mik
    return L, U

def factorizacion_lu_parcial(A):
    n = A.shape[0] #dimensión de la matriz = número de variables
    P = np.eye(n)  # Matriz de permutación (inicialmente identidad)
    L = np.eye(n)  # Matriz triangular inferior (inicialmente identidad)
    U = A.copy()   # U empieza como una copia de A

    for k in range(n):
        # Encontrar el índice del pivote: máximo absoluto en la columna i desde la fila i
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k  # Indice global del pivote
        if k != pivot_row:
            # Intercambiar filas en U
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            # Intercambiar filas en P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # Intercambiar filas correspondientes en L (solo las filas anteriores al pivote)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Realizar la eliminación de Gauss
        for i in range(k+1, n):
            mik = U[i, k] / U[k, k]  # El multiplicador de eliminación
            U[i, k:] = U[i, k:] - mik * U[k, k:]  # Actualizar fila j de U
            L[i, k] = mik  # Guardar el multiplicador en L

    return P, L, U

def sustitución_progresiva(L, b): #forward subs
    n = len(b)
    y = np.zeros_like(b)  # Vector de soluciones, inicializado en ceros
    # Resolución por sustitución hacia adelante
    for i in range(n):
        # La suma de los términos previos multiplicados por las incógnitas ya conocidas
        suma = np.dot(L[i,:i], y[:i])

        # Resolver para la incógnita x[i]
        y[i] = (b[i] - suma) / L[i,i]
    return y

def sustitucion_regresiva(U, y):#backward_substitution
    n = len(y)
    x = np.zeros_like(y)  # Vector de soluciones, inicializado en ceros
    # Resolución por sustitución hacia atrás
    for i in range(n-1, -1, -1):
        # La suma de los términos que ya están resueltos
        suma = np.dot(U[i, i+1:], x[i+1:])
        # Resolver para la incógnita x[i]
        x[i] = (y[i] - suma) / U[i,i]
    return x

def resuelve_sistema_lineal(A,b):
  P,L,U=factorizacion_lu_parcial(A)
  y=sustitución_progresiva(L,P.dot(b)) #Ly=Pb
  x=sustitucion_regresiva(U, y) #Ux
  return x

def resuelve_sistema_lineal_sin_permutacion(A,b):
  L,U = factorizacion_lu_sp(A)
  y=sustitución_progresiva(L,b)
  x=sustitucion_regresiva(U, y)
  return x

def Determinante(A):
    _,U=factorizacion_lu_sp(A)
    return np.prod(np.diag(U))

def inversa_por_gauss_jordan(A):
    n = A.shape[0]
    # Creamos la matriz aumentada [A | I], donde I es la identidad
    A_augmented = np.hstack([A, np.eye(n)])

    for i in range(n):
        # Buscamos el pivote (máximo valor absoluto en la columna i)
        max_row = np.argmax(np.abs(A_augmented[i:n, i])) + i
        if A_augmented[max_row, i] == 0:
            raise ValueError("La matriz no es invertible.")

        # Intercambiamos las filas i y max_row
        A_augmented[[i, max_row]] = A_augmented[[max_row, i]]

        # Hacemos 1 el pivote
        A_augmented[i] = A_augmented[i] / A_augmented[i, i]

        # Hacemos ceros en la columna i para las otras filas
        for j in range(n):
            if j != i:
                A_augmented[j] = A_augmented[j] - A_augmented[j, i] * A_augmented[i]

    # Extraemos la parte de la matriz aumentada que corresponde a la inversa de A
    A_inv = A_augmented[:, n:]

    return A_inv

def metodo_jacobi(A, b, tol=1e-8, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b)  # Vector de soluciones, inicializado en ceros
    x_new = np.zeros_like(b)  # Vector para almacenar los valores de la siguiente iteración

    # Iteraciones del método de Jacobi
    for k in range(max_iter):
        for i in range(n):
            # Calculamos el valor de x_i^{(k+1)} usando la fórmula de Jacobi
            sum_ax = np.dot(A[i, :], x)  # Suma A_ij * x_j
            sum_ax -= A[i, i] * x[i]  # Restamos A_ii * x_i (no se usa el valor de la diagonal)
            x_new[i] = (b[i] - sum_ax) / A[i, i]

        # Comprobamos el criterio de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergió en la iteración {k+1}")
            return x_new

        # Actualizamos el valor de x para la siguiente iteración
        x = x_new.copy()

    print("Alcanzó el número máximo de iteraciones sin converger.")
    return x_new

'''
Ejemplo de uso
A = np.array([[2,-3,-1,4],[-4,9,3,-12],[10,-9,-2,19],[2,-6,-4,-7]], dtype=float)
b = np.array([2,-1,3,5], dtype=float)

display(sp.latex(A))
display(sp.latex(b))
L,U=factorizacion_lu_sp(A)

display(sp.latex(L))
display(sp.latex(U))
'''
'''
Ejemplo de uso
#A = np.array([[4, 7], [2, 6]], dtype=float)
A=np.array([[10, -1, 2, 0],[-1, 11, -1, 3],[2, -1, 10, -1],[0, 3, -1, 8]],dtype=float)
try:
    A_inv = inversa_por_gauss_jordan(A)
    print("La inversa de la matriz A es:")
    print(A_inv)
except ValueError as e:
    print(e)
'''
# Metodos con relajacion 

def RichardsonSOR(A,b,x0,tol,maxiter,w):
    x=x0
    iter=0
    n=len(A)
    N=np.eye(n)-w*A
    while (np.linalg.norm(np.dot(A,x)-b)>tol)&(iter<maxiter):
        x=w*b+np.dot(N,x)
        iter=iter+1
    return x,np.dot(A,x)-b,iter

def JacobiSOR(A,b,x0,tol,maxiter,w):
    x=x0
    iter=0
    n=len(A)
    M=np.diag(np.diag(A))
    N=M-w*A
    invM=np.diag(1/np.diag(A))
    while (np.linalg.norm(np.dot(A,x)-b)>tol)&(iter<maxiter):
        x=np.dot(invM,w*b+np.dot(N,x))
        iter=iter+1
    return x,np.dot(A,x)-b,iter

def GaussSeidelSOR(A,b,x0,tol,maxiter,w):
    x=x0
    iter=0
    n=len(A)
    M=np.tril(A)
    N=M-w*A
    invM=np.linalg.inv(M)
    while (np.linalg.norm(np.dot(A,x)-b)>tol)&(iter<maxiter):
        x=np.dot(invM,w*b+np.dot(N,x))
        iter=iter+1
    return x,np.dot(A,x)-b,iter

'''
-----------------------------------------------------------------------------------------------------------------------------------------
'''


'''
A = scipy.array([ [7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6] ])
P, L, U = scipy.linalg.lu(A)

print( "A:")
pprint.pprint(A)

print( "P:")
pprint.pprint(P)

print( "L:")
pprint.pprint(L)

print( "U:")
pprint.pprint(U)
'''

def errores(A,b):
    #jacobi
    d=np.diag(A)
    D=np.diag(d)
    N = D-A
    Tj=np.linalg.inv(D).dot(N)# D^{-1}N
    print("Tj=",Tj)
    vj,_=np.linalg.eig(Tj)
    print("Jacobi: max autovalor",np.max(np.abs(vj)))
    #Gauss-Seidel
    M=np.tril(A)
    N=M-A
    Tgs=np.linalg.inv(M).dot(N)# M^{-1}N
    print("Tgs=",Tgs)
    vgs,_=np.linalg.eig(Tgs)
    print("Gauss-Seidel: max autovalor",np.max(np.abs(vgs)))
    #Richardson
    n = A.shape[0]
    N = np.eye(n)-A
    Tr = N
    print('Tr=',Tr)
    vr,_=np.linalg.eig(Tr)
    print("Richardson: max autovalor",np.max(np.abs(vr)))

import pandas as pd
from IPython.display import display

tabla_actualizada = pd.DataFrame({
    "Método": ["Richardson", "Jacobi", "Gauss-Seidel"],
    "Punto inicial": ["[0, 0, 0, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 0, 0, 0, 0]"],
    "Número de iteraciones": [152, 18, 5],
    "Estado de convergencia": ["Convergió", "Convergió", "Convergió"],
    "Máximo autovalor de T": [0.9287607649735783, 0.3996020610540534, 0.2106880470619698],
    "Tiempo de procesamiento (s)": [0.003310, 0.000740, 0.001490],
    "Error de convergencia": ["9.571865e-07", "9.720618e-07", "0.000000e+00"]
})
