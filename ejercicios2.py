import matrices as mtc
import numpy as np
import scipy
import sympy
import scipy.linalg
import pprint
#import np.random


Amatriz1 = np.array([[5, -3, 1, 2], [-5, 4, -3, 1], [10, -8, 8, -4],[-10, 8, -10, 9]])
Bmatriz1 = np.array([[19], [-20], [40], [-37]])

def primeraLU(a):
    p, l, u = scipy.linalg.lu(a)
    print("matriz A")
    pprint.pprint(a)
    print("matriz P")
    pprint.pprint(p)
    print("matriz L")
    pprint.pprint(l)
    print("matriz U")
    pprint.pprint(u)

def segundaLU(a):
    l,u = mtc.factorizacion_lu_sp(a)
    pprint.pprint(a)
    pprint.pprint(l)
    pprint.pprint(u)
    
def terceraLU(a):
    p, l, u = mtc.factorizacion_lu_parcial(a)
    pprint.pprint(a)
    pprint.pprint(p)
    pprint.pprint(l)
    pprint.pprint(u)

#segundaLU(a)
#terceraLU(a)
def insico2(a, b):
    l, u = mtc.factorizacion_lu_sp(a)

    y = mtc.sustitución_progresiva(l, b)
    # pprint.pprint(y)
    x = mtc.sustitucion_regresiva(u, y)
    print("resolucion por sistemas triangulares ")
    pprint.pprint(x)
    r = np.linalg.solve(a, b)
    print("resolucion por np.linalg.solve ")
    pprint.pprint(r)
    # print(mtc.resuelve_sistema_lineal_sin_permutacion(Amatriz1, Bmatriz1))

# insico2(Amatriz1, Bmatriz1)

# print(mtc.Determinante(Amatriz1))

Amatriz2=np.array([[16, -2, -1,  3,  3, -2,  1,  3],
       [-2, 16,  3,  2,  3, -2,  1,  2],
       [ 2, -1, 11,  0,  0,  3,  3,  1],
       [-1,  3,  2, 14, -2,  1, -1, -3],
       [-1,  2,  3, -3, 15,  0,  2,  3],
       [-2, -2,  3,  0,  3, 13, -2,  0],
       [ 2,  3,  1, -3, -1, -2, 14,  1],
       [-3, -2, -1, -1, -1,  3,  3, 15]],dtype=float)

Bmatriz2=np.array([-13, 110, -61, -37, 124, -75,  96,  31],dtype=float)

# primeraLU(Amatriz2) 
#print(mtc.resuelve_sistema_lineal_sin_permutacion(Amatriz2, Bmatriz2))
#print(np.linalg.solve(Amatriz2, Bmatriz2))

n= 5000

from time import process_time
def tiempo_proceso(n,num=244): #n=tamaño de la matriz A
    np.random.seed(num)
    A = 6+4*np.random.random((n,n))
    b = 10+5*np.random.random(n)
    #tiempo de proceso de solución con script personal
    t1_inicio_personal = process_time()
    r = mtc.resuelve_sistema_lineal(A, b)# Ax=b personal
    t1_fin_personal = process_time()
    t1_personal=t1_fin_personal-t1_inicio_personal
    #tiempo de proceso de solución con Python
    t2_inicio_python = process_time()
    x = np.linalg.solve(A,b)# Ax = b
    t2_fin_python = process_time()
    t2_python=t2_fin_python-t2_inicio_python
    print("para tiempo = ", n)
    return t1_personal,t2_python

# print(tiempo_proceso(n))

#import numpy as np
import matplotlib.pyplot as plt
def grafica_tiempos(a):
# Datos iniciales
    T1 = np.array([0.013702161999999962, 0.055114378999999936, 0.3517668079999998, 
                   1.8293395600000002, 8.645131509, 92.81534519200001])
    T2 = np.array([0.00033058699999966024, 0.0004239250000002137, 0.004367353999999768, 
                   0.3422941210000001, 1.1267749259999995, 2.6681982610000006])

    # Simular la función `tiempo_proceso`
    def tiempo_proceso(n):
        # Función de ejemplo: puedes reemplazar con el cálculo real
        t1 = n * 0.01  # Cambiar por la lógica de tiempo personalizado
        t2 = n * 0.005  # Cambiar por la lógica de tiempo en Python
        return t1, t2

    # Crear el rango de tamaños de problemas
    tps = list(range(100, 2000, 100))

    # Añadir valores a T1 y T2
    for n in tps:
        t1, t2 = tiempo_proceso(n)
        T1 = np.append(T1, t1)
        T2 = np.append(T2, t2)

    # Graficar
    plt.plot(range(len(T1)), T1, label="Personalizado")
    plt.plot(range(len(T2)), T2, label="Python")
    plt.xlabel("Índice del tamaño del problema")
    plt.ylabel("Tiempo de procesamiento (s)")
    plt.title("Comparación de tiempos de procesamiento")
    plt.legend()
    plt.show()

# grafica_tiempos(0)
#x0= np.zeros_like(Bmatriz2)
#print(mtc.RichardsonSOR(Amatriz2, Bmatriz2, x0, 10**-8,100, 0.1))
#print("-------------------")
#print(mtc.JacobiSOR(Amatriz2, Bmatriz2, x0, 10**-8, 100, 0.1))
#print("---------------------------------")
#print(mtc.GaussSeidelSOR(Amatriz2, Bmatriz2, x0, 10**-8, 100, 0.1))
# mtc.errores(Amatriz2, Bmatriz2)

import numpy as np
import pandas as pd
import time

# Funciones de los métodos
def RichardsonSOR(A, b, x0, tol, maxiter, w):
    x = x0
    iter_count = 0
    n = len(A)
    N = np.eye(n) - w * A
    start_time = time.time()
    while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
        x = w * b + np.dot(N, x)
        iter_count += 1
    elapsed_time = time.time() - start_time
    T = np.eye(n) - w * A  # Matriz de iteración
    max_eigenvalue = max(abs(np.linalg.eigvals(T)))
    return x, np.linalg.norm(np.dot(A, x) - b), iter_count, elapsed_time, max_eigenvalue

def JacobiSOR(A, b, x0, tol, maxiter, w):
    x = x0
    iter_count = 0
    n = len(A)
    M = np.diag(np.diag(A))
    N = M - w * A
    invM = np.diag(1 / np.diag(A))
    start_time = time.time()
    while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
        x = np.dot(invM, w * b + np.dot(N, x))
        iter_count += 1
    elapsed_time = time.time() - start_time
    T = np.dot(invM, N)  # Matriz de iteración
    max_eigenvalue = max(abs(np.linalg.eigvals(T)))
    return x, np.linalg.norm(np.dot(A, x) - b), iter_count, elapsed_time, max_eigenvalue

def GaussSeidelSOR(A, b, x0, tol, maxiter, w):
    x = x0
    iter_count = 0
    n = len(A)
    M = np.tril(A)
    N = M - w * A
    invM = np.linalg.inv(M)
    start_time = time.time()
    while np.linalg.norm(np.dot(A, x) - b) > tol and iter_count < maxiter:
        x = np.dot(invM, w * b + np.dot(N, x))
        iter_count += 1
    elapsed_time = time.time() - start_time
    T = np.dot(invM, N)  # Matriz de iteración
    max_eigenvalue = max(abs(np.linalg.eigvals(T)))
    return x, np.linalg.norm(np.dot(A, x) - b), iter_count, elapsed_time, max_eigenvalue

# Matrices y parámetros
A = np.array([[16, -2, -1, 3, 3, -2, 1, 3],
              [-2, 16, 3, 2, 3, -2, 1, 2],
              [2, -1, 11, 0, 0, 3, 3, 1],
              [-1, 3, 2, 14, -2, 1, -1, -3],
              [-1, 2, 3, -3, 15, 0, 2, 3],
              [-2, -2, 3, 0, 3, 13, -2, 0],
              [2, 3, 1, -3, -1, -2, 14, 1],
              [-3, -2, -1, -1, -1, 3, 3, 15]], dtype=float)

b = np.array([-13, 110, -61, -37, 124, -75, 96, 31], dtype=float)
x0_1 = np.full(len(b), -100, dtype=float)
x0_2 = np.full(len(b), 100, dtype=float)
tol = 1e-8
maxiter = 100
w = 1  # Sin sobre-relajación

# Aplicar los métodos
results = []
for method, func in zip(["Richardson", "Jacobi", "Gauss-Seidel"], 
                        [RichardsonSOR, JacobiSOR, GaussSeidelSOR]):
    for x0, x0_label in zip([x0_1, x0_2], ["x0 = -100", "x0 = 100"]):
        solution, error, iterations, elapsed_time, max_eigenvalue = func(A, b, x0, tol, maxiter, w)
        results.append({
            "Método": method,
            "Punto inicial": x0_label,
            "Iteraciones": iterations,
            "Error de convergencia": error,
            "Tiempo (s)": elapsed_time,
            "Máx. autovalor T": max_eigenvalue,
            "Estado": "Converge" if error < tol else "Diverge"
        })

# Crear una tabla con los resultados
df_results = pd.DataFrame(results)
print(df_results)
