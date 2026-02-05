import MatricesOptimizadas as mop1
import MatricesOptimizadas2 as mop2
import raicesOptimizadas as rop
import numpy as np
import sympy as sp
from IPython.display import display
import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
# -------------------------------------------------------------ejercicio 1 ------------------------------------------------------------
#m1 = np.array([[-3, 1], [2, 3]],dtype=float)
#m2 = np.array([-1, 4], dtype=float)
#x = mop1.JacobiSORi(m1, m2, 1, 10e-8, 1000, 0.1)
#print(x)

#--------------------------------------------------------------ejercicio 2 ------------------------------------------------------------
def f1(x):
    return x-2*np.sin(x)

def df1(x):
    return 1 - 2 * np.cos(x)

def grafica(c):
    x=np.linspace(-c,c,200,-200)
    y= x-2*np.sin(x)
    plt.plot(x,y,x,np.zeros(len(x)),'r')
    plt.grid(True)  
    plt.show()
#grafica(10)

x1 = rop.biseccion2(f1, -2, 4, 1e-6, 1000)
x2 = rop.biseccion2(f1, -2, 0, 1e-6, 1000)
x3 = rop.biseccion2(f1, -2, 2, 1e-6, 1000)
#print(x1)
#print(x2)
#print(x3)
#x4 = rop.secante(f1, -2, 4, 1e-6, 1000)
#print(x4)
#x5 = rop.secante(f1, -2, 2, 1e-6, 1000)
#print(x5)
# rop.Newton(f1, df1, 1, 1e-6, 1000)

#----------------------------------------------------------------ejercicio 3 -------------------------------------------------------------------
A = np.array([[24, -2, -1,  3,  3, -2,  1,  3, -2, -1,  3,  2],
[ 3, 18,  1,  2,  2, -1,  1,  0,  0,  3,  3,  1],
[-1,  3, 24, -3, -2,  1, -1, -3, -1,  2,  3, -3],
[-3,  0,  2, 18, -2, -2,  3,  0,  3,  0, -2,  0],
[ 2,  3,  1, -3, 21, -2, -1,  1, -3, -2, -1, -1],
[-1,  3,  3,  3, -1, 23,  3, -3,  1,  1, -3,  0],
[-1, -3,  3,  2, -2,  1, 20,  0,  2,  2, -3,  0],
[-1,  2,  1, -2,  1, -3, -2, 21,  3, -3,  0,  2],
[ 1, -3, -3, -2,  0,  3,  3,  2, 23,  3,  1, -1],
[ 1,  0,  1,  3,  2, -1,  1,  0, -3, 17,  1, -3],
[ 0,  1,  1,  2, -3,  3,  0,  2,  2,  2, 19, -2],
[-1,  1,  3,  0,  1, -2,  3,  0,  3, -2,  0, 17]])

b = np.array([  93,  -24,  -15,   41,   73, -126,   33,   13, -104,   86, 4,-73])

p,l,u = mop1.factorizacion_lu_parcial(A)
#pprint.pprint(p)
#pprint.pprint(l)
#print.pprint(u)

#x7 = mop1.resuelve_sistema_lineal(A,b)
#pprint.pprint(x7)
#xr7 = np.linalg.solve(A,b)
#print()
#pprint.pprint(xr7)
#pprint.pprint(mop1.compara_metodos(A,b))

#mop1.errores(A,b)
import numpy as np
import pandas as pd
import time

# Función de Richardson con Sobrerrelajación
def RichardsonSOR(A, b, x0, tol, maxiter, w):
    """
    Método de Richardson con Sobrerrelajación.

    Parámetros:
    - A: Matriz de coeficientes.
    - b: Vector de términos independientes.
    - x0: Vector inicial.
    - tol: Tolerancia para la convergencia.
    - maxiter: Número máximo de iteraciones.
    - w: Factor de relajación.

    Retorno:
    - x: Vector solución.
    - residuo: Norma del residuo final.
    - iter: Número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    N = np.eye(n) - w * A
    start_time = time.time()
    while (np.linalg.norm(np.dot(A, x) - b) > tol) and (iter < maxiter):
        x = w * b + np.dot(N, x)
        iter += 1
    elapsed_time = time.time() - start_time
    residuo = np.linalg.norm(np.dot(A, x) - b)
    return x, residuo, iter, elapsed_time

# Función de Jacobi con Sobrerrelajación
def JacobiSOR(A, b, x0, tol, maxiter, w):
    """
    Método de Jacobi con Sobrerrelajación.

    Parámetros:
    - A: Matriz de coeficientes.
    - b: Vector de términos independientes.
    - x0: Vector inicial.
    - tol: Tolerancia para la convergencia.
    - maxiter: Número máximo de iteraciones.
    - w: Factor de relajación.

    Retorno:
    - x: Vector solución.
    - residuo: Norma del residuo final.
    - iter: Número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    M = np.diag(np.diag(A))
    N = M - w * A
    invM = np.diag(1 / np.diag(A))
    start_time = time.time()
    while (np.linalg.norm(np.dot(A, x) - b) > tol) and (iter < maxiter):
        x = np.dot(invM, w * b + np.dot(N, x))
        iter += 1
    elapsed_time = time.time() - start_time
    residuo = np.linalg.norm(np.dot(A, x) - b)
    return x, residuo, iter, elapsed_time

# Función de Gauss-Seidel con Sobrerrelajación
def GaussSeidelSOR(A, b, x0, tol, maxiter, w):
    """
    Método de Gauss-Seidel con Sobrerrelajación.

    Parámetros:
    - A: Matriz de coeficientes.
    - b: Vector de términos independientes.
    - x0: Vector inicial.
    - tol: Tolerancia para la convergencia.
    - maxiter: Número máximo de iteraciones.
    - w: Factor de relajación.

    Retorno:
    - x: Vector solución.
    - residuo: Norma del residuo final.
    - iter: Número de iteraciones realizadas.
    """
    x = x0
    iter = 0
    n = len(A)
    M = np.tril(A)
    N = M - w * A
    invM = np.linalg.inv(M)
    start_time = time.time()
    while (np.linalg.norm(np.dot(A, x) - b) > tol) and (iter < maxiter):
        x = np.dot(invM, w * b + np.dot(N, x))
        iter += 1
    elapsed_time = time.time() - start_time
    residuo = np.linalg.norm(np.dot(A, x) - b)
    return x, residuo, iter, elapsed_time

# Matriz de ejemplo y parámetros
A = np.array([[24, -2, -1,  3,  3, -2,  1,  3, -2, -1,  3,  2],
              [ 3, 18,  1,  2,  2, -1,  1,  0,  0,  3,  3,  1],
              [-1,  3, 24, -3, -2,  1, -1, -3, -1,  2,  3, -3],
              [-3,  0,  2, 18, -2, -2,  3,  0,  3,  0, -2,  0],
              [ 2,  3,  1, -3, 21, -2, -1,  1, -3, -2, -1, -1],
              [-1,  3,  3,  3, -1, 23,  3, -3,  1,  1, -3,  0],
              [-1, -3,  3,  2, -2,  1, 20,  0,  2,  2, -3,  0],
              [-1,  2,  1, -2,  1, -3, -2, 21,  3, -3,  0,  2],
              [ 1, -3, -3, -2,  0,  3,  3,  2, 23,  3,  1, -1],
              [ 1,  0,  1,  3,  2, -1,  1,  0, -3, 17,  1, -3],
              [ 0,  1,  1,  2, -3,  3,  0,  2,  2,  2, 19, -2],
              [-1,  1,  3,  0,  1, -2,  3,  0,  3, -2,  0, 17]], dtype=float)

b = np.array([93, -24, -15, 41, 73, -126, 33, 13, -104, 86, 4, -73], dtype=float)

# Condiciones iniciales y parámetros
x0_1 = np.full(len(b), -100, dtype=float)
x0_2 = np.full(len(b), 100, dtype=float)
tol = 1e-8
maxiter = 100
w = 1  # Sin sobre-relajación

# Evaluación de los métodos
results = []
for method, func in zip(["Richardson", "Jacobi", "Gauss-Seidel"], 
                        [RichardsonSOR, JacobiSOR, GaussSeidelSOR]):
    for x0, x0_label in zip([x0_1, x0_2], ["x0 = -100", "x0 = 100"]):
        solution, error, iterations, elapsed_time = func(A, b, x0, tol, maxiter, w)
        results.append({
            "Método": method,
            "Punto inicial": x0_label,
            "Iteraciones": iterations,
            "Error de convergencia": error,
            "Tiempo (s)": elapsed_time,
            "Estado": "Converge" if error < tol else "Diverge"
        })

# Mostrar resultados
df_results = pd.DataFrame(results)
print(df_results)
