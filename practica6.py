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
from scipy.interpolate import lagrange
from scipy import integrate
import interpolacionOptimizada as itp
import derivIntgOTP as diotp 

#ejercicio 1 


def calculate_derivatives_and_errors(f, exact_derivatives, x, h_values):
    """
    Calcula las primeras, segundas y terceras derivadas de una función `f` en el punto `x` 
    utilizando diferencias progresivas, regresivas y centradas. También calcula errores absolutos 
    y relativos comparados con los valores exactos.

    Parámetros:
        f (function): Función objetivo.
        exact_derivatives (list): Valores exactos de la primera, segunda y tercera derivadas.
        x (float): Punto donde se evalúan las derivadas.
        h_values (list): Lista de valores de paso `h`.

    Retorna:
        pd.DataFrame: Tabla con los resultados de las derivadas, errores absolutos y relativos.
    """
    def forward_diff(f, x, h):
        return (f(x + h) - f(x)) / h

    def backward_diff(f, x, h):
        return (f(x) - f(x - h)) / h

    def centered_diff(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_centered_diff(f, x, h):
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

    def third_centered_diff(f, x, h):
        return (f(x + 2*h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2*h)) / (2 * h**3)

    results = []
    for h in h_values:
        # Primera derivada
        d1_fwd = forward_diff(f, x, h)
        d1_bwd = backward_diff(f, x, h)
        d1_ctr = centered_diff(f, x, h)
        
        # Segunda derivada
        d2_ctr = second_centered_diff(f, x, h)
        
        # Tercera derivada
        d3_ctr = third_centered_diff(f, x, h)
        
        # Valores exactos
        exact_d1, exact_d2, exact_d3 = exact_derivatives
        
        # Cálculo de errores con control para evitar divisiones por cero
        errors = {
            "h": h,
            "D1_FWD": d1_fwd, 
            "Error_ABS_D1_FWD": abs(d1_fwd - exact_d1), 
            "Error_REL_D1_FWD": abs(d1_fwd - exact_d1) / abs(exact_d1) if exact_d1 != 0 else np.nan,
            "D1_BWD": d1_bwd, 
            "Error_ABS_D1_BWD": abs(d1_bwd - exact_d1), 
            "Error_REL_D1_BWD": abs(d1_bwd - exact_d1) / abs(exact_d1) if exact_d1 != 0 else np.nan,
            "D1_CTR": d1_ctr, 
            "Error_ABS_D1_CTR": abs(d1_ctr - exact_d1), 
            "Error_REL_D1_CTR": abs(d1_ctr - exact_d1) / abs(exact_d1) if exact_d1 != 0 else np.nan,
            "D2_CTR": d2_ctr, 
            "Error_ABS_D2_CTR": abs(d2_ctr - exact_d2), 
            "Error_REL_D2_CTR": abs(d2_ctr - exact_d2) / abs(exact_d2) if exact_d2 != 0 else np.nan,
            "D3_CTR": d3_ctr, 
            "Error_ABS_D3_CTR": abs(d3_ctr - exact_d3), 
            "Error_REL_D3_CTR": abs(d3_ctr - exact_d3) / abs(exact_d3) if exact_d3 != 0 else np.nan,
        }
        results.append(errors)
    
    return pd.DataFrame(results)

## Ejemplo de uso
#if __name__ == "__main__":
#    # Definición de funciones
#    def f1(x): return ((3*x - 1) / (x**2 + 3))**2
#    def f2(x): return x / (x**2 + 4)**(3/2)
#    def f3(x): return np.log(np.cbrt(1 - x**4))
#
#    # Derivadas exactas en los puntos dados
#    exact_f1 = [1.2, -0.36, 0.192]  # Primera, segunda y tercera derivada de f1 en x = 1
#    exact_f2 = [0.267261, -0.190985, 0.102997]  # Derivadas de f2 en x = 2
#    exact_f3 = [0, -4/3, 0]  # Derivadas de f3 en x = 0
#
#    # Configuración
#    h_values = [0.1, 0.05, 0.01, 0.005, 0.001]
#
#    # Resultados para cada función
#    results_f1 = calculate_derivatives_and_errors(f1, exact_f1, x=1, h_values=h_values)
#    results_f2 = calculate_derivatives_and_errors(f2, exact_f2, x=2, h_values=h_values)
#    results_f3 = calculate_derivatives_and_errors(f3, exact_f3, x=0, h_values=h_values)
#
#    # Mostrar resultados
#    print("Resultados para f1:")
#    print(results_f1)
#
#    print("\nResultados para f2:")
#    print(results_f2)
#
#    print("\nResultados para f3:")
#    print(results_f3)
#
##ejercicio 2
#temperatura = np.array([0,5,10,20,30,40])
#viscosidad = np.array([1.787,1.519,1.002,0.796,0.653])
datos = np.array([[0,1.787],[5,1.519],[10,1.307],[20,1.002],[30,0.795],[40,0.653]])
pol = itp.Vec2Pol(datos)
pprint.pprint(pol) 
## Restaurando los datos tras el reinicio
#
#import pandas as pd
#
## Datos
#temperatura = [0, 5, 10, 20, 30, 40]
#viscosidad = [1.787, 1.519, 1.307, 1.002, 0.796, 0.653]
#
## Paso (asumimos pasos iguales entre puntos)
#h = 5
#
## Inicializamos listas para las derivadas
#primera_derivada = [None] * len(temperatura)
#segunda_derivada = [None] * len(temperatura)
#
## Cálculo de derivadas centradas donde sea posible
#for i in range(1, len(temperatura) - 1):
#    # Primera derivada centrada
#    primera_derivada[i] = (viscosidad[i + 1] - viscosidad[i - 1]) / (2 * h)
#    # Segunda derivada centrada
#    segunda_derivada[i] = (viscosidad[i + 1] - 2 * viscosidad[i] + viscosidad[i - 1]) / (h ** 2)
#
## Para los extremos, usamos fórmulas hacia adelante y hacia atrás
## Primera derivada hacia adelante en el primer punto
#primera_derivada[0] = (viscosidad[1] - viscosidad[0]) / h
## Primera derivada hacia atrás en el último punto
#primera_derivada[-1] = (viscosidad[-1] - viscosidad[-2]) / h
#
## Segunda derivada no calculable en los extremos
#segunda_derivada[0] = None
#segunda_derivada[-1] = None
#
## Crear DataFrame para mostrar resultados
#resultados = pd.DataFrame({
#    "Temperatura (°C)": temperatura,
#    "Viscosidad": viscosidad,
#    "Primera Derivada": primera_derivada,
#    "Segunda Derivada": segunda_derivada,
#})
#
#resultados
#

#tercer ejercicio 
def f1(x): return ((3*x - 1) / (x**2 + 3))**2
def f2(x): return x / (x**2 + 4)**(3/2)
def f3(x): return np.log(np.cbrt(1 - x**4))

def DerivIntuitiva(f,x,h):
    return (f(x+h)-f(x))/h

def DerivO2(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)

def DerivO3(f,x,h):
    return (16*f(x+h)-f(x-2*h)-3*f(x+2*h)-12*f(x))/12/h

def SegDeriv(f,x,h):
    return (f(x+h)+f(x-h)-2*f(x))/h**2

def Laplaciano(u,x,y,h):
    return (u(x+h,y)+u(x-h,y)+u(x,y+h)+u(x,y-h)-4*u(x,y))/h**2

def Richardson(phi,fun,x,h,n):
    v=np.zeros((n,1))
    R=np.zeros((n,n))
    for k in range(n):
        v[k]=phi(fun,x,h/2**k)
        R[k,0]=v[k]
    for j in range(1,n):
        for i in range(j,n):
            R[i,j]=(4**j*R[i,j-1]-R[i-1,j-1])/(4**j-1)
    return np.diag(R),R

#Richardson(DerivO2, f3, 1, 0.2,4)

#-------------------------------------------------------------
#ejercicio 4
#ejercicio 4 
def f1(x):
  return (1)/np.sqrt(1+x**4)
# de 0 a 2

def f2(x):
  return 2/x
#de 2 a 7

f1 = lambda x : 5.0*t**4 + 10.0*t**3 + 20.0*t**2 + 30.0*t + 40.0, 
f2 = lambda x: 1.787*t**5 + 1.519*t**4 + 1.307*t**3 + 1.002*t**2 + 0.795*t + 0.653