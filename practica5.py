import interpolacionOptimizada as itop 
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

#ejercicio 2 
x = np.array([-2,-1,1,2])
y = np.array([-9,8,0,-1])

# Polinomio de Lagrange
p_lagrange = itop.Lagrange(x, y)
#print("Polinomio de interpolación de Lagrange:\n", p_lagrange)

# Interpolación en x = 0
xi = sp.Symbol('t')
interpolacion_x0 = p_lagrange.subs(xi, 0)
#print("\nValor interpolado en x = 0:", interpolacion_x0)

#ejercicio 3 
x = np.array([0, 1, 2, 3])
y = np.array([-1, 6, 31, 18])
t = 1.5
#itop.interpolacionNewtonGrafica(x,y,t)

#ejercicio 4
class Hermite:
    """
    Implementa la interpolación de Hermite.
    """
    x = sp.symbols('x')

    def __init__(self, xi, yi, dyi):
        self.xi, self.yi, self.dyi = xi, yi, dyi
        self.ph = 0  # Polinomio de Hermite
        self.mostrar_en_fracciones = 1  # Mostrar fracciones

    def dl(self, i):
        """Calcula la derivada del polinomio de Lagrange en el índice i."""
        result = 0
        for j in range(len(self.xi)):
            if j != i:
                result += 1 / (self.xi[i] - self.xi[j])
        return result

    def l(self, i):
        """Calcula el polinomio base de Lagrange para el índice i."""
        nume, deno = 1, 1
        for j in range(len(self.xi)):
            if j != i:
                deno *= (self.xi[i] - self.xi[j])
                nume *= (self.x - self.xi[j])
        return nume / deno

    def pol_hermite(self):
        """Calcula el polinomio de Hermite."""
        for i in range(len(self.xi)):
            L_i = self.l(i)
            deriv = self.dyi[i] - 2 * self.yi[i] * self.dl(i)
            self.ph += (self.yi[i] + (self.x - self.xi[i]) * deriv) * (L_i**2)
        return sp.expand(self.ph)

    def evaluar(self, x_val):
        """Evalúa el polinomio de Hermite en un punto dado."""
        return self.ph.subs(self.x, x_val)

    def graficar(self):
        """Grafica el polinomio de Hermite y los puntos de datos."""
        x_vals = np.linspace(min(self.xi) - 1, max(self.xi) + 1, 100)
        y_vals = [self.evaluar(val) for val in x_vals]
        plt.plot(self.xi, self.yi, 'bo', label='Datos')
        plt.plot(x_vals, y_vals, 'r-', label='Polinomio de Hermite')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Interpolación de Hermite')
        plt.legend()
        plt.grid()
        plt.show()

# Datos del problema
xi = [-2, 1]
yi = [-12, 9]
dyi = [22, 10]

# Crear objeto Hermite
#hermite = Hermite(xi, yi, dyi)

# Calcular el polinomio
#polinomio = hermite.pol_hermite()
#print("Polinomio de Hermite:")
#print(polinomio)

# Interpolar en x = -1.5
#x_interp = -1.5
#y_interp = hermite.evaluar(x_interp)
#print(f"El valor interpolado en x = {x_interp} es: {y_interp}")

# Graficar el polinomio
#hermite.graficar()

#ejercicio 5 
x = [-2, -1, 1, 3]
y = [3, 1, 2, -1]
x_interp = 2

#y_interp = itop.spline_cubico_natural(x, y, x_interp)

#ejercicio 6 

import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from fractions import Fraction
from decimal import Decimal
import pandas as pd



def interpolacion_unificada(metodo, x, y, dy=None, x_interp=None, grafica=False):
    """
    Realiza interpolaciones utilizando varios métodos: Sistema Lineal, Lagrange,
    Diferencias Divididas de Newton, Hermite, y Spline cúbico con frontera natural.

    Parámetros:
    metodo : str
        Método a utilizar: 'lineal', 'lagrange', 'newton', 'hermite', o 'spline'.
    x : list
        Coordenadas x de los puntos.
    y : list
        Coordenadas y de los puntos.
    dy : list, opcional
        Derivadas en los puntos (requerido para Hermite).
    x_interp : float, opcional
        Punto donde se desea interpolar.

    Retorna:
    tuple
        Polinomio o vector de coeficientes y el valor interpolado en x_interp.
    """
    x_sym = sp.symbols('x')

    if metodo == 'lineal':
        # Sistema Lineal
        n = len(x)
        A = np.vander(x, n, increasing=True)
        coef = np.linalg.solve(A, y)
        polinomio = sum(coef[i] * x_sym**i for i in range(n))
        valor_interp = np.polyval(coef[::-1], x_interp) if x_interp else None

    elif metodo == 'lagrange':
        # Interpolación de Lagrange
        def l(i):
            num, den = 1, 1
            for j in range(len(x)):
                if j != i:
                    num *= (x_sym - x[j])
                    den *= (x[i] - x[j])
            return num / den

        polinomio = sum(y[i] * l(i) for i in range(len(x)))
        valor_interp = polinomio.subs(x_sym, x_interp) if x_interp else None

    elif metodo == 'newton':
        # Interpolación de Newton
        def diferencias_divididas(x, y):
            n = len(x)
            A = np.zeros((n, n))
            A[:, 0] = y
            for j in range(1, n):
                for i in range(j, n):
                    A[i, j] = (A[i, j-1] - A[i-1, j-1]) / (x[i] - x[i-j])
            return np.diag(A)

        coef = diferencias_divididas(x, y)
        polinomio = coef[0]
        for i in range(1, len(coef)):
            term = coef[i]
            for j in range(i):
                term *= (x_sym - x[j])
            polinomio += term
        valor_interp = polinomio.subs(x_sym, x_interp) if x_interp else None

    elif metodo == 'hermite':
        # Interpolación de Hermite
        if dy is None:
            raise ValueError("Para Hermite, las derivadas 'dy' deben proporcionarse.")

        def dl(i):
            result = 0
            for j in range(len(x)):
                if j != i:
                    result += 1 / (x[i] - x[j])
            return result

        def l(i):
            num, den = 1, 1
            for j in range(len(x)):
                if j != i:
                    den *= (x[i] - x[j])
                    num *= (x_sym - x[j])
            return num / den

        polinomio = 0
        for i in range(len(x)):
            L_i = l(i)
            deriv = dy[i] - 2 * y[i] * dl(i)
            polinomio += (y[i] + (x_sym - x[i]) * deriv) * (L_i**2)
        valor_interp = polinomio.subs(x_sym, x_interp) if x_interp else None

    elif metodo == 'spline':
        # Spline cúbico con frontera natural
        spline = CubicSpline(x, y, bc_type='natural')
        polinomio = None  # No hay polinomio explícito
        valor_interp = spline(x_interp) if x_interp else None

        # Para graficar, devolver el spline
        def spline_grafica(x_vals):
            return spline(x_vals)
    else:
        raise ValueError("Método no reconocido. Usa: 'lineal', 'lagrange', 'newton', 'hermite', o 'spline'.")

    if grafica == True:# Graficar
        x_vals = np.linspace(min(x) - 1, max(x) + 1, 100)
        y_vals = [polinomio.subs(x_sym, val) if polinomio else spline(val) for val in x_vals]
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'bo', label='Datos')
        plt.plot(x_vals, y_vals, 'r-', label=f'Interpolación: {metodo}')
        if x_interp is not None:
            plt.axvline(x=x_interp, color='green', linestyle='--', label=f'x = {x_interp}')
            plt.scatter(x_interp, valor_interp, color='purple', label=f'Interpolación: ({x_interp}, {valor_interp})', zorder=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Interpolación usando {metodo.capitalize()}')
        plt.legend()
        plt.grid()
        plt.show()

    return polinomio, valor_interp


# Ejemplo de uso:
x = [0, 5, 10, 20, 30, 40]
y = [1.787, 1.519, 1.307, 1.002, 0.796, 0.653]
dy = [0, 0, 0, 0, 0, 0]  # Si fuera Hermite, se usarían valores reales.
x_interp = 15

# Sistema Lineal
print("Sistema Lineal:")
print(interpolacion_unificada('lineal', x, y, x_interp=x_interp))

# Lagrange
print("\nLagrange:")
print(interpolacion_unificada('lagrange', x, y, x_interp=x_interp))

# Newton
print("\nNewton:")
print(interpolacion_unificada('newton', x, y, x_interp=x_interp))

# Hermite (opcional, si hay derivadas)
print("\nHermite:")
print(interpolacion_unificada('hermite', x, y, dy=dy, x_interp=x_interp))

# Spline cúbico
print("\nSpline cúbico:")
print(interpolacion_unificada('spline', x, y, x_interp=x_interp))
