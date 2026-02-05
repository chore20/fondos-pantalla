import numpy as np
import sympy as sp
from IPython.display import display
import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
import matplotlib.pyplot as plt
import math
import pandas as pd
from fractions import Fraction
from decimal import Decimal
from scipy.interpolate import lagrange
import scipy.integrate as integrate

#-------------------------------------Derivadas--------------------------------------

# Derivada hacia adelante (primera orden de precisión)
def DerivIntuitiva(f, x, h):
    """
    Aproxima la derivada de la función f en el punto x usando diferencias hacia adelante.
    Fórmula: f'(x) ≈ (f(x + h) - f(x)) / h
    Parámetros:
        - f: Función objetivo.
        - x: Punto donde se evalúa la derivada.
        - h: Incremento o paso.
    """
    return (f(x + h) - f(x)) / h

# Derivada centrada (segunda orden de precisión)
def DerivO2(f, x, h):
    """
    Aproxima la derivada de la función f en el punto x usando diferencias centradas.
    Fórmula: f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
    Parámetros:
        - f: Función objetivo.
        - x: Punto donde se evalúa la derivada.
        - h: Incremento o paso.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Derivada avanzada (tercera orden de precisión)
def DerivO3(f, x, h):
    """
    Aproxima la derivada de la función f en el punto x con mayor precisión (orden 3).
    Fórmula: f'(x) ≈ (16*f(x+h) - f(x-2*h) - 3*f(x+2*h) - 12*f(x)) / (12 * h)
    Parámetros:
        - f: Función objetivo.
        - x: Punto donde se evalúa la derivada.
        - h: Incremento o paso.
    """
    return (16 * f(x + h) - f(x - 2 * h) - 3 * f(x + 2 * h) - 12 * f(x)) / (12 * h)

# Segunda derivada
def SegDeriv(f, x, h):
    """
    Aproxima la segunda derivada de la función f en el punto x.
    Fórmula: f''(x) ≈ (f(x + h) + f(x - h) - 2 * f(x)) / h^2
    Parámetros:
        - f: Función objetivo.
        - x: Punto donde se evalúa la derivada.
        - h: Incremento o paso.
    """
    return (f(x + h) + f(x - h) - 2 * f(x)) / h**2

# Laplaciano para una función escalar u(x, y)
def Laplaciano(u, x, y, h):
    """
    Aproxima el laplaciano de una función escalar u(x, y) en el punto (x, y).
    Fórmula: Δu ≈ (u(x+h, y) + u(x-h, y) + u(x, y+h) + u(x, y-h) - 4*u(x, y)) / h^2
    Parámetros:
        - u: Función escalar u(x, y).
        - x: Coordenada x del punto.
        - y: Coordenada y del punto.
        - h: Incremento o paso.
    """
    return (u(x + h, y) + u(x - h, y) + u(x, y + h) + u(x, y - h) - 4 * u(x, y)) / h**2

# Extrapolación de Richardson (1D)
def Richardson(phi, fun, x, h, n):
    """
    Aplica la extrapolación de Richardson para mejorar la precisión de una aproximación numérica.
    Parámetros:
        - phi: Función que realiza la aproximación inicial.
        - fun: Función objetivo.
        - x: Punto donde se evalúa la derivada.
        - h: Paso inicial.
        - n: Número de niveles de refinamiento.
    Salida:
        - Diagonal principal de la matriz R (aproximaciones mejoradas).
        - Matriz completa R.
    """
    v = np.zeros((n, 1))  # Vector de aproximaciones
    R = np.zeros((n, n))  # Matriz de Richardson
    for k in range(n):
        v[k] = phi(fun, x, h / 2**k)
        R[k, 0] = v[k]
    for j in range(1, n):
        for i in range(j, n):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    return np.diag(R), R

# Extrapolación de Richardson (2D)
def Richardson2d(phi, fun, x, y, h, n):
    """
    Aplica la extrapolación de Richardson para una función bidimensional.
    Parámetros:
        - phi: Función que realiza la aproximación inicial.
        - fun: Función objetivo.
        - x, y: Coordenadas donde se evalúa la derivada.
        - h: Paso inicial.
        - n: Número de niveles de refinamiento.
    Salida:
        - Diagonal principal de la matriz R (aproximaciones mejoradas).
        - Matriz completa R.
    """
    v = np.zeros((n, 1))  # Vector de aproximaciones
    R = np.zeros((n, n))  # Matriz de Richardson
    for k in range(n):
        v[k] = phi(fun, x, y, h / 2**k)
        R[k, 0] = v[k]
    for j in range(1, n):
        for i in range(j, n):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    return np.diag(R), R

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


#-----------------------------------------------------integrales------------------------------------------------------------------------

from sympy import Eq, S, Matrix, Number, MatMul
from sympy.utilities.lambdify import lambdify
import scipy.linalg as al
from mpl_toolkits.mplot3d import Axes3D

def solve_poisson_equation(Lx=1, Ly=1, n=64, m=64, Uizq=0, Uder=0, Uinf=0, Usup=0, display_intermediate=False):
    """
    Resuelve la ecuación de Poisson para una placa rectangular con potencial eléctrico en sus bordes.
    Muestra gráficos 3D del potencial calculado y el mapa de calor.

    Args:
        Lx (float): Longitud del lado x de la placa.
        Ly (float): Longitud del lado y de la placa.
        n (int): Número de nodos en la dirección x.
        m (int): Número de nodos en la dirección y.
        Uizq, Uder, Uinf, Usup (float): Condiciones de frontera en los lados izquierdo, derecho, inferior y superior.
        display_intermediate (bool): Si es True, muestra pasos intermedios con Sympy.

    Returns:
        np.ndarray: Matriz con los valores del potencial eléctrico.
    """
    # Función auxiliar para redondear expresiones
    def round_expr(expr, num_digits):
        return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})

    # Definición de la función h(x, y)
    h = lambda x, y: -6000 * np.exp(x**4.5) * (
        x**2 * (0.75 - 1.5*y) + x**5.5*y*(7.125*y**2 - 10.6875*y + 3.5625)
        + x**4.5*y*(-8.4375*y**2 + 12.6563*y - 4.21875)
        + x**3.5*y*(2.0625*y**2 - 3.09375*y + 1.03125)
        + x**10*y*(3.375*y**2 - 5.0625*y + 1.6875)
        + x**9*y*(-5.0625*y**2 + 7.59375*y - 2.53125)
        + x**8*y*(1.6875*y**2 - 2.53125*y + 0.84375)
        + x**3*(y - 0.5) + x*(y**3 - 1.5*y**2 + y - 0.25)
        + y*(-0.5*y**2 + 0.75*y - 0.25)
    )

    # Parámetros
    L, M = n - 1, m - 1
    N = L * M
    dx, dy = Lx / n, Ly / m

    # Nodos
    xi = [i * dx for i in range(n + 1)]
    yj = [j * dy for j in range(m + 1)]

    if display_intermediate and n < 15:
        display(Eq(S('x_i'), Matrix(xi).T, evaluate=False))
    if display_intermediate and m < 15:
        display(Eq(S('y_j'), Matrix(yj).T, evaluate=False))

    # Construcción de la matriz del sistema
    lam = dx**2 / dy**2
    d1 = np.ones(N - 1)
    i1 = [L * k - 1 for k in range(1, L) if L * k <= N - 1]
    d1[i1] = 0
    d2 = -2 * (1 + lam) * np.ones(N)
    d3 = lam * np.ones(N - L)
    b = np.zeros(N)

    for j in range(1, m):
        for i in range(1, n):
            b[i + (j - 1) * L - 1] = -dx**2 * h(xi[i], yj[j])

    A = np.diag(d2) + np.diag(d1, -1) + np.diag(d1, 1) + np.diag(d3, L) + np.diag(d3, -L)
    if display_intermediate and n < 15:
        display(Eq(MatMul(Matrix(A), S('x')), Matrix(b), evaluate=False))

    # Resolución del sistema lineal
    x = al.solve(A, b)

    if display_intermediate and n < 15:
        display(Eq(S('x'), Matrix(x), evaluate=False))

    # Reconstrucción de la solución en la malla
    u = np.zeros((m + 1, n + 1))
    u[:, 0] = Uinf
    u[:, m] = Usup
    u[0, :] = Uizq
    u[n, :] = Uder
    for j in range(1, m):
        for i in range(1, n):
            u[i, j] = x[i + (j - 1) * L - 1]

    if display_intermediate and n < 15:
        display(round_expr(Matrix(u.T), 5))

    # Gráficos
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(xi, yj)
    surf = ax.plot_surface(X, Y, u.T, cmap=plt.cm.coolwarm)
    ax.contour(X, Y, u.T, 10, cmap=plt.cm.cividis, linestyles="solid", offset=-1)
    ax.contour(X, Y, u.T, 10, colors="k", linestyles="solid")
    ax.set_xlabel('Lado y', labelpad=5)
    ax.set_ylabel('Lado x', labelpad=5)
    ax.set_zlabel('Potencial Eléctrico U', labelpad=5)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

    # Mapa de calor
    fig = plt.figure()
    ax1 = plt.contourf(X, Y, u.T)
    fig.colorbar(ax1, orientation='horizontal')
    plt.show()

    return u


def solve_poisson_jacobi(Lx=1, Ly=1, n=8, m=8, Uizq=0, Uder=0, Uinf=0, Usup=0, tol=1e-8, display_intermediate=False):
    """
    Resuelve la ecuación de Poisson usando el método de Jacobi para una placa rectangular con potencial eléctrico
    en sus bordes. Muestra gráficos 3D del potencial calculado y el mapa de calor.

    Args:
        Lx (float): Longitud del lado x de la placa.
        Ly (float): Longitud del lado y de la placa.
        n (int): Número de nodos en la dirección x.
        m (int): Número de nodos en la dirección y.
        Uizq, Uder, Uinf, Usup (float): Condiciones de frontera en los lados izquierdo, derecho, inferior y superior.
        tol (float): Tolerancia para el criterio de convergencia.
        display_intermediate (bool): Si es True, muestra pasos intermedios con Sympy.

    Returns:
        np.ndarray: Matriz con los valores del potencial eléctrico.
    """
    # Función auxiliar para redondear expresiones
    def round_expr(expr, num_digits):
        return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})

    # Definición de la función h(x, y)
    h = lambda x, y: -6000 * np.exp(x**4.5) * (
        x**2 * (0.75 - 1.5*y) + x**5.5*y*(7.125*y**2 - 10.6875*y + 3.5625)
        + x**4.5*y*(-8.4375*y**2 + 12.6563*y - 4.21875)
        + x**3.5*y*(2.0625*y**2 - 3.09375*y + 1.03125)
        + x**10*y*(3.375*y**2 - 5.0625*y + 1.6875)
        + x**9*y*(-5.0625*y**2 + 7.59375*y - 2.53125)
        + x**8*y*(1.6875*y**2 - 2.53125*y + 0.84375)
        + x**3*(y - 0.5) + x*(y**3 - 1.5*y**2 + y - 0.25)
        + y*(-0.5*y**2 + 0.75*y - 0.25)
    )

    # Parámetros
    dx, dy = Lx / n, Ly / m
    xi = [i * dx for i in range(n + 1)]
    yj = [j * dy for j in range(m + 1)]
    lam = dx**2 / dy**2

    # Inicialización de la malla
    u = np.zeros((n + 1, m + 1))
    u[:, 0] = Uinf
    u[:, m] = Usup
    u[0, :] = Uizq
    u[n, :] = Uder

    # Mostrar nodos iniciales si se requiere
    if display_intermediate and n < 15:
        display(Eq(S('x_i'), Matrix(xi).T, evaluate=False))
        display(Eq(S('y_j'), Matrix(yj).T, evaluate=False))

    # Método de Jacobi
    err, norm1 = 1, al.norm(u)
    while err > tol:
        norm0 = norm1
        for j in range(1, m):
            for i in range(1, n):
                u[i, j] = (dx**2 * h(xi[i], yj[j]) + lam * u[i, j-1] + u[i-1, j] 
                           + u[i+1, j] + lam * u[i, j+1]) / (2 * (1 + lam))
        norm1 = al.norm(u)
        err = np.abs(norm1 - norm0)

    # Mostrar matriz resultado si se requiere
    if display_intermediate and n < 15:
        display(round_expr(Matrix(u.T), 5))

    # Gráficos
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(xi, yj)
    surf = ax.plot_surface(X, Y, u.T, cmap=plt.cm.coolwarm)
    ax.contour(X, Y, u.T, 10, cmap=plt.cm.cividis, linestyles="solid", offset=-1)
    ax.contour(X, Y, u.T, 10, colors="k", linestyles="solid")
    ax.set_xlabel('Lado x', labelpad=5)
    ax.set_ylabel('Lado y', labelpad=5)
    ax.set_zlabel('Potencial Eléctrico U', labelpad=5)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

    # Mapa de calor
    fig = plt.figure()
    ax1 = plt.contourf(X, Y, u.T)
    fig.colorbar(ax1, orientation='horizontal')
    plt.show()

    return u


# Clase para integrar funciones utilizando métodos numéricos
class IntegracionNumerica:
    method_name = "Integración Numérica: Newton-Cotes"

    def __init__(self, x, y, a=0, b=2, f=0):
        """
        Inicializa los datos necesarios para la integración.
        Parámetros:
            - x, y: Arrays de puntos de la función a integrar.
            - a, b: Límites del intervalo de integración.
            - f: Función a integrar (si es conocida).
        """
        self.x, self.y, self.a, self.b, self.f = np.array(x), np.array(y), a, b, f
        self.n = len(self.x)  # Número de puntos
        self.integral = 0  # Resultado de la integral
        self.t = self.x  # Puntos de evaluación
        self.p = self.y  # Valores de la función
        self.h = self.x[1:] - self.x[:self.n-1]  # Distancia entre puntos consecutivos
        print("{:^10}".format(self.method_name))
        self.riemann_point = 'inf'  # Punto de evaluación para el método de Riemann

    def riemann(self):
        """
        Calcula la integral usando la regla de Riemann.
        Dependiendo de `self.riemann_point`, evalúa en los puntos:
            - 'inf': Inferior (izquierda).
            - 'med': Medio.
            - 'sup': Superior (derecha).
        """
        if self.riemann_point == 'inf':
            self.t = self.x[:self.n-1]
        elif self.riemann_point == 'med':
            self.t = (self.x[1:] + self.x[:self.n-1]) / 2
        elif self.riemann_point == 'sup':
            self.t = self.x[1:self.n]
        else:
            self.t = self.x[:self.n-1]
        self.p = self.f(self.t)  # Evalúa la función en los puntos seleccionados
        self.integral = self.p.dot(self.h)  # Producto escalar de alturas y bases
        print("Riemann = {}".format(self.integral))
        return self.integral

    def riemann_plot(self):
        """
        Grafica el método de Riemann con rectángulos.
        Muestra cómo se aproximan las áreas bajo la curva.
        """
        n_suave = self.n * 10  # Para suavizar la curva de la función
        xk = np.linspace(self.a, self.b, n_suave)
        fk = self.f(xk)
        plt.plot(xk, fk, label='f(x)')
        plt.scatter(self.t, self.p, marker='o', color='orange', label=self.riemann_point)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Integral: Método de Riemann')
        plt.legend(loc="best")
        for i in range(self.n-1):
            plt.fill_between([self.x[i], self.x[i+1]], 0, [self.p[i], self.p[i]],
                             color='g', alpha=0.9)
            plt.plot([self.x[i], self.x[i+1]], [self.p[i], self.p[i]], color='orange')
        for i in range(self.n):
            plt.axvline(self.x[i], color='w')
        plt.show()

    def trapecio(self):
        """
        Calcula la integral usando la regla del trapecio.
        Fórmula: Integral ≈ (y[i] + y[i+1]) * h / 2
        """
        self.integral = (self.y[:self.n-1] + self.y[1:]).dot(self.h) / 2
        print("Trapecio = {}".format(self.integral))
        return self.integral

    def trapecio_plot(self):
        """
        Grafica el método del trapecio mostrando la aproximación lineal.
        """
        n_suave = self.n * 10
        xk = np.linspace(self.a, self.b, n_suave)
        fk = self.f(xk)
        plt.plot(xk, fk, label='f(x)')
        plt.plot(self.x, self.y, marker='o', color='orange', label='Lineal')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Integral: Regla de Trapecios')
        plt.legend(loc="best")
        plt.fill_between(self.x, 0, self.y, color='b', alpha=0.9)
        for i in range(self.n):
            plt.axvline(self.x[i], color='w')
        plt.show()

    def simpson13(self):
        """
        Calcula la integral usando la regla de Simpson 1/3.
        Requiere que el número de puntos sea impar.
        Fórmula: Integral ≈ (y[i] + 4*y[i+1] + y[i+2]) * h / 3
        """
        if self.n % 2 == 1:
            simp = [self.y[i] + 4*self.y[i+1] + self.y[i+2] for i in np.arange(0, self.n-2, 2)]
            self.integral = np.array(simp).dot(self.h[0:self.n-2:2]) / 3
            print("Simpson 1/3 = {}".format(self.integral))
        else:
            self.integral = None
            print('Para Simpson 1/3, el número de puntos debe ser impar')
        return self.integral

    def simpson13_plot(self):
        """
        Grafica el método de Simpson 1/3 mostrando aproximaciones parabólicas.
        """
        if self.n % 2 != 1:
            return None
        n_suave = self.n * 10
        xk = np.linspace(self.a, self.b, n_suave)
        fk = self.f(xk)
        plt.plot(xk, fk, label='f(x)')
        plt.scatter(self.x, self.y, marker='o', color='orange', label='Parabólico')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Integral: Regla de Simpson 1/3')
        for i in np.arange(0, self.n-2, 2):
            p2 = lagrange(self.x[i:i+3], self.y[i:i+3])  # Aproximación parabólica
            x2 = np.linspace(self.x[i], self.x[i+2], 30)
            y2 = p2(x2)
            plt.plot(x2, y2, color='orange')
            plt.fill_between(x2, 0, y2, color='blueviolet', alpha=0.9)
        for i in range(self.n):
            plt.axvline(self.x[i], color='w')
        plt.legend(loc="best")
        plt.show()

    def simpson38(self):
        """
        Calcula la integral usando la regla de Simpson 3/8.
        Requiere que el número de puntos sea múltiplo de 3 + 1.
        Fórmula: Integral ≈ (y[i] + 3*y[i+1] + 3*y[i+2] + y[i+3]) * 3*h / 8
        """
        if self.n % 3 == 1:
            simp = [self.y[i] + 3*self.y[i+1] + 3*self.y[i+2] + self.y[i+3]
                    for i in np.arange(0, self.n-3, 3)]
            self.integral = 3 * np.array(simp).dot(self.h[0:self.n-3:3]) / 8
            print("Simpson 3/8 = {}".format(self.integral))
        else:
            self.integral = None
            print('Para Simpson 3/8, el número de puntos debe ser múltiplo de tres + 1')
        return self.integral

    def simpson38_plot(self):
        """
        Grafica el método de Simpson 3/8 mostrando aproximaciones cúbicas.
        """
        if self.n % 3 != 1:
            return None
        n_suave = self.n * 10
        xk = np.linspace(self.a, self.b, n_suave)
        fk = self.f(xk)
        plt.plot(xk, fk, label='f(x)')
        plt.scatter(self.x, self.y, marker='o', color='orange', label='Cúbico')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Integral: Regla de Simpson 3/8')
        for i in np.arange(0, self.n-3, 3):
            p2 = lagrange(self.x[i:i+4], self.y[i:i+4])  # Aproximación cúbica
            x2 = np.linspace(self.x[i], self.x[i+3], 30)
            y2 = p2(x2)
            plt.plot(x2, y2, color='coral')
            plt.fill_between(x2, 0, y2, color='teal', alpha=0.9)
        for i in range(self.n):
            plt.axvline(self.x[i], color='w')
        plt.legend(loc="best")
        plt.show()
        '''
        Ejemplo de uso 
        a, b, N = 2, 7, 9
        #f = lambda x: 2*np.sin(3*x)
        f = lambda x: 2/x
        #f = lambda x: (1)/np.sqrt(1+x**4)
        x = np.linspace(a,b,N+1) #print("x = {}".format(x))
        y = f(x)
        
        integ=IntegracionNumerica(x,y,a,b,f)
        integ.riemann_point = 'med' #inf(default), med, sup
        valR=integ.riemann()
        integ.riemann_plot()
        valT=integ.trapecio()
        integ.trapecio_plot()
        integ.simpson13()
        integ.simpson13_plot()
        integ.simpson38()
        integ.simpson38_plot()
        '''


def ejecutar_integracion(a, b, N, f):
    """
    Ejecuta los métodos de integración numérica y muestra los resultados y gráficos.
    Parámetros:
        - a, b: Límites de integración.
        - N: Número de subdivisiones (puntos - 1).
        - f: Función a integrar.
    """
    # Define los puntos y evalúa la función
    x = np.linspace(a, b, N+1)
    y = f(x)
    
    # Crea la instancia de la clase IntegracionNumerica
    integ = IntegracionNumerica(x, y, a, b, f)
    
    # Ejecuta el método de Riemann (punto medio por defecto)
    integ.riemann_point = 'med'  # Opciones: 'inf', 'med', 'sup'
    valR = integ.riemann()
    print("Resultado Riemann (punto medio):", valR)
    integ.riemann_plot()
    
    # Ejecuta el método del trapecio
    valT = integ.trapecio()
    print("Resultado Trapecio:", valT)
    integ.trapecio_plot()
    
    # Ejecuta el método de Simpson 1/3
    valS13 = integ.simpson13()
    if valS13 is not None:
        print("Resultado Simpson 1/3:", valS13)
        integ.simpson13_plot()
    
    # Ejecuta el método de Simpson 3/8
    valS38 = integ.simpson38()
    if valS38 is not None:
        print("Resultado Simpson 3/8:", valS38)
        integ.simpson38_plot()


# Función de integración por el método del trapecio
def IT(f, a, b, n):
    """
    Calcula la integral definida de la función `f` en el intervalo [a, b]
    usando la regla del trapecio con `n` subdivisiones.

    Parámetros:
        f (function): Función a integrar.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        n (int): Número de subdivisiones.

    Retorna:
        float: Valor aproximado de la integral.
    """
    h = (b - a) / n  # Ancho de cada subintervalo
    # Suma los trapecios definidos por los puntos de la partición
    return sum([(f(a + i * h) + f(a + (i + 1) * h)) * h / 2 for i in range(int(n))])


# Función auxiliar para mostrar resultados en formato de tabla
def print_row(lst):
    """
    Imprime una fila formateada con valores numéricos.

    Parámetros:
        lst (list): Lista de valores numéricos.
    """
    print(' '.join('%11.8f' % x for x in lst))


# Integración por el método de Romberg
def romberg(f, a, b, eps=1E-8):
    """
    Calcula la integral definida de `f` en el intervalo [a, b] usando
    el método de Romberg, que mejora la precisión con extrapolación de Richardson.

    Parámetros:
        f (function): Función a integrar.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        eps (float): Tolerancia para el criterio de convergencia.

    Retorna:
        float: Valor aproximado de la integral.
    """
    R = [[0.5 * (b - a) * (f(a) + f(b))]]  # Primera aproximación (trapecio)
    print_row(R[0])
    n = 1  # Nivel de refinamiento
    while True:
        h = float(b - a) / 2**n  # Longitud del subintervalo
        R.append((n + 1) * [None])  # Añade una nueva fila en R
        # Refinar usando más puntos intermedios
        R[n][0] = 0.5 * R[n - 1][0] + h * sum(f(a + (2 * k - 1) * h) for k in range(1, 2**(n - 1) + 1))
        # Extrapolación de Richardson para mayor precisión
        for m in range(1, n + 1):
            R[n][m] = R[n][m - 1] + (R[n][m - 1] - R[n - 1][m - 1]) / (4**m - 1)
        print_row(R[n])
        # Criterio de convergencia
        if abs(R[n][n - 1] - R[n][n]) < eps:
            return R[n][n]
        n += 1


# Integración por cuadratura de Gauss
def IG(f, n):
    """
    Calcula la integral definida de `f` usando la cuadratura de Gauss.

    Parámetros:
        f (function): Función a integrar.
        n (int): Número de puntos de Gauss.

    Retorna:
        float: Valor aproximado de la integral.
    """
    c = GaussTable[n - 1][1]  # Pesos de Gauss
    x = GaussTable[n - 1][0]  # Puntos de evaluación de Gauss
    return sum([c[i] * f(x[i]) for i in range(n)])  # Aproximación


# Cuadratura Gaussiana tabulada
def IGtabulada(f, n):
    """
    Crea una tabla con los puntos, pesos y evaluaciones de la cuadratura de Gauss.

    Parámetros:
        f (function): Función a integrar.
        n (int): Número de puntos de Gauss.

    Retorna:
        list: Tabla con información de la cuadratura.
    """
    c = GaussTable[n - 1][1]  # Pesos de Gauss
    x = GaussTable[n - 1][0]  # Puntos de evaluación de Gauss
    return [[c[i], x[i], f(x[i]), c[i] * f(x[i])] for i in range(n)]


# Integración de Gauss en intervalos arbitrarios
def IGAL(f, n, a, b):
    """
    Calcula la integral definida de `f` en el intervalo [a, b]
    usando la cuadratura de Gauss con cambio de variable.

    Parámetros:
        f (function): Función a integrar.
        n (int): Número de puntos de Gauss.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.

    Retorna:
        float: Valor aproximado de la integral.
    """
    c = GaussTable[n - 1][1]  # Pesos de Gauss
    x = GaussTable[n - 1][0]  # Puntos de evaluación de Gauss
    return sum([(b - a) / 2 * c[i] * f((b - a) / 2 * (x[i] + 1) + a) for i in range(n)])


# Integración de funciones de dos variables usando Gauss
def IG2D(f, n, m):
    #integrate.dblquad(func, x_inferior, x_superior, y_inf, y_supr)
    """
    Calcula una integral doble de `f` usando la cuadratura de Gauss.

    Parámetros:
        f (function): Función de dos variables a integrar.
        n (int): Número de puntos de Gauss para la variable x.
        m (int): Número de puntos de Gauss para la variable y.

    Retorna:
        float: Valor aproximado de la integral.
    """
    c1 = GaussTable[n - 1][1]  # Pesos para x
    c2 = GaussTable[m - 1][1]  # Pesos para y
    x = GaussTable[n - 1][0]  # Puntos de evaluación para x
    y = GaussTable[m - 1][0]  # Puntos de evaluación para y
    return sum([c1[i] * c2[j] * f(x[i], y[j]) for i in range(n) for j in range(m)])


def IG2Dg(f, a, b, c, d, n, m):
    #integrate.dblquad(func, x_inferior, x_superior, y_inf, y_supr)
    """
    Calcula una integral doble de `f` en un dominio arbitrario usando Gauss.

    Parámetros:
        f (function): Función de dos variables a integrar.
        a, b (float): Límites del intervalo para x.
        c, d (float): Límites del intervalo para y.
        n, m (int): Número de puntos de Gauss para x e y.

    Retorna:
        float: Valor aproximado de la integral.
    """
    c1 = GaussTable[n - 1][1]  # Pesos para x
    c2 = GaussTable[m - 1][1]  # Pesos para y
    x = GaussTable[n - 1][0]  # Puntos de evaluación para x
    y = GaussTable[m - 1][0]  # Puntos de evaluación para y
    K1, K2 = (b - a) / 2, (d - c) / 2  # Escalamiento
    m1, m2 = (a + b) / 2, (c + d) / 2  # Desplazamiento
    return K1 * K2 * sum([c1[i] * c2[j] * f(K1 * x[i] + m1, K2 * y[j] + m2) for i in range(n) for j in range(m)])

def IG3Dg(f, a, b, c, d, e, f_, n, m, l):
    #integrate.tplquad(func, x_inf, x_sup, y_inf, y_sup, z_inf, z_sup)
    """
    Calcula una integral triple de `f` en un dominio arbitrario usando Gauss.

    Parámetros:
        f (function): Función de tres variables a integrar.
        a, b (float): Límites del intervalo para x.
        c, d (float): Límites del intervalo para y.
        e, f_ (float): Límites del intervalo para z.
        n, m, l (int): Número de puntos de Gauss para x, y, y z respectivamente.

    Retorna:
        float: Valor aproximado de la integral.
    """
    # Pesos y puntos para las tres variables
    c1 = GaussTable[n - 1][1]  # Pesos para x
    c2 = GaussTable[m - 1][1]  # Pesos para y
    c3 = GaussTable[l - 1][1]  # Pesos para z
    x = GaussTable[n - 1][0]  # Puntos de evaluación para x
    y = GaussTable[m - 1][0]  # Puntos de evaluación para y
    z = GaussTable[l - 1][0]  # Puntos de evaluación para z

    # Escalamiento y desplazamiento para cada dimensión
    K1, K2, K3 = (b - a) / 2, (d - c) / 2, (f_ - e) / 2  # Factores de escala
    m1, m2, m3 = (a + b) / 2, (c + d) / 2, (e + f_) / 2  # Factores de desplazamiento

    # Suma triple
    return K1 * K2 * K3 * sum(
        c1[i] * c2[j] * c3[k] * f(
            K1 * x[i] + m1,  # Cambio de variable para x
            K2 * y[j] + m2,  # Cambio de variable para y
            K3 * z[k] + m3   # Cambio de variable para z
        )
        for i in range(n) for j in range(m) for k in range(l)
    )
