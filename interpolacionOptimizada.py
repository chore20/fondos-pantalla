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
from scipy.interpolate import CubicSpline

def InterpolacionSis(x, y):
    """
    Realiza la interpolación de un conjunto de puntos utilizando un sistema de ecuaciones lineales.
    
    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    y : list
        Lista de coordenadas y de los puntos.
    
    Retorna:
    np.ndarray
        Un vector de coeficientes del polinomio interpolador.
    """
    if len(x) == len(y):
        n = len(x)
        b = np.array(y).reshape((n, 1))
        x = np.array(x).reshape((n, 1))
        A = np.array(x**(n-1))
        for i in range(1, n):
            A = np.concatenate([A, x**(n-1-i)], axis=1)
        print(A)
        p = np.linalg.solve(A, b)
        return p.reshape(n)
    else:
        print('No hay la misma cantidad de puntos')


def li(x, k):
    """
    Calcula el k-ésimo polinomio base de Lagrange.
    
    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    k : int
        Índice del polinomio base a calcular.
    
    Retorna:
    sympy expression
        Polinomio base de Lagrange correspondiente al índice k.
    """
    t = sp.symbols('t')
    n = len(x)
    x1 = np.array(x)
    x1 = np.delete(x1, k)
    p = 1
    for i in range(len(x1)):
        p = p * (t - x1[i]) / (x[k] - x1[i])
    return sp.expand(p)


def Lagrange(x, y):
    """
    Calcula el polinomio de interpolación de Lagrange para un conjunto de puntos.
    
    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    y : list
        Lista de coordenadas y de los puntos.
    
    Retorna:
    sympy expression
        Polinomio interpolador de Lagrange.
    """
    n = len(x)
    p = 0
    for i in range(n):
        lk = li(x, i)
        p = p + y[i] * lk
    return sp.expand(p)


def Pol2Vec(p):
    """
    Convierte un polinomio a un vector de coeficientes.
    
    Parámetros:
    p : sympy expression
        Polinomio.
    
    Retorna:
    np.ndarray
        Vector de coeficientes del polinomio.
    """
    n = sp.degree(p)
    v = np.zeros(n + 1)
    t = sp.symbols('t')
    k = len(p.args)
    for i in range(n + 1):
        for j in range(k):
            temp = p.args[j] / t**i
            if temp.is_constant():
                v[n - i] = p.args[j] / t**i
    return v


def Vec2Pol(v):
    """
    Convierte un vector de coeficientes a un polinomio.
    
    Parámetros:
    v : np.ndarray
        Vector de coeficientes.
    
    Retorna:
    sympy expression
        Polinomio correspondiente al vector de coeficientes.
    """
    n = len(v)
    t = sp.symbols('t')
    p = 0
    for i in range(n):
        p = p + v[i] * t**(n - 1 - i)
    return p


def Polyval(v, x):
    """
    Evalúa un polinomio en un punto específico.
    
    Parámetros:
    v : np.ndarray
        Vector de coeficientes del polinomio.
    x : float
        Valor en el cual se evalúa el polinomio.
    
    Retorna:
    float
        Valor del polinomio evaluado en x.
    """
    n = len(v)
    s = 0
    for i in range(n):
        s = s + v[i] * x**(n - 1 - i)
    return s


def DiferenciasDivididas(x, y):
    """
    Calcula la tabla de diferencias divididas.
    
    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    y : list
        Lista de coordenadas y de los puntos.
    
    Retorna:
    tuple
        Diagonal de coeficientes y la tabla de diferencias divididas.
    """
    n = len(x)
    A = np.zeros((n, n))
    A[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            A[i, j] = (A[i, j - 1] - A[i - 1, j - 1]) / (x[i] - x[i - j])
    return np.diag(A), A


def Newton(x, y):
    """
    Calcula el polinomio de interpolación de Newton utilizando diferencias divididas.
    
    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    y : list
        Lista de coordenadas y de los puntos.
    
    Retorna:
    sympy expression
        Polinomio de interpolación de Newton.
    """
    n = len(x)
    c, _ = DiferenciasDivididas(x, y)
    t = sp.symbols('t')
    p = c[n - 1]
    for i in range(1, n):
        p = p * (t - x[n - 1 - i]) + c[n - 1 - i]
    return sp.expand(p)


def PuntosCheby(a, b, n):
    """
    Genera puntos de Chebyshev en el intervalo [a, b].
    
    Parámetros:
    a : float
        Límite inferior del intervalo.
    b : float
        Límite superior del intervalo.
    n : int
        Número de puntos.
    
    Retorna:
    np.ndarray
        Array de puntos de Chebyshev.
    """
    x = []
    for i in range(n):
        x.append((a + b) / 2 + (b - a) / 2 * np.cos((2 * i + 1) / 2 / n * np.pi))
    return np.array(x)


def EstimacionApriori(a, b, n, fn1, tipo):
    """
    Estima el error de la interpolación.
    
    Parámetros:
    a : float
        Límite inferior del intervalo.
    b : float
        Límite superior del intervalo.
    n : int
        Grado del polinomio.
    fn1 : float
        Valor de la derivada de orden n+1 en el intervalo.
    tipo : str
        Tipo de estimación ('C' para Chebyshev, 'E' para error en intervalos equidistantes).
    
    Retorna:
    float
        Estimación del error.
    """
    if tipo == 'C':
        return np.abs(fn1) / ma.factorial(n + 1) / 2**n * (b - a)
    elif tipo == 'E':
        h = (b - a) / (n - 1)
        return np.abs(fn1 * h**(n + 1)) / 4 / (n + 1)
    else:
        print('Error: Comando desconocido')


class Hermite:
    """
    Implementa la interpolación de Hermite.
    
    Atributos:
    xi : list
        Lista de coordenadas x donde se interpolan los puntos.
    yi : list
        Lista de coordenadas y donde se interpolan los puntos.
    dyi : list
        Lista de derivadas en los puntos de interpolación.
    ph : sympy expression
        Polinomio de Hermite resultante.
    mostrar_en_fracciones : int
        Indica si se deben mostrar los resultados en fracciones (1=si, 0=no).
    """

    method_name = 'Interpolación de Hermite'
    x = sp.symbols('x')

    def __init__(self, xi, yi, dyi):
        self.xi, self.yi, self.dyi = xi, yi, dyi
        self.ph = 0
        self.mostrar_en_fracciones = 1  # 1=si, 0=no

    def dl(self, i):
        """
        Calcula el valor de la derivada del polinomio de Lagrange en el índice i.
        
        Parámetros:
        i : int
            Índice del punto.
        
        Retorna:
        float
            Valor de la derivada.
        """
        if self.mostrar_en_fracciones:
            result = 0
        else:
            result = 0.0
        for j in range(0, len(self.xi)):
            if j != i:
                result += 1 / (self.xi[i] - self.xi[j])
        return result

    def l(self, i):
        """
        Calcula el polinomio base de Lagrange para el índice i.
        
        Parámetros:
        i : int
            Índice del polinomio base.
        
        Retorna:
        sympy expression
            Polinomio base de Lagrange correspondiente al índice i.
        """
        x = sp.symbols('x')
        if self.mostrar_en_fracciones:
            deno, nume = 1, 1
        else:
            deno, nume = 1.0, 1.0
        for j in range(len(self.xi)):
            if j != i:
                deno *= (self.xi[i] - self.xi[j])
                if self.mostrar_en_fracciones:
                    sx = Fraction(Decimal(self.xi[j])).limit_denominator(1000)
                else:
                    sx = self.xi[j]
                nume *= (x - sx)
        if self.mostrar_en_fracciones:
            denom = Fraction(Decimal(deno)).limit_denominator(1000)
        else:
            denom = deno
        return nume / denom

    def pol_hermite(self):
        """
        Calcula el polinomio de Hermite utilizando los puntos y derivadas proporcionados.
        
        Retorna:
        sympy expression
            Polinomio de Hermite resultante.
        """
        x = sp.symbols('x')
        if self.mostrar_en_fracciones:
            self.ph = 0
        else:
            self.ph = 0.0
        for i in range(len(self.xi)):
            if self.mostrar_en_fracciones:
                sy = Fraction(Decimal(self.yi[i])).limit_denominator(1000)
                sx = Fraction(Decimal(self.xi[i])).limit_denominator(1000)
                cd = Fraction(Decimal(self.dyi[i] - 2 * self.yi[i] * self.dl(i))).limit_denominator(1000)
            else:
                sy, sx, cd = self.yi[i], self.xi[i], self.dyi[i] - 2 * self.yi[i] * self.dl(i)
            self.ph += (sy + (x - sx) * (cd)) * ((self.l(i))**2)
        return self.ph

    def hermite_plot(self):
        """
        Grafica el polinomio de Hermite junto con los puntos de datos.
        """
        x = sp.symbols('x')
        xval = np.linspace(min(self.xi), max(self.xi), 100)
        pval = [self.ph.subs(x, i) for i in xval]
        plt.plot(self.xi, self.yi, 'bo')
        plt.plot(xval, pval)
        plt.legend(['Datos', 'Hermite'], loc='best')

class HermiteOTP:
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

class HermiteDiferenciasDivididasNewton:
    """
    Implementa la interpolación de Hermite utilizando diferencias divididas.
    
    Atributos:
    xi : list
        Lista de coordenadas x donde se interpolan los puntos.
    yi : list
        Lista de coordenadas y donde se interpolan los puntos.
    yp : list
        Lista de derivadas en los puntos de interpolación.
    coef : list
        Coeficientes del polinomio interpolador.
    dif : np.ndarray
        Tabla de diferencias divididas.
    mostrar_en_fracciones : int
        Indica si se deben mostrar los resultados en fracciones (1=si, 0=no).
    """

    method_name = 'Interpolación de Diferencias Divididas'
    x = sp.symbols('x')

    def __init__(self, xi, yi, yp):
        self.xi, self.yi, self.yp = xi, yi, yp
        self.coef, self.pval, self.p, self.pn = [], [], 0, 0
        x1 = np.array([[i] * 2 for i in xi])
        self.z = []
        for sublist in x1:
            for item in sublist:
                self.z.append(item)
        y1 = np.array([[i] * 2 for i in yi])
        self.fz = []
        for sublist in y1:
            for item in sublist:
                self.fz.append(item)

        n = len(self.z)
        self.dif = np.zeros([n, n])
        self.mostrar_en_fracciones = 1  # 1=si, 0=no

    def divided_diff(self):
        """
        Calcula la tabla de diferencias divididas.
        
        Retorna:
        np.ndarray
            Tabla de diferencias divididas.
        """
        n = len(self.fz)
        self.dif[:, 0] = self.fz
        for j in range(1, n):
            for i in range(n - j):
                if (j == 1) & (i % 2 == 0):
                    self.dif[i][j] = self.yp[i // 2]
                else:
                    self.dif[i][j] = (self.dif[i + 1][j - 1] - self.dif[i][j - 1]) / (self.z[i + j] - self.z[i])
        return self.dif

    def print_diff_table(self):
        """
        Imprime la tabla de diferencias divididas en formato legible.
        """
        n = len(self.z) - 1
        print("{:^40}".format(self.method_name))
        df = pd.DataFrame(self.dif, index=self.z, columns=['dif' + str(i) for i in range(n + 1)])
        if self.mostrar_en_fracciones:
            display(df.applymap(lambda x: Fraction(Decimal(x)).limit_denominator(1000) if x < math.inf else x))
        else:
            display(df)

    def factores_pol(self, k):
        """
        Calcula el producto de factores para el polinomio de Newton.
        
        Parámetros:
        k : int
            Índice del polinomio.
        
        Retorna:
        sympy expression
            Producto de factores para el polinomio de Newton.
        """
        x = sp.symbols('x')
        pk = np.prod([(x - self.z[i]) for i in range(k)])
        return pk

    def newton_polinomio(self):
        """
        Calcula el polinomio de Newton utilizando la tabla de diferencias divididas.
        
        Retorna:
        sympy expression
            Polinomio de Newton.
        """
        x = sp.symbols('x')
        n = len(self.z) - 1
        self.coef = self.dif[0, :]
        if self.mostrar_en_fracciones:
            self.pn = self.coef[0] + np.sum([Fraction(Decimal(self.coef[k])).limit_denominator(1000) * self.factores_pol(k) for k in range(1, n + 1)])
        else:
            self.pn = self.coef[0] + np.sum([self.coef[k] * self.factores_pol(k) for k in range(1, n + 1)])
        display(self.pn, sp.expand(self.pn))
        return self.pn

    def newton_poly(self):
        """
        Calcula el polinomio de Newton de forma recursiva y lo muestra.
        
        Retorna:
        sympy expression
            Polinomio de Newton.
        """
        x = sp.symbols('x')
        n = len(self.z) - 1
        self.coef = self.dif[0, :]
        if self.mostrar_en_fracciones:
            self.p = Fraction(Decimal(self.coef[n])).limit_denominator(1000)
            for k in range(1, n + 1):
                self.p = Fraction(Decimal(self.coef[n - k])).limit_denominator(1000) + (x - Fraction(self.z[n - k]).limit_denominator(1000)) * self.p
        else:
            self.p = self.coef[n]
            for k in range(1, n + 1):
                self.p = self.coef[n - k] + (x - self.z[n - k]) * self.p
        print("Polinomio de Newton recursivo")
        display(self.p)
        print("Polinomio de Newton desarrollada")
        display(sp.expand(self.p))
        return self.p

    def difdiv_plot(self):
        """
        Grafica el polinomio de Newton junto con los puntos de datos.
        """
        x = sp.symbols('x')
        xval = np.linspace(min(self.z), max(self.z), 100)
        pval = [self.p.subs(x, i) for i in xval]
        plt.plot(self.z, self.fz, 'bo')
        plt.plot(xval, pval)
        plt.legend(['Datos', 'Diferencias Divididas'], loc='best')

def interpolacionLagrange(x,y,t): 
    '''
    x es una lista 
    y es una lista
    t es el valor a interpolar
    '''

    p_lagrange = Lagrange(x, y)
    print("Polinomio de interpolación de Lagrange:\n", p_lagrange)

    # Interpolación en x = 0
    t = sp.Symbol('t')
    interpolacion_x0 = p_lagrange.subs(xi, t)
    print("\nValor interpolado en x = 0:", interpolacion_x0)
    return interpolacion_x0

def interpolacionNewtonGrafica(x,y,t):
    
    x_data = x
    y_data = y
    def evaluar_newton(x, x_data, coef):
        n = len(x_data)
        resultado = coef[0]
        for i in range(1, n):
            producto = coef[i]
            for j in range(i):
                producto *= (x - x_data[j])
            resultado += producto
        return resultado

    x_data = x
    y_data = y
    
    coef, matriz_dif_divididas = DiferenciasDivididas(x_data, y_data)

    # Mostrar los coeficientes
    print("Coeficientes del polinomio de Newton:", coef)

    # Interpolar en x = 1.5
    x_interp = t
    y_interp = evaluar_newton(x_interp, x_data, coef)
    print(f"El valor interpolado en x = {x_interp} es: {y_interp}")

    # Graficar
    x_plot = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
    y_plot = [evaluar_newton(x, x_data, coef) for x in x_plot]

    plt.plot(x_data, y_data, 'ro', label='Datos')
    plt.plot(x_plot, y_plot, 'b-', label='Polinomio de Newton')
    plt.scatter(x_interp, y_interp, color='green', label=f'Interpolación (x={x_interp})')
    plt.axvline(x_interp, color='green', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Newton')
    plt.grid()
    plt.show()

def spline_cubico_natural(x, y, x_interp):
    """
    Encuentra el polinomio de interpolación cúbico natural (Spline cúbico)
    y realiza la interpolación en un valor dado.

    Parámetros:
    x : list
        Lista de coordenadas x de los puntos.
    y : list
        Lista de coordenadas y de los puntos.
    x_interp : float
        Valor en el cual se desea realizar la interpolación.

    Retorna:
    float
        Valor interpolado en x_interp.
    """
    # Construir el spline cúbico natural
    cs = CubicSpline(x, y, bc_type='natural')

    # Interpolación en el valor deseado
    y_interp = cs(x_interp)

    # Graficar el spline cúbico y los puntos de datos
    x_vals = np.linspace(min(x), max(x), 100)  # Valores para la gráfica
    y_vals = cs(x_vals)  # Evaluar el spline en estos puntos

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o', label='Datos', markersize=8)  # Puntos originales
    plt.plot(x_vals, y_vals, label='Spline cúbico', color='red')  # Spline cúbico
    plt.axvline(x=x_interp, color='green', linestyle='--', label=f'x = {x_interp}')
    plt.scatter(x_interp, y_interp, color='purple', label=f'Interpolación: ({x_interp}, {y_interp:.3f})', zorder=5)
    plt.title('Interpolación cúbica (Spline Natural)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    print('el valor interpolado es de: ', y_interp)

    return y_interp

def interpolacion_hermite(xi, yi, dyi, x_interp):
    """
    Realiza la interpolación de Hermite para los datos proporcionados y evalúa
    el polinomio en un punto dado.

    Parámetros:
    xi : list
        Lista de coordenadas x de los puntos.
    yi : list
        Lista de coordenadas y de los puntos.
    dyi : list
        Lista de derivadas en los puntos.
    x_interp : float
        Valor en el cual se desea realizar la interpolación.

    Retorna:
    tuple
        El polinomio de Hermite y el valor interpolado en x_interp.
    """
    x = sp.symbols('x')

    def dl(i):
        """Calcula la derivada del polinomio base de Lagrange en el índice i."""
        result = 0
        for j in range(len(xi)):
            if j != i:
                result += 1 / (xi[i] - xi[j])
        return result

    def l(i):
        """Calcula el polinomio base de Lagrange para el índice i."""
        nume, deno = 1, 1
        for j in range(len(xi)):
            if j != i:
                deno *= (xi[i] - xi[j])
                nume *= (x - xi[j])
        return nume / deno

    # Calcular el polinomio de Hermite
    ph = 0
    for i in range(len(xi)):
        L_i = l(i)
        deriv = dyi[i] - 2 * yi[i] * dl(i)
        ph += (yi[i] + (x - xi[i]) * deriv) * (L_i**2)
    ph = sp.expand(ph)

    # Evaluar el polinomio en x_interp
    y_interp = ph.subs(x, x_interp)

    # Graficar el polinomio y los puntos
    x_vals = np.linspace(min(xi) - 1, max(xi) + 1, 100)
    y_vals = [ph.subs(x, val) for val in x_vals]
    plt.figure(figsize=(8, 6))
    plt.plot(xi, yi, 'bo', label='Datos')
    plt.plot(x_vals, y_vals, 'r-', label='Polinomio de Hermite')
    plt.axvline(x=x_interp, color='green', linestyle='--', label=f'x = {x_interp}')
    plt.scatter(x_interp, y_interp, color='purple', label=f'Interpolación: ({x_interp}, {y_interp:.3f})', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Hermite')
    plt.legend()
    plt.grid()
    plt.show()

    return ph, y_interp

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

    if grafica == True: # Graficar
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