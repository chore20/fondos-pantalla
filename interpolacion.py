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

def InterpolacionSis(x,y):
    if len(x)==len(y):
        n=len(x)
        b=np.array(y).reshape((n,1))
        x=np.array(x).reshape((n,1))
        A=np.array(x**(n-1))
        for i in range(1,n):
            A=np.concatenate([A,x**(n-1-i)],axis=1)
        print(A)
        p=np.linalg.solve(A,b)
        return p.reshape(n)
    else:
        print('No hay la misma cantidad de puntos')


def li(x,k):
    t=sp.symbols('t')
    n=len(x)
    x1=np.array(x)
    x1=np.delete(x1,k)
    p=1
    for i in range(len(x1)):
        p=p*(t-x1[i])/(x[k]-x1[i])
    return sp.expand(p)

def Lagrange(x,y):
    n=len(x)
    p=0
    for i in range(n):
        lk=li(x,i)
        p=p+y[i]*lk
    return sp.expand(p)

def Pol2Vec(p):
    n=sp.degree(p)
    v=np.zeros(n+1)
    t=sp.symbols('t')  #print(p.args)
    k=len(p.args)
    for i in range(n+1):
        for j in range(k):
            temp=p.args[j]/t**i
            if temp.is_constant():
                v[n-i]=p.args[j]/t**i
    return v

def Vec2Pol(v):
    n=len(v)
    t=sp.symbols('t')
    p=0
    for i in range(n):
        p=p+v[i]*t**(n-1-i)
    return p

def Polyval(v,x):
    n=len(v)
    s=0
    for i in range(n):
        s=s+v[i]*x**(n-1-i)
    return s

def DiferenciasDivididas(x,y):
    n=len(x)
    A=np.zeros((n,n))
    A[:,0]=y
    for j in range(1,n):
        for i in range(j,n):
            A[i,j]=(A[i,j-1]-A[i-1,j-1])/(x[i]-x[i-j])
    return np.diag(A),A

def Newton(x,y):
    n=len(x)
    c,_=DiferenciasDivididas(x,y)
    t=sp.symbols('t')
    p=c[n-1]
    for i in range(1,n):
        p=p*(t-x[n-1-i])+c[n-1-i]
    return sp.expand(p)

def  PuntosCheby(a,b,n):
    x=[]
    for i in range(n):
        x.append((a+b)/2+(b-a)/2*np.cos((2*i+1)/2/n*np.pi))
    return np.array(x)

def EstimacionApriori(a,b,n,fn1,tipo):
    if tipo=='C':
        return np.abs(fn1)/ma.factorial(n+1)/2**n*(b-a)
    elif tipo=='E':
        h=(b-a)/(n-1)
        return np.abs(fn1*h**(n+1))/4/(n+1)
    else:
        print('Error: Comando desconocido')


class Hermite:
  method_name = 'Interpolación de Hermite'
  x = sp.symbols('x')

  def __init__(self, xi, yi, dyi):
    self.xi, self.yi, self.dyi = xi, yi, dyi
    self.ph = 0
    self.mostrar_en_fracciones = 1 #1=si, 0 no

  def dl(self,i):
    if self.mostrar_en_fracciones: result = 0
    else: result = 0.0
    for j in range(0,len(self.xi)):
        if j!=i:
            result += 1/(self.xi[i]-self.xi[j])
    return result

  def l(self,i):
    x = sp.symbols('x')
    if self.mostrar_en_fracciones: deno, nume = 1, 1
    else: deno, nume = 1.0, 1.0
    for j in range(len(self.xi)):
      if j!= i:
        deno *= (self.xi[i]-self.xi[j])
        if self.mostrar_en_fracciones:
          sx = Fraction(Decimal(self.xi[j])).limit_denominator(1000)
        else: sx = self.xi[j]
        nume *= (x-sx)
    if self.mostrar_en_fracciones:
      denom = Fraction(Decimal(deno)).limit_denominator(1000)
    else: denom = deno
    return nume/denom

  def pol_hermite(self):
    x = sp.symbols('x')
    if self.mostrar_en_fracciones: self.ph = 0
    else: self.ph = 0.0
    for i in range(len(self.xi)):
      if self.mostrar_en_fracciones:
        sy=Fraction(Decimal(self.yi[i])).limit_denominator(1000)
        sx=Fraction(Decimal(self.xi[i])).limit_denominator(1000)
        cd=Fraction(Decimal(self.dyi[i]-2*self.yi[i]*self.dl(i))).limit_denominator(1000)
      else:
        sy, sx, cd = self.yi[i], self.xi[i], self.dyi[i]-2*self.yi[i]*self.dl(i)
      self.ph += (sy+(x-sx)*(cd))*((self.l(i))**2)
    return self.ph

  def hermite_plot(self):
    x = sp.symbols('x')
    xval = np.linspace(min(self.xi),max(self.xi), 100)
    pval = [self.ph.subs(x,i) for i in xval]
    plt.plot(self.xi, self.yi, 'bo')
    plt.plot(xval, pval)
    plt.legend(['Datos', 'Hermite'], loc='best')

class HermiteDiferenciasDivididasNewton:
  method_name = 'Interpolación de Diferencias Divididas'
  x = sp.symbols('x')

  def __init__(self, xi, yi, yp):
    self.xi, self.yi, self.yp = xi, yi, yp
    self.coef, self.pval, self.p, self.pn = [], [], 0, 0
    x1=np.array([[i]*2 for i in xi])
    self.z = []
    for sublist in x1:
        for item in sublist:
            self.z.append(item)
    y1=np.array([[i]*2 for i in yi])
    self.fz = []
    for sublist in y1:
        for item in sublist:
            self.fz.append(item)

    n = len(self.z)
    self.dif = np.zeros([n, n])
    self.mostrar_en_fracciones = 1 #1=si, 0 no

  def divided_diff(self):
    n = len(self.fz)
    self.dif[:,0] = self.fz
    for j in range(1,n):
      for i in range(n-j):
        if (j == 1) & (i % 2 == 0): self.dif[i][j] = self.yp[i // 2];
        else:
          self.dif[i][j] = (self.dif[i+1][j-1] - self.dif[i][j-1])/(self.z[i+j]-self.z[i])
    return self.dif

  def print_diff_table(self):
    n = len(self.z) - 1
    print("{:^40}".format(self.method_name))
    df = pd.DataFrame(self.dif,index=self.z,columns=['dif'+str(i) for i in range(n+1)])
    if self.mostrar_en_fracciones:
      display(df.applymap(lambda x:Fraction(Decimal(x))
      .limit_denominator(1000) if x < math.inf else x))
    else:
      display(df) #.apply(lambda x: np.round(x, 2))

  def factores_pol(self,k):
    x = sp.symbols('x')
    pk = np.prod([(x-self.z[i]) for i in range(k)])
    return pk

  def newton_polinomio(self):
    x = sp.symbols('x')
    n = len(self.z) - 1
    self.coef = self.dif[0,:]
    if self.mostrar_en_fracciones:
      self.pn = self.coef[0] + np.sum([Fraction(Decimal(self.coef[k]))
      .limit_denominator(1000)*self.factores_pol(k) for k in range(1,n+1)])
    else:
      self.pn = self.coef[0] + np.sum([self.coef[k]*self.factores_pol(k) for k in range(1,n+1)])
    display(self.pn,sp.expand(self.pn))
    return self.pn

  def newton_poly(self):
    x = sp.symbols('x')
    n = len(self.z) - 1
    self.coef = self.dif[0,:]
    if self.mostrar_en_fracciones:
      self.p = Fraction(Decimal(self.coef[n])).limit_denominator(1000)
      for k in range(1,n+1):
          self.p = Fraction(Decimal(self.coef[n-k])).limit_denominator(1000) + (x - Fraction(self.z[n-k]).limit_denominator(1000))*self.p
    else:
      self.p = self.coef[n]
      for k in range(1,n+1):
          self.p = self.coef[n-k] + (x - self.z[n-k])*self.p
    print("Polinomio de Newton recursivo")
    display(self.p)
    print("Polinomio de Newton desarrollada")
    display(sp.expand(self.p))
    return self.p

  def difdiv_plot(self):
    x = sp.symbols('x')
    xval = np.linspace(min(self.z),max(self.z), 100)
    pval = [self.p.subs(x,i) for i in xval]
    plt.plot(self.z, self.fz, 'bo')
    plt.plot(xval, pval)
    plt.legend(['Datos', 'Diferencias Divididas'], loc='best')