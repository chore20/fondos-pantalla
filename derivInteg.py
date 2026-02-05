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

#------------------------------------------------------derivadas-------------------------------------------------------

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

def Richardson2d(phi,fun,x,y,h,n):
    v=np.zeros((n,1))
    R=np.zeros((n,n))
    for k in range(n):
        v[k]=phi(fun,x,y,h/2**k)
        R[k,0]=v[k]
    for j in range(1,n):
        for i in range(j,n):
            R[i,j]=(4**j*R[i,j-1]-R[i-1,j-1])/(4**j-1)
    return np.diag(R),R


#----------------------------------------------------integrales-------------------------------------------------------

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

h = lambda x,y: -6000*np.exp(x**4.5)*(x**2*(0.75 - 1.5*y) + x**5.5*y*(7.125*y**2
                - 10.6875*y + 3.5625) + x**4.5*y*(-8.4375*y**2 + 12.6563*y - 4.21875)
                + x**3.5*y*(2.0625*y**2 - 3.09375*y + 1.03125) + x**10*y*(3.375*y**2
                - 5.0625*y + 1.6875) + x**9*y *(-5.0625*y**2 + 7.59375*y - 2.53125)
                + x**8*y*(1.6875*y**2 - 2.53125*y + 0.84375) + x**3*(y - 0.5)
                + x*(y**3 - 1.5*y**2 + y - 0.25) + y*(-0.5*y**2 + 0.75*y - 0.25))
Lx, Ly = 1, 1
n = 64 # numero de sub intervalos de x en [0,Lx]
m = 64 # numero de sub intervalos de x en [0,Ly]
L, M = n-1, m-1
N = L * M
dx, dy = Lx/n, Ly/m
Uizq, Uder, Uinf, Usup = 0, 0, 0, 0
xi = [i*dx for i in range(n+1)] #nodos o particion de [0,Lx]
yj = [j*dy for j in range(m+1)] #nodos o particion de [0,Ly]

if n < 15: display(Eq(S('x_i'),Matrix(xi).T,evaluate=False))
if m < 15: display(Eq(S('y_j'),Matrix(yj).T,evaluate=False))
lam = dx**2/dy**2
d1 = np.ones(N-1) #diagonal superior e inferior
i1 = [L*k-1 for k in range(1,L) if L*k <= N-1]
d1[i1] = 0
d2 = -2*(1+lam)*np.ones(N) #diagonal principal
d3 = lam * np.ones(N-L)
b = np.zeros(N)
for j in range(1,m):
  for i in range(1,n):
     b[i+(j-1)*L-1] = -dx**2*h(xi[i],yj[j])
A = np.diag(d2) + np.diag(d1,-1)+np.diag(d1,1) + np.diag(d3,L) + np.diag(d3,-L)
if n < 15: display(Eq(MatMul(Matrix(A),S('x')),Matrix(b),evaluate=False))
x=al.solve(A,b)
if n < 15: display(Eq(S('x'),Matrix(x),evaluate=False))
u = np.zeros((m+1,n+1))
u[:,0]=Uinf
u[:,m]=Usup
u[0,:]=Uizq
u[n,:]=Uder
for j in range(1,m):
  for i in range(1,n):
    u[i,j] = x[i+(j-1)*L-1]
if n < 15: display(round_expr(Matrix(u.T),5))

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(xi, yj)
surf = ax.plot_surface(X, Y, u.T, cmap = plt.cm.coolwarm)
ax.contour(X, Y, u.T, 10, cmap=plt.cm.cividis, linestyles="solid", offset=-1)
ax.contour(X, Y, u.T, 10, colors="k", linestyles="solid")
ax.set_xlabel('Lado y', labelpad=5)
ax.set_ylabel('Lado x', labelpad=5)
ax.set_zlabel('Potencial Electrico U', labelpad=5)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

fig = plt.figure()
ax1 = plt.contourf(X,Y,u.T)
fig.colorbar(ax1, orientation='horizontal')
plt.show()

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

h = lambda x,y: -6000*np.exp(x**4.5)*(x**2*(0.75 - 1.5*y) + x**5.5*y*(7.125*y**2
                - 10.6875*y + 3.5625) + x**4.5*y*(-8.4375*y**2 + 12.6563*y - 4.21875)
                + x**3.5*y*(2.0625*y**2 - 3.09375*y + 1.03125) + x**10*y*(3.375*y**2
                - 5.0625*y + 1.6875) + x**9*y *(-5.0625*y**2 + 7.59375*y - 2.53125)
                + x**8*y*(1.6875*y**2 - 2.53125*y + 0.84375) + x**3*(y - 0.5)
                + x*(y**3 - 1.5*y**2 + y - 0.25) + y*(-0.5*y**2 + 0.75*y - 0.25))
Lx, Ly = 1, 1
n = 8 # numero de sub intervalos de x en [0,Lx]
m = 8 # numero de sub intervalos de x en [0,Ly]
L, M = n-1, m-1
N = L * M
dx, dy = Lx/n, Ly/m
Uizq, Uder, Uinf, Usup = 0, 0, 0, 0
xi = [i*dx for i in range(n+1)] #nodos o particion de [0,Lx]
yj = [j*dy for j in range(m+1)] #nodos o particion de [0,Ly]

if n < 15: display(Eq(S('x_i'),Matrix(xi).T,evaluate=False))
if m < 15: display(Eq(S('y_j'),Matrix(yj).T,evaluate=False))
lam = dx**2/dy**2
u = np.zeros((n+1,m+1))
u[:,0]=Uinf
u[:,m]=Usup
u[0,:]=Uizq
u[m,:]=Uder
err, tol, norm1 = 1, 10**(-8), al.norm(u)
while err > tol:
  norm0 = norm1
  for j in range(1,m):
    for i in range(1,n):
      u[i,j] = (dx**2*h(xi[i],yj[j])+lam*u[i,j-1]+u[i-1,j] + u[i+1,j] + lam*u[i,j+1])/(2*(1+lam))
  norm1 = al.norm(u)
  err = np.abs(norm1-norm0)
if n < 15: display(round_expr(Matrix(u.T),5))

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(xi, yj)
surf = ax.plot_surface(X, Y, u.T, cmap = plt.cm.coolwarm)
ax.contour(X, Y, u.T, 10, cmap=plt.cm.cividis, linestyles="solid", offset=-1)
ax.contour(X, Y, u.T, 10, colors="k", linestyles="solid")
ax.set_xlabel('Lado x', labelpad=5)
ax.set_ylabel('Lado y', labelpad=5)
ax.set_zlabel('Potencial Electrico U', labelpad=5)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

fig = plt.figure()
ax1 = plt.contourf(X,Y,u)
fig.colorbar(ax1, orientation='horizontal')
plt.show()

class IntegracionNumerica:
  method_name="Integración Numérica: Newton-Cotes"

  def __init__(self, x, y, a=0, b=2, f=0):
    self.x, self.y, self.a, self.b, self.f = np.array(x), np.array(y), a, b, f
    self.n, self.integral,self.t, self.p = len(self.x), 0, self.x, self.y
    self.h = self.x[1:]-self.x[:self.n-1]
    print("{:^10}".format(self.method_name))
    self.riemann_point='inf'

  def riemann(self):
    if self.riemann_point == 'inf':
      self.t = self.x[:self.n-1]
    elif self.riemann_point == 'med':
      self.t = (self.x[1:]+self.x[:self.n-1])/2
    elif self.riemann_point == 'sup':
      self.t = self.x[1:self.n]
    else: self.t = self.x[:self.n-1]
    self.p = self.f(self.t)
    self.integral = self.p.dot(self.h)
    print("Riemann = {}".format(self.integral))
    return self.integral

  def riemann_plot(self):
    n_suave = self.n*10
    xk = np.linspace(self.a,self.b,n_suave)
    fk = self.f(xk)
    plt.plot(xk,fk, label ='f(x)')
    plt.scatter(self.t, self.p, marker='o',
            color='orange', label = self.riemann_point)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Trapecios')
    plt.legend(loc="best")
    for i in range(self.n-1):
      plt.fill_between([self.x[i],self.x[i+1]],0,[self.p[i],self.p[i]],
                       color='g',alpha=0.9)
      plt.plot([self.x[i],self.x[i+1]],[self.p[i],self.p[i]], color='orange')
    for i in range(self.n):
      plt.axvline(self.x[i], color='w')
    plt.show()

  def trapecio(self):
    self.integral = (self.y[:self.n-1]+self.y[1:]).dot(self.h)/2
    print("Trapecio = {}".format(self.integral))
    return self.integral

  def trapecio_plot(self):
    n_suave = self.n*10
    xk = np.linspace(self.a,self.b,n_suave)
    fk = self.f(xk)
    plt.plot(xk,fk, label ='f(x)')
    plt.plot(self.x, self.y, marker='o',
            color='orange', label ='Lineal')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Trapecios')
    plt.legend(loc="best")
    plt.fill_between(self.x,0,self.y, color='b',alpha=0.9)
    for i in range(self.n):
        plt.axvline(self.x[i], color='w')
    plt.show()

  def simpson13(self):
    if self.n % 2 == 1:
      simp = [self.y[i]+4*self.y[i+1]+self.y[i+2] for i in np.arange(0,self.n-2,2)]
      self.integral = np.array(simp).dot(self.h[0:self.n-2:2])/3
      print("Simpson 1/3 = {}".format(self.integral))
    else:
      self.integral=None
      print('Para Simpson 1/3, el número de puntos debe ser impar')
    return self.integral

  def simpson13_plot(self):
    if self.n % 2 != 1:
      return None
    n_suave = self.n*10
    xk = np.linspace(self.a,self.b,n_suave)
    fk = self.f(xk)
    plt.plot(xk,fk, label ='f(x)')
    plt.scatter(self.x, self.y, marker='o',
            color='orange', label ='Parabólico')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Simpson 1/3')
    for i in np.arange(0,self.n-2,2):
      p2 = lagrange(self.x[i:i+3],self.y[i:i+3])
      x2 = np.linspace(self.x[i],self.x[i+2],30)
      y2 = p2(x2)
      plt.plot(x2,y2,color='orange')
      plt.fill_between(x2,0,y2, color='blueviolet',alpha=0.9)
    for i in range(self.n):
        plt.axvline(self.x[i], color='w')
    plt.legend(loc="best")
    plt.show()

  def simpson38(self):
    if self.n % 3 == 1:
      simp = [self.y[i]+3*self.y[i+1]+3*self.y[i+2]+self.y[i+3] for i in np.arange(0,self.n-3,3)]
      self.integral = 3*np.array(simp).dot(self.h[0:self.n-3:3])/8
      print("Simpson 3/8 = {}".format(self.integral))
    else:
      self.integral=None
      print('Para Simpson 3/8, el número de puntos debe ser multiplo de tres + 1')
    return self.integral

  def simpson38_plot(self):
    if self.n % 3 != 1:
      return None
    n_suave = self.n*10
    xk = np.linspace(self.a,self.b,n_suave)
    fk = self.f(xk)
    plt.plot(xk,fk, label ='f(x)')
    plt.scatter(self.x, self.y, marker='o',
            color='orange', label ='Cúbico')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Simpson 3/8')
    for i in np.arange(0,self.n-3,3):
      p2 = lagrange(self.x[i:i+4],self.y[i:i+4])
      x2 = np.linspace(self.x[i],self.x[i+3],30)
      y2 = p2(x2)
      plt.plot(x2,y2,color='coral')
      plt.fill_between(x2,0,y2, color='teal',alpha=0.9)
    for i in range(self.n):
        plt.axvline(self.x[i], color='w')
    plt.legend(loc="best")
    plt.show()

#------------------------------------

def IT(f, a, b, n):
  h = (b - a)/n
  return sum([(f(a + i*h) + f(a + (i + 1)*h))*h/2 for i in range(int(n))])
a = 0.0
b = 1.0
def f(x): return 2/np.sqrt(math.pi)*np.exp(-x**2)
Itrue, error = integrate.quad(f, a, b)
Ii = []
for i in range(8):
  ss = IT(f, a, b, 2**i)
  Ii.append([2**i, (b - a)/2**i, ss, Itrue, abs(Itrue - ss)])
Ii = pd.DataFrame(Ii, columns=["n", "h", "I_Trapecio", "I_exacta", "|E|"])
display(Ii)

def print_row(lst):
  print(' '.join('%11.8f' % x for x in lst))

def romberg(f, a, b, eps = 1E-8):
  R = [[0.5 * (b - a) * (f(a) + f(b))]] # R[0][0]
  print_row(R[0])
  n = 1
  while True:
    h = float(b-a)/2**n
    R.append((n+1)*[None]) # Add an empty row.
    R[n][0]=0.5*R[n-1][0]+h*sum(f(a+(2*k-1)*h) for k in range(1,2**(n-1)+1))
    for m in range(1, n+1):
      R[n][m] = R[n][m-1] + (R[n][m-1] - R[n-1][m-1]) / (4**m - 1)
    print_row(R[n])
    if abs(R[n][n-1] - R[n][n]) < eps:
      return R[n][n]
    n += 1

def IG(f, n):
  c = GaussTable[n - 1][1]
  x = GaussTable[n - 1][0]
  return sum([c[i]*f(x[i]) for i in range(n)])

def IGtabulada(f, n):
  c = GaussTable[n - 1][1]
  x = GaussTable[n - 1][0]
  tab=[[c[i], x[i], f(x[i]), c[i]*f(x[i])] for i in range(n)]
  return tab

def IG(f, n):
  c = GaussTable[n - 1][1]
  x = GaussTable[n - 1][0]
  return sum([c[i]*f(x[i]) for i in range(n)])

def f(x): return np.cos(x)
Iexact, error = integrate.quad(f, -1, 1)
print("Integral exacto: ",Iexact)
Itable = [[i + 1, sp.N(IG(f, i + 1)), (Iexact - IG(f, i + 1))/Iexact] for i in range(6)]
Itable = pd.DataFrame(Itable, columns=["Numero de puntos",  "Cuadratura Gaussiana", "Error relativo"])
display(Itable)

def IGAL(f, n, a, b):
  c = GaussTable[n - 1][1]
  x = GaussTable[n - 1][0]
  return sum([(b - a)/2*c[i]*f((b - a)/2*(x[i] + 1) + a) for i in range(n)])

def f(x): return x*np.exp(x)
Iexact, error = integrate.quad(f, 0, 3)
print("Integral exacta: ",Iexact)
Itable = [[i + 1, sp.N(IGAL(f, i + 1, 0, 3)), (Iexact - IGAL(f, i + 1, 0, 3))/Iexact] for i in range(6)]
Itable = pd.DataFrame(Itable, columns=["Numbero de Puntos",  "Integral nmérica", "Error relativo"])
display(Itable)

def IG2D(f,n,m):
  c1 = GaussTable[n - 1][1]
  c2 = GaussTable[m - 1][1]
  x = GaussTable[n - 1][0]
  y = GaussTable[m - 1][0]
  return sum([c1[i]*c2[j]*f(x[i],y[j]) for i in range(n) for j in range(m)])

def IG2Dg(f,a,b,c,d,n,m):
  c1 = GaussTable[n - 1][1]
  c2 = GaussTable[m - 1][1]
  x = GaussTable[n - 1][0]
  y = GaussTable[m - 1][0]
  K1, K2 = (b-a)/2,(d-c)/2
  m1, m2 = (a+b)/2, (c+d)/2
  return K1*K2*sum([c1[i]*c2[j]*f(K1*x[i]+m1,K2*y[j]+m2) for i in range(n) for j in range(m)])

