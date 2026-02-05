import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Math, Latex, Markdown
sp.init_printing()  


def Euler(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        U[i+1]=U[i]+h*f(U[i],T[i])
    return T,U

def TaylorExp(ua,a,b,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        U[i+1]=U[i]+h*U[i]+h**2/2*U[i]
    return T,U 

def RungeKutta01O2(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[i],T[i])
        K2=h*f(U[i]+K1/2,T[i]+h/2)
        U[i+1]=U[i]+K2
    return T,U

def RungeKutta1_2O2(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[i],T[i])
        K2=h*f(U[i]+K1,T[i]+h)
        U[i+1]=U[i]+K1/2+K2/2
    return T,U 

def Euler(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        U[i+1]=U[i]+h*f(U[i],T[i])
    return T,U

def EULER(f,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    m=len(Ua)
    U=np.zeros((m,n+1))
    for k in range(m):
        U[k,0]=Ua[k]
    h=T[1]-T[0]
    for i in range(n):
        for k in range(m):
            U[k,i+1]=U[k,i]+h*f(U[:,i],T[i])[k]
    return T,U

def RungeKutta01O2(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[i],T[i])
        K2=h*f(U[i]+K1/2,T[i]+h/2)
        U[i+1]=U[i]+K2
    return T,U 

def RUNGEKUTTA01O2(f,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    m=len(Ua)
    U=np.zeros((m,n+1))
    for k in range(m):
        U[k,0]=Ua[k]
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[:,i],T[i])
        K2=h*f(np.array(U[:,i]).reshape((m,1))+np.array(K1)/2,T[i]+h/2)
        for k in range(m):
            U[k,i+1]=U[k,i]+K2[k]
    return T,U

def RUNGEKUTTA12O2(f,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    m=len(Ua)
    U=np.zeros((m,n+1))
    for k in range(m):
        U[k,0]=Ua[k]
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[:,i],T[i])
        K2=h*f(np.array(U[:,i]).reshape((m,1))+np.array(K1),T[i]+h)
        for k in range(m):
            U[k,i+1]=U[k,i]+1/2*K1[k]+1/2*K2[k]
    return T,U

def RungeKutta4(f,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    U=np.zeros(len(T))
    U[0]=ua
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[i],T[i])
        K2=h*f(U[i]+K1/2,T[i]+h/2)
        K3=h*f(U[i]+K2/2,T[i]+h/2)
        K4=h*f(U[i]+K3,T[i]+h)
        U[i+1]=U[i]+1/6*(K1+2*K2+2*K3+K4)
    return T,U

def RUNGEKUTTA4(f,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    m=len(Ua)
    U=np.zeros((m,n+1))
    for k in range(m):
        U[k,0]=Ua[k]
    h=T[1]-T[0]
    for i in range(n):
        K1=h*f(U[:,i],T[i])
        K2=h*f(np.array(U[:,i]).reshape((m,1))+np.array(K1)/2,T[i]+h/2)
        K3=h*f(np.array(U[:,i]).reshape((m,1))+np.array(K2)/2,T[i]+h/2)
        K4=h*f(np.array(U[:,i]).reshape((m,1))+np.array(K3),T[i]+h)
        for k in range(m):
            U[k,i+1]=U[k,i]+1/6*(K1[k]+2*K2[k]+2*K3[k]+K4[k])
    return T,U

def Multipaso2(f,fRK,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    h=(b-a)/n
    X=np.zeros(n+1)
    X[0]=Ua
    _,aux=fRK(f,T[0],T[1],Ua,1)#aqui cambiamos para el numero de pasos
    X[1]=aux[-1]
    for i in range(2,n+1):
        X[i]=X[i-2]+2*h*f(X[i-1],T[i-1])
    return T,X

def AdamBashford2(f,fRK,a,b,ua,n):
    T=np.linspace(a,b,n+1)
    h=(b-a)/n
    X=np.zeros(n+1)
    X[0]=ua
    _,aux=fRK(f,T[0],T[1],ua,1)#aqui cambiamos para el numero de pasos
    X[1]=aux[-1]
    for i in range(2,n+1):
        X[i]=X[i-1]+3/2*h*f(X[i-1],T[i-1])-1/2*h*f(X[i-2],T[i-2])
    return T,X

def AdamBashford3(f,fRK,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    h=(b-a)/n
    X=np.zeros(n+1)
    X[0]=Ua
    _,aux=fRK(f,T[0],T[2],Ua,2)#aqui cambiamos para el numero de pasos
    X[1]=aux[1]
    X[2]=aux[2]
    for i in range(3,n+1):
        X[i]=X[i-1]+23/12*h*f(X[i-1],T[i-1])-16/12*h*f(X[i-2],T[i-2])+5/12*h*f(X[i-3],T[i-3])
    return T,X

def AdamBashford4(f,fRK,a,b,Ua,n):
    T=np.linspace(a,b,n+1)
    h=(b-a)/n
    X=np.zeros(n+1)
    X[0]=Ua
    _,aux=fRK(f,T[0],T[3],Ua,3)#aqui cambiamos para el numero de pasos
    X[1]=aux[-3]
    X[2]=aux[-2]
    X[3]=aux[-1]
    for i in range(4,n+1):
        X[i]=X[i-1]+55/24*h*f(X[i-1],T[i-1])-59/24*h*f(X[i-2],T[i-2])+37/24*h*f(X[i-3],T[i-3])-9/24*h*f(X[i-4],T[i-4])
    return T,X

