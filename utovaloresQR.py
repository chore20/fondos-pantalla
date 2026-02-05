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



def Householder(A):
    n=len(A)
    z=A[:,0]
    alfa=-np.sign(z[0])*np.linalg.norm(z)
    e1=np.zeros((n,1))
    e1[0]=1
    v = alfa*e1
    u=z.reshape((n,1))-v
    u=u/np.linalg.norm(u)
    H=np.eye(n)-2*u.dot(np.transpose(u))
    return H


import numpy as np
def descQR(A):
    n=A.shape[0]
    R=A
    Q=np.eye(n)
    for k in range(n-1):
        Ak=R[k:,k:]
        Hk=Householder(Ak)
        #print(Hk)
        Qk=np.eye(n)
        Qk[k:,k:]=Hk
        Q=Qk.dot(Q)
        R=Qk.dot(R)
    return np.transpose(Q),R

def AlgQR(A,tol,maxiter):
    iter=0
    while (np.linalg.norm(np.tril(A,-1))>tol)&(iter<maxiter):
        Q,R=descQR(A)
        A=R.dot(Q)
        iter=iter+1
    return np.diag(A),A,iter

