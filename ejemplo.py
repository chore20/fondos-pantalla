import numpy as np 

def raiz(x):
    y = np.sqrt(x)
    #print(y)
    return y 

def limites_adelante_atras(z):
    y=np.sqrt(z)
    x=z
    yaprox=1.73
    xaprox=1.73**2
    EAF=np.abs(y-yaprox)
    EAB=np.abs(x-xaprox)
    print(EAF,EAB) 

limites_adelante_atras(3)