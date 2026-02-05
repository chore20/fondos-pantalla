import numpy as np 

def raiz(x):
    y = np.sqrt(x)
    print(y)
    return y 

def limites_adelante_atras(z):
    y=np.sqrt(z)
    x=z
    yaprox=round(np.sqrt(z),2)
    xaprox=round(np.sqrt(z),2)**2
    EAF=np.abs(y-yaprox)
    EAB=np.abs(x-xaprox)
    print(EAF,EAB) 

numero = float(input("ingrese un numero para sacar la raiz y los limites: " ))

raiz(numero)
limites_adelante_atras(numero)