import matplotlib.pyplot as plt
import numpy as np

def grafica(x,y):
    plt.figure(figsize = (5,5))
    plt.plot(x,y,'r-',label ="tabla")
    plt.plot(x,y,'bo')
    plt.xlabel = 'x'
    plt.ylabel = 'y'
    plt.grid()
    plt.show()

x = np.array([0.33, 0.83,0.83, 1.03])
y = np.array([1.456, 2.996, 2.193, 2.487])

grafica(x,y)

