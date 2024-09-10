import numpy as np
import matplotlib.pyplot as plt 
import math
Y=np.loadtxt("samplesVDA1.txt")
X=np.arange(0,len(Y)*1/400,1/400)
puntos = list(zip(X, Y))
N = len(puntos) 

plt.plot(X, Y, marker='o')  # marker='o' es para mostrar los puntos, puedes omitirlo si no quieres puntos
plt.xlabel('Eje X')  # Etiqueta para el eje X
plt.ylabel('Eje Y')  # Etiqueta para el eje Y
plt.title('Gráfico de X vs Y')  # Título del gráfico
plt.grid(True)  # Mostrar una cuadrícula
plt.show()  # Mostrar el gráfico


#def ClusteringSubstractivo(puntos):
    vec_pot = np.zeros(N)
    max_pot = 0

    #hiperparametros
   # ar=
    #rr= 
    for i in range (N):
        vec_pot(i) = CalculoDePotencial(N,puntos,Ra,i)
        if (max_pot < vec_pot(i))
           max_pot = vec_pot(i)

    
def CalculoDePotencial(N,puntos,R,i):
    sum = 0
    for j in range(N):
       sum += math.e**(-(np.linalg.norm(puntos(i) - puntos(j)))**2 /((R/2)**2))
    return 