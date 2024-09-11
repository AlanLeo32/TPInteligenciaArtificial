import numpy as np
import matplotlib.pyplot as plt 
import math
from sklearn.preprocessing import MinMaxScaler

def ClusteringSubstractivo(puntos,centros_clusters,N):
    vec_pot = np.zeros(N)
    max_pot = 0
    c=0
    #hiperparametros
    Ra=0.6
    Rb=0.9
    RR=0.5 #coeficiente de rechazo. Tiene que ser menor que AR? 
    AR=0.7 #coeficiente de aceptacion
    k=0
    for i in range (N):
        for j in range(N):
            dist=calcular_distancia(puntos[i],puntos[j])
            vec_pot[i]+=np.exp(-dist**2/(Ra/2)**2)
    k=np.argmax(vec_pot) # posicion del punto de maximo potencial
    centros_clusters.append(puntos[k]) 
    
    while(np.max(vec_pot) >= 1e-2):
        Pk=np.max(vec_pot)
        recalcular_potencial(puntos,vec_pot,k,Rb)
        max_pot=np.max(vec_pot)
        k=np.argmax(vec_pot)
        if(max_pot>AR*Pk):
            centros_clusters.append(puntos[k])
        else:
            if(max_pot<RR*Pk):
                break
            else:
                Dr=distancia_minima(puntos[k],centros_clusters)
                if((Dr/Ra)+(Pk/max_pot)>=1):
                    centros_clusters.append(puntos[k])
                else:
                    vec_pot[k]=0
        
def calcular_distancia(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def recalcular_potencial(puntos, vec_pot,c, Rb):
    for i in range(len(puntos)):
        dist = calcular_distancia(puntos[i],puntos[c])
        vec_pot[i] -= vec_pot[c]*np.exp(-(dist**2) / (Rb/2)**2)
        if vec_pot[i]<0:
            vec_pot[i]=0
    return vec_pot
def distancia_minima(punto,puntos):
# Encontrar la distancia mínima
    min= float('inf')  # Inicializamos con infinito

    for p in puntos:
        distancia = calcular_distancia(punto, p)
        if distancia < min:
            min = distancia
    return min

#main

Y=np.loadtxt("samplesVDA1.txt")
X=np.arange(0,len(Y)*1/400,1/400)
puntos = list(zip(X, Y))

centros_clusters=[]
plt.plot(X, Y, marker='o')  # marker='o' es para mostrar los puntos, puedes omitirlo si no quieres puntos
plt.xlabel('Tiempo')  # Etiqueta para el eje X
plt.ylabel('Señal')  # Etiqueta para el eje Y
plt.title('Gráfico de Tiempo vs Señal')  # Título del gráfico
plt.grid(True)  # Mostrar una cuadrícula
plt.show()  # Mostrar el gráfico


#NORMALIZACION DE DATOS
data = np.array(puntos)
# Crea una instancia del escalador MinMaxScaler
scaler = MinMaxScaler()

# Ajusta (fit) el escalador a tus datos para aprender los parámetros de la transformación
scaler.fit(data)

# Luego, puedes usar el método transform para normalizar tus datos
normalized_data = scaler.transform(data)
puntos_normalizados = tuple(map(tuple, normalized_data))
X,Y=zip(*puntos_normalizados)

plt.plot(X, Y, marker='o')  # marker='o' es para mostrar los puntos, puedes omitirlo si no quieres puntos
plt.xlabel('Tiempo')  # Etiqueta para el eje X
plt.ylabel('Señal')  # Etiqueta para el eje Y
plt.title('Gráfico de Tiempo vs Señal Normalizado')  # Título del gráfico
plt.grid(True)  # Mostrar una cuadrícula
plt.show()  # Mostrar el gráfico

N = len(puntos_normalizados) 
centros_clusters=[]

ClusteringSubstractivo(puntos_normalizados,centros_clusters,N)
print(np.float64(centros_clusters))
