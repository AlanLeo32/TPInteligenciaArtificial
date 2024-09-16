import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

def subclust2(data, Ra, Rb=0.3, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)
    P = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    P = np.sum(np.exp(-P**2/alpha),axis=0)

    centers = []
    i=np.argmax(P)
    C = ndata[i]
    p=P[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-C) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,C))
            else:
                P[i]=0
                restarP = False
        if not any(v>0 for v in P):
            continuar = False
    distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers



from sklearn.preprocessing import MinMaxScaler
import time

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        plt.xlabel("X1")
        plt.ylabel("U")
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii)
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)
        return len(self.rules)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu
        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()




data_y=np.loadtxt("samplesVDA1.txt")


# Frecuencia de muestreo
sample_frequency = 400  # Hz

# Intervalo entre muestras en segundos
sample_interval = 1 / sample_frequency  # 2.5 ms

# Crear un arreglo de tiempo en segundos
data_x = np.arange(0, len(data_y) * sample_interval, sample_interval)

plt.plot(data_x,data_y)
plt.xlabel('Tiempo (s)')
plt.ylabel('Señal VDA')
plt.title('Gráfico de la Señal VDA en función del Tiempo')
plt.grid(True)
plt.show()

data = np.vstack((data_x,data_y)).T
# Graficar los datos en función del tiempo

#--------------------------------------------
# Crea una instancia del escalador MinMaxScaler
scaler = MinMaxScaler()

# Ajusta (fit) el escalador a tus datos para aprender los parámetros de la transformación
scaler.fit(data)

# Luego, puedes usar el método transform para normalizar tus datos
normalized_data = scaler.transform(data)



#--------------------------------------------
plt.plot(normalized_data[:,0],normalized_data[:,1])
plt.xlabel('Tiempo (s)')
plt.ylabel('Señal VDA')
plt.title('Gráfico de la Señal VDA en función del Tiempo')
plt.grid(True)
plt.show()

# Crea una instancia de la clase 'fis'
fis2 = fis()

reglas = np.array([])
MseHistorial = np.array([])
msea=999;
a=0
# Llama a la función 'genfis' para generar el modelo Sugeno (ajusta el valor de 'radii' según tus necesidades)
val = np.linspace(0.01, 5.0, 10)

for i in val:
  print(val)
  reglas=np.append(reglas,fis2.genfis(data,i))
  r = fis2.evalfis(np.vstack(data_x))
  # Calcula las diferencias entre las predicciones y los valores reales
  diferencias = r - data_y

  # Eleva al cuadrado las diferencias
  diferencias_al_cuadrado = diferencias**2

  # Calcula el MSE como el promedio de las diferencias al cuadrado
  mse = np.mean(diferencias_al_cuadrado)
  MseHistorial=np.append(MseHistorial,mse)

#--------------------------------------------------------------------------
# Ordenar los datos
sorted_indices = np.argsort(reglas)
reglas_sorted = reglas[sorted_indices]
MseHistorial_sorted = MseHistorial[sorted_indices]

# Configurar la figura
plt.figure()
plt.scatter(reglas_sorted, MseHistorial_sorted, label="Datos", marker='o')


plt.xlabel("Reglas")
plt.ylabel("MSE")
plt.legend()
plt.show()


#---------------------------------------------------------------------------
# Define los pesos para cantidad de reglas y MSE (ajusta según tu preferencia)
peso_reglas = 0.6  # Peso para la cantidad de reglas
peso_mse = 0.4    # Peso para el MSE
puntaje_combinado= np.array([])
for i in range(0,len(reglas),1):
  # Calcula el puntaje combinado para cada modelo
  puntaje_combinado=np.append(puntaje_combinado, peso_reglas * reglas[i] + peso_mse * MseHistorial[i])
  print("Reglas:", reglas[i],"Mejor modelo - MSE:", MseHistorial[i])

# Encuentra el índice del modelo con el puntaje combinado más bajo
mejor_modelo_indice = np.argmin(puntaje_combinado)
print(mejor_modelo_indice)
# Imprime la cantidad de reglas y el MSE del mejor modelo
print("Mejor modelo - Reglas:", reglas[mejor_modelo_indice])
print("Mejor modelo - MSE:", MseHistorial[mejor_modelo_indice])

#---------------------------------------------------------------------------

print(val[mejor_modelo_indice],"este")
fis2.genfis(data,val[mejor_modelo_indice])

# Visualiza las funciones de membresía de entrada (opcional)
fis2.viewInputs()

# Evalúa el modelo Sugeno en tus datos de entrada
r = fis2.evalfis(np.vstack(data_x))


# Calcula las diferencias entre las predicciones y los valores reales
diferencias = r - data_y

# Eleva al cuadrado las diferencias
diferencias_al_cuadrado = diferencias**2

# Calcula el MSE como el promedio de las diferencias al cuadrado
mse = np.mean(diferencias_al_cuadrado)

print(mse)
# Grafica los resultados
plt.figure()
plt.plot(data_x, data_y, label="Datos reales")
plt.plot(data_x, r, linestyle='--', label="Modelo Sugeno")
plt.xlabel("Tiempo(segundos)")
plt.ylabel("VDA")
plt.legend()
plt.show()



sobremuestreo=data_x+0.005


# Evalúa el modelo Sugeno en tus datos de entrada
r = fis2.evalfis(np.vstack(sobremuestreo))


# Calcula las diferencias entre las predicciones y los valores reales
diferencias = r - data_y

# Eleva al cuadrado las diferencias
diferencias_al_cuadrado = diferencias**2

# Calcula el MSE como el promedio de las diferencias al cuadrado
mse = np.mean(diferencias_al_cuadrado)

print(mse)
# Grafica los resultados
plt.figure()
plt.plot(data_x, data_y, label="Datos reales")
plt.plot(data_x, r, linestyle='--', label="Modelo Sugeno")
plt.xlabel("Tiempo(segundos)")
plt.ylabel("VDA")
plt.legend()
plt.show()