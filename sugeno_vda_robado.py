import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.linear_model import LinearRegression

def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
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
    print(centers)
    return labels, centers

def calcular_ecm(data, labels, centers):
    """Calcula el Error Cuadrático Medio (ECM)"""
    ecm = 0
    for i, point in enumerate(data):
        center = centers[labels[i]]
        ecm += np.linalg.norm(point - center) ** 2
    return ecm / len(data)

# Regresión lineal por cluster
def regresion_por_cluster(data, labels, centers):
    """Realiza una regresión lineal para cada cluster."""
    unique_labels = np.unique(labels)
    plt.figure()
    for label in unique_labels:
        cluster_data = data[labels == label]  # Filtrar datos del cluster
        X_cluster = cluster_data[:, 0].reshape(-1, 1)  # Usar la primera columna como X
        y_cluster = cluster_data[:, 1]  # Usar la segunda columna como y

        # Ajustar modelo de regresión lineal
        reg = LinearRegression().fit(X_cluster, y_cluster)
        y_pred = reg.predict(X_cluster)

        # Graficar puntos del cluster y la regresión lineal
        plt.scatter(X_cluster, y_cluster, label=f'Cluster {label}')
        plt.plot(X_cluster, y_pred, label=f'Regresión Cluster {label}', linestyle='--')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regresiones Lineales por Cluster")
    plt.legend()

with open("samplesVDA4.txt", "r") as file:
    data = [int(line.strip()) for line in file]

# Genera el array con pares (x, y)
m = np.array([(0.0025 * i, y) for i, y in enumerate(data)])

r_values = np.linspace(0.5, 2.0, 20)  # Rango de valores para Ra
ecm_values = []

# Calcular ECM para cada valor de Ra
for Ra in r_values:
    labels, centers = subclust2(m, Ra)
    ecm = calcular_ecm(m, labels, centers)
    ecm_values.append(ecm)

# Graficar ECM vs Ra
plt.figure()
plt.plot(r_values, ecm_values, marker='o')
plt.xlabel("Radio de aceptación (Ra)")
plt.ylabel("Error cuadrático medio (ECM)")
plt.title("ECM vs Ra")
plt.grid(True)

# Obtener clusters y realizar regresiones
labels, centers = subclust2(m, 1)

plt.figure()
plt.scatter(m[:,0],m[:,1], c=labels)
plt.scatter(centers[:,0],centers[:,1], marker='X')


# Realizar regresiones por cluster
regresion_por_cluster(m, labels, centers)
plt.show()