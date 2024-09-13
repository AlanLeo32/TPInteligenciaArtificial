import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from scipy import interpolate

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



with open("samplesVDA1.txt", "r") as file:
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



r,c = subclust2(m,0.7)

plt.figure()
plt.scatter(m[:,0],m[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')

# Ejemplo de datos
x_original = np.linspace(0, 10, 100)  # Puntos de la señal original
y_original = np.sin(x_original)  # Señal original (por ejemplo, una función seno)

# Crear la función de interpolación spline cúbica
spline = interpolate.CubicSpline(x_original, y_original)

# Generar puntos adicionales para sobremuestrear
x_new = np.linspace(0, 10, 1000)  # Aumentar resolución
y_new = spline(x_new)

# Graficar señal original y sobremuestreada
plt.plot(x_original, y_original, 'o', label="Original")
plt.plot(x_new, y_new, '-', label="Sobremuestreada")
plt.legend()
plt.show()
