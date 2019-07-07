import numpy as np
import matplotlib.pyplot as plt
import parzen
import knn
#import bayesian

CLASS_1 = 1
CLASS_2 = 2

# ITEM a)
# Generamos muestras con las distribuciones
N = int(10e4)
x_samples_1 = np.random.uniform(2,10,N)
x_samples_2 = np.random.normal(2,4,N)

# Comprobamos que las distribuciones son correctas
plt.hist(x_samples_1,bins='auto')
plt.show()
plt.hist(x_samples_2,bins='auto')
plt.show()

# ITEM b)
# Objetivo: buscar el mejor h
# Generamos unos datos de validacion que sirvan para encontrar el h optimo
N_validate = 10000
x_validate_1 = np.random.uniform(2, 10, N_validate)
x_validate_2 = np.random.uniform(2, 4, N_validate)
#parzen.find_h([0.001, 0.01, 0.03, 0.1, 0.3, 0.9],x_validate_1,x_validate_2,100)

# El h elegido
h = 0.3

# Usar las muestras generadas en a) y el h elegido para estimar las distribuciones F1 y F2
X = np.linspace(-10,10,100)

h_list = np.linspace(0,1,10)
h = 0.6
p_estim_1 = parzen.estimate(x_samples_1,X,h)
p_estim_2 = parzen.estimate(x_samples_2,X,h)

plt.plot(X,p_estim_1, 'r-')
plt.plot(X,p_estim_2, 'b-')
plt.show()

# ITEM c) Estimar usando k vecinos usando k= 1, 10, 50, 100
k_list=[2]
for k in k_list:
    p_estim_1_knn = knn.knn_estimate(k,x_samples_1,X)
    p_estim_2_knn = knn.knn_estimate(k,x_samples_2,X)

# d) Realizar un clasificador para B y C y clasificar 10e2 muestras nuevas

# e) Implementar clasificacion del K vecino mas cercano para K = 1, 11 y 51.

#       Calcular el error al clasificar las mismas muestras de D).
# Conclusionesss