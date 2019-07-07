import numpy as np
import matplotlib.pyplot as plt
import parzen

CLASS_1 = 1
CLASS_2 = 2

# ITEM a)
# Generamos muestras con las distribuciones
N = int(10e4)
x_samples_class1 = np.random.uniform(2,10,N)
x_samples_class2 = np.random.normal(2,4,N)

# Comprobamos que las distribuciones son correctas
plt.hist(x_samples_class1,bins='auto')
plt.show()
plt.hist(x_samples_class2,bins='auto')
plt.show()

# ITEM b)
# Estimar distribucion usando parzen
X = np.linspace(-10,10,100)

h_list = np.linspace(0,1,10)
h = 0.6
p_estimate_normal = parzen.estimate(x_samples_normal,X,h)
p_estimate_uniform = parzen.estimate(x_samples_uniform,X,h)

plt.plot(X,p_estimate_normal, 'r-')
plt.plot(X,p_estimate_uniform, 'b-')
plt.show()

# Juntamos las muestras de cada clase para crear una mezcla
U = np.random.uniform(0,1,2*N)
x_mix = []
label_real = []

for u in U:
    if u < 0.5:
        x_mix.append(x_samples_uniform)
        label_real.append(CLASS_1)
    else:
        x_mix.append(x_samples_normal)
        label_real.append(CLASS_2)

for h in h_list:
    #bayesian classifier with x_mix
# Criterio: estimar y usar para clasificar


# c) Estimar usando k vecinos usando k= 1, 10, 50, 100
# d) Realizar un clasificador para B y C y clasificar 10e2 muestras nuevas

# e) Implementar clasificacion del K vecino mas cercano para K = 1, 11 y 51.

#       Calcular el error al clasificar las mismas muestras de D).
# Conclusionesss

