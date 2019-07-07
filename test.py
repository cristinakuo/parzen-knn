# d) Realizar un clasificador para B y C y clasificar 10e2 muestras nuevas
import numpy as np
import matplotlib.pyplot as plt
import parzen
import knn
import plots


N = int(10e4)
CLASS_1 = 1
CLASS_2 = 2
x_samples_1 = np.random.uniform(2,10,N)
x_samples_2 = np.random.normal(2,4,N)


# e) Implementar clasificacion del K vecino mas cercano para K = 1, 11 y 51.

#       Calcular el error al clasificar las mismas muestras de D).
# Conclusionesss