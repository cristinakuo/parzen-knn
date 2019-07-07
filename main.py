import numpy as np
import matplotlib.pyplot as plt
import parzen
import knn
import plots as myplt
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
p_estim_1_parzen = parzen.estimate(x_samples_1,X,h)
p_estim_2_parzen = parzen.estimate(x_samples_2,X,h)

plt.plot(X,p_estim_1_parzen, 'r-')
plt.plot(X,p_estim_2_parzen, 'b-')
plt.show()

# ITEM c) Estimar usando k vecinos usando k= 1, 10, 50, 100
k_list = [1,10,50,100]
for k in k_list:
    p_estim_1_knn = knn.knn_estimate(k, x_samples_1, X)
    p_estim_2_knn = knn.knn_estimate(k, x_samples_2, X)

    myplt.plot_distributions(X,p_estim_1_knn,p_estim_2_knn)

# d) Realizar un clasificador para B y C y clasificar 10e2 muestras nueva
# Generate mixture distribution
N_test = 100
x_mix = []
label_real = []
label_test = []

U = np.random.uniform(0, 1, N_test)
for u in U:
    if u < 0.5:
        sample = np.random.uniform(2, 10)
        label_real.append(CLASS_1)
    else:
        sample = np.random.normal(2, 4)
        label_real.append(CLASS_2)
    x_mix.append(sample)

h=0.3
k = 10

# Clasificador con parzen
px_given_1_parzen = parzen.estimate(x_samples_1, x_mix, h)
px_given_2_parzen = parzen.estimate(x_samples_2, x_mix, h)
label_test_parzen = parzen.bayesian_classify(px_given_1_parzen, px_given_2_parzen)
myplt.plot_distributions(x_mix,px_given_1_parzen,px_given_2_parzen)
myplt.plot_with_labels(x_mix,label_real,label_test_parzen)
err_parzen = myplt.get_error(label_real,label_test_parzen)
print('Error with parzen is: '+str(err_parzen))

# Clasificador con knn
for k in k_list:
    px_given_1_knn = knn.knn_estimate(k,x_samples_1, x_mix)
    px_given_2_knn = knn.knn_estimate(k,x_samples_2, x_mix)
    label_test_knn = parzen.bayesian_classify(px_given_1_knn, px_given_2_knn)
    myplt.plot_distributions(x_mix,px_given_1_knn,px_given_2_knn)
    myplt.plot_with_labels(x_mix,label_real,label_test_knn)
    err_knn = myplt.get_error(label_real,label_test_knn)
    print('Error with knn is: '+str(err_knn) + " k=" +str(k))

# e) Implementar clasificacion del K vecino mas cercano para K = 1, 11 y 51.

#       Calcular el error al clasificar las mismas muestras de D).
# Conclusionesss