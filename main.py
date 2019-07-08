import numpy as np
import matplotlib.pyplot as plt
import parzen
import knn
import plots as myplt
import rand_gen
import errors as err
#import bayesian

CLASS_1 = 1
CLASS_2 = 2

# ITEM a)
print("ITEM a) Generacion de distribuciones")
# Generate random training samples
N = int(10e4)
x_samples_1 = np.random.uniform(2,10,N)
x_samples_2 = np.random.normal(2,2,N)

# Check with histograms
plt.figure('Histograma clase 1')
plt.hist(x_samples_1,bins='auto')
plt.title('Muestras clase 1: Uniforme (2,10)')
plt.ylabel('Frecuencia')
plt.show()

plt.figure('Histograma clase 2')
plt.hist(x_samples_2,bins='auto')
plt.title('Muestras clase 2: Normal (2,4)')
plt.ylabel('Frecuencia')
plt.show()

# ITEM b)
print("ITEM b) Parzen estimate")

# Objetivo: buscar el mejor h
# Generamos unos datos de validacion que sirvan para encontrar el h optimo
N_validate = 10000
x_validate_1 = np.random.uniform(2, 10, N_validate)
x_validate_2 = np.random.uniform(2, 4, N_validate)
# DEBUG: descomentar
#parzen.try_several_h([0.001, 0.01, 0.03, 0.1, 0.3, 0.9],x_validate_1,x_validate_2,100)


# Usar las muestras generadas en a) y el h elegido para estimar las distribuciones F1 y F2
X = np.linspace(-4,12,100) # TODO: aclarar esto en el informe
h = 0.3 # El valor elegido
#p_estim_1_parzen = parzen.parzen_estimate(x_samples_1,X,h)
#p_estim_2_parzen = parzen.parzen_estimate(x_samples_2,X,h)
#myplt.plot_est_vs_theo(X,p_estim_1_parzen,p_estim_2_parzen,'-','Estimacion con Parzen, h=0.3')

# Para analisis
h = 0.01 # El valor elegido
#p_estim_1_parzen = parzen.parzen_estimate(x_samples_1,X,h)
#p_estim_2_parzen = parzen.parzen_estimate(x_samples_2,X,h)
#myplt.plot_est_vs_theo(X,p_estim_1_parzen,p_estim_2_parzen,'-','Estimacion con Parzen, h=0.01')

h = 0.9 # El valor elegido
#p_estim_1_parzen = parzen.parzen_estimate(x_samples_1,X,h)
#p_estim_2_parzen = parzen.parzen_estimate(x_samples_2,X,h)
#myplt.plot_est_vs_theo(X,p_estim_1_parzen,p_estim_2_parzen,'-','Estimacion con Parzen, h=0.9')


print('ITEM c) knn estimate')
# ITEM c) Estimar usando k vecinos usando k= 1, 10, 50, 100
k_list = [1,10,50,100]
#for k in k_list:
#    p_estim_1_knn = knn.knn_estimate(k, x_samples_1, X)
#    p_estim_2_knn = knn.knn_estimate(k, x_samples_2, X)
#
#    myplt.plot_est_vs_theo(X,p_estim_1_knn,p_estim_2_knn,'-','Estimacion con KNN, k='+str(k))

print('ITEM d) Bayes classifier with parzen and knn estimation')
# ITEM d) Realizar un clasificador para B y C y clasificar 10e2 muestras nueva
# Generar una mezcla de 100 puntos
N_test = 100
x_test,label_real = rand_gen.rand_mix(N_test)

# Clasificador con parzen
#h=0.3
#px_given_1_parzen = parzen.parzen_estimate(x_samples_1, x_test, h)
#px_given_2_parzen = parzen.parzen_estimate(x_samples_2, x_test, h)
#label_test_parzen = parzen.bayesian_classify(px_given_1_parzen, px_given_2_parzen)
#myplt.plot_with_labels(x_test,label_real,label_test_parzen,'Clasificacion bayesiana con estimacion Parzen, h='+str(h))
#err_parzen = err.get_error(label_real,label_test_parzen)
#print('Error with parzen is: '+str(err_parzen))

# Clasificador con knn
#for k in k_list:
#    px_given_1_knn = knn.knn_estimate(k,x_samples_1, x_test)
#    px_given_2_knn = knn.knn_estimate(k,x_samples_2, x_test)
#    label_test_knn = parzen.bayesian_classify(px_given_1_knn, px_given_2_knn)
#    myplt.plot_with_labels(x_test,label_real,label_test_knn,'Clasificacion bayesiana con estimacion KNN, k='+str(k))
#    err_knn = err.get_error(label_real,label_test_knn)
#    print('Error with knn is: '+str(err_knn) + " k=" +str(k))

print('ITEM e) knn classifier')
# e) Implementar clasificacion del K vecino mas cercano para K = 1, 11 y 51.
k_list = [1,11,51]
for k in k_list:
    label_test_knnclas = knn.knn_classify(k,x_samples_1,x_samples_2,x_test)
    err_knnclas = err.get_error(label_real,label_test_knnclas)
    myplt.plot_with_labels(x_test, label_real, label_test_knnclas,'Clasificacion de KNN, k=' + str(k))
    print('Error with knn classify is: ' + str(err_knnclas) + " k=" + str(k))

