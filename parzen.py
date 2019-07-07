import numpy as np
import matplotlib.pyplot as plt

def estimate(data,X,h):
    N = len(data)
    estimate = []
    for x in X:

        sum = 0

        for x_i in data:
            sum = sum + window_function((x-x_i)/h)

        estimate.append(sum/N/h)
    return estimate

def window_function(x):
    if np.abs(x) <= 0.5:
        return 1
    else:
        return 0

def bayesian_classify():
    # Generar muestras de mezcla
    N = 100
    x_mix = []
    label_real = []
    label_test = []

    U = np.random.uniform(0, 1, N)
    for i in range(0, N):
        if U[i] > 0.5:
            sample = np.random.uniform(2, 10)
            label_real.append(2)
        else:
            sample = np.random.normal(2, 4)
            label_real.append(1)
        x_mix.append(sample)

    # Comprobamos que sale bien
    plt.hist(x_mix, bins='auto')
    plt.show()

    p_mix_given_normal = estimate(x_samples_normal, x_mix, h)
    p_mix_given_uniform = estimate(x_samples_uniform, x_mix, h)

    # Clasificacion
    for p1, p2 in zip(p_mix_given_normal, p_mix_given_uniform):
        if p1 > p2:
            label_test.append(1)
        else:
            label_test.append(2)

    k = 0
    for lb_test, lb_real in zip(label_test, label_real):
        if lb_test == 1:
            plt.plot(x_mix[k], 'r.')

        else:
            plt.plot(x_mix[k], 'b.')

        if lb_test != lb_real:
            plt.plot(x_mix[k], 'yo')
        k = k + 1
    plt.show()

    # Calculo de error
    label_error = [a - b for a, b in zip(label_real, label_test)]

    n_errors = len([n for n in label_error if n != 0])
    error_rate = n_errors / N

    print("Error rate is: ")
    print(error_rate)


if __name__ == "__main__":
    # Generamos muestras con las distribuciones
    N = int(10e4)
    x_samples_uniform = np.random.uniform(2, 10, N)
    x_samples_normal = np.random.normal(2, 4, N)
    h = 1
    #X = np.linspace(-10, 10, 100)
    #p_estimate = estimate(x_samples_normal, X,h)
    # TODO: elegir un h

    # Grafico
    #plt.plot(X, p_estimate, 'r-')
    #plt.show()

