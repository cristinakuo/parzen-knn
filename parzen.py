import numpy as np
import matplotlib.pyplot as plt
import rand_gen
import plots as myplt
import errors as err

CLASS_1 = 1
CLASS_2 = 2
def parzen_estimate(data,X,h):
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

def bayesian_classify(px_1,px_2):
    label_test = []
    for p1, p2 in zip(px_1, px_2):
        if p1 > p2:
            label_test.append(1)
        else:
            label_test.append(2)

    return label_test

def try_several_h(h_list, x_validate_1, x_validate_2, N_classify):
    x_mix,label_real = rand_gen.rand_mix(N_classify)

    for h in h_list:
        # Obtenemos las densidades estimadas
        px_given_1 = parzen_estimate(x_validate_1, x_mix, h)
        px_given_2 = parzen_estimate(x_validate_2, x_mix, h)
        label_test = bayesian_classify(px_given_1, px_given_2)

        # Grafico
        myplt.plot_distributions(x_mix,px_given_1,px_given_2)

        # Error
        error_rate = err.get_error(label_real,label_test)
        print("h="+str(h)+" error="+str(error_rate))


if __name__ == "__main__":

    # Generamos muestras con las distribuciones
    N = int(10e4)
    x_samples_1 = np.random.uniform(2, 10, N)
    x_samples_2 = np.random.normal(2, 4, N)
    try_several_h([0.03,0.1,0.3,0.9], x_samples_1, x_samples_2, 100)
    exit()
    h = 1
    #X = np.linspace(-10, 10, 100)
    #p_estimate = estimate(x_samples_normal, X,h)
    # TODO: elegir un h

    # Grafico
    #plt.plot(X, p_estimate, 'r-')
    #plt.show()

    # Generate mixture distribution
    N = 100
    x_mix = []
    label_real = []
    label_test = []

    U = np.random.uniform(0, 1, N)
    for i in range(0, N):
        if U[i] < 0.5:
            sample = np.random.uniform(2, 10)
            label_real.append(1)
        else:
            sample = np.random.normal(2, 4)
            label_real.append(2)
        x_mix.append(sample)

    plt.hist(x_mix, bins='auto')
    plt.show()

    px_given_1 = parzen_estimate(x_samples_1, x_mix, h)
    px_given_2 = parzen_estimate(x_samples_2, x_mix, h)

    label_test = bayesian_classify(px_given_1,px_given_2)

    myplt.plot_with_labels(x_validate,label_real,label_test)

    # Error
    label_error = [a - b for a, b in zip(label_real, label_test)]
    n_errors = len([n for n in label_error if n != 0])
    error_rate = n_errors / N

    print("Error rate: " + str(error_rate))

