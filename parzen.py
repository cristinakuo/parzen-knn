import numpy as np
import matplotlib.pyplot as plt
CLASS_1 = 1
CLASS_2 = 2
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

def bayesian_classify(px_1,px_2):
    label_test = []
    for p1, p2 in zip(px_1, px_2):
        if p1 > p2:
            label_test.append(1)
        else:
            label_test.append(2)

    return label_test

def find_h(h_list, x_sample_1, x_sample_2, N_classify):
    x_mix = []
    label_real = []

    # Generate mix of distributions
    U = np.random.uniform(0, 1, N_classify)
    for u in U:
        if u < 0.5:
            sample = np.random.uniform(2, 10)
            label_real.append(CLASS_1)
        else:
            sample = np.random.normal(2, 4)
            label_real.append(CLASS_2)
        x_mix.append(sample)

    plt.hist(x_mix, bins='auto')
    plt.show()

    errors = []
    for h in h_list:
        px_given_1 = estimate(x_sample_1, x_mix, h)
        px_given_2 = estimate(x_sample_2, x_mix, h)
        label_test = bayesian_classify(px_given_1, px_given_2)

        k = 0

        # PLOT
        for lb_test, lb_real in zip(label_test, label_real):
            if lb_test == 1:
                plt.plot(x_mix[k], 'r.')

            else:
                plt.plot(x_mix[k], 'b.')

            if lb_test != lb_real:
                plt.plot(x_mix[k], 'yo')
            k = k + 1
        plt.show()

        # Error
        label_error = [a - b for a, b in zip(label_real, label_test)]
        n_errors = len([n for n in label_error if n != 0])
        error_rate = n_errors / len(label_error)
        errors.append(error_rate)
        print("h="+str(h)+" error="+str(error_rate))



if __name__ == "__main__":
    # Generamos muestras con las distribuciones
    N = int(10e4)
    x_samples_1 = np.random.uniform(2, 10, N)
    x_samples_2 = np.random.normal(2, 4, N)
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

    px_given_1 = estimate(x_samples_1, x_mix, h)
    px_given_2 = estimate(x_samples_2, x_mix, h)

    label_test = bayesian_classify(px_given_1,px_given_2)

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
    # Error
    label_error = [a - b for a, b in zip(label_real, label_test)]
    n_errors = len([n for n in label_error if n != 0])
    error_rate = n_errors / N

    print("Error rate: " + str(error_rate))

