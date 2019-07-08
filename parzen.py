import numpy as np
import matplotlib.pyplot as plt
import rand_gen
import plots as myplt
import errors as err
import bayes as by

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

# Receives a list of h and validation samples
# Prints out the error rate for each h
def try_several_h(h_list, x_validate_1, x_validate_2, N_classify):
    x_mix,label_real = rand_gen.rand_mix(N_classify)

    for h in h_list:
        # Estimate densities with parzen
        px_given_1 = parzen_estimate(x_validate_1, x_mix, h)
        px_given_2 = parzen_estimate(x_validate_2, x_mix, h)

        # Bayesian classification
        label_test = by.bayesian_classify(px_given_1, px_given_2)

        # Plot
        myplt.plot_est_vs_theo(x_mix,px_given_1,px_given_2,'o','h='+str(h))

        # Error
        error_rate = err.get_error(label_real,label_test)
        print("h="+str(h)+" error="+str(error_rate))


if __name__ == "__main__":

    # Generamos muestras con las distribuciones
    N = int(10e4)
    x_samples_1 = np.random.uniform(2, 10, N)
    x_samples_2 = np.random.normal(2, 2, N)
    try_several_h([0.03,0.1,0.3,0.9], x_samples_1, x_samples_2, 100)
    exit()
    h = 1
    X = np.linspace(-10, 10, 100)
    p_estimate = parzen_estimate(x_samples_1, X,h)


