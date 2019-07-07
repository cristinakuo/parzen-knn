import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    N = 100
    x_mix = []

    U = np.random.uniform(0, 1, N)
    for i in range(0,N):
        if U[i] > 0.5:
            sample = np.random.uniform(2,10)
        else:
            sample = np.random.normal(2,4)

        x_mix.append(sample)
    # Comprobamos que sale bien
    plt.hist(x_mix, bins='auto')
    plt.show()