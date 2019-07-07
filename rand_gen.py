import numpy as np

CLASS_1 = 1
CLASS_2 = 2
def rand_mix(N):
    x_mix = []
    label_real = []
    U = np.random.uniform(0, 1,N)
    for u in U:
        if u < 0.5:
            sample = np.random.uniform(2, 10)
            label_real.append(CLASS_1)
        else:
            sample = np.random.normal(2, 4)
            label_real.append(CLASS_2)
        x_mix.append(sample)
    return x_mix,label_real

