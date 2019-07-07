import numpy as np
import matplotlib.pyplot as plt

# Returns a list of tuples (index, value)
def knn_search(k,data,ref):
    data_indexed = enumerate(data)
    return sorted(data_indexed, key=lambda x: abs(x[1]-ref))[:k]

def knn_estimate(k,samples_data,X):
    p_estimate = []
    N = len(samples_data)

    if k == 1:
        for x in X:
            k_nearest = knn_search(k, samples_data, x)
            nearest = [x[1] for x in k_nearest]  # TODO: es solo 1 tuple
            p_estimate.append(1/(2*abs(x-nearest[0])))
    else:
        for x in X:
            k_nearest = knn_search(k,samples_data,x)
            k_nearest = [x[1] for x in k_nearest] # keeps second element of each tupple
            V = abs(max(k_nearest)-min(k_nearest))
            p_estimate.append(k/(N*V))

    return p_estimate

def knn_classify(k,samples_class1,samples_class2,x):
    result = knn_search(k,samples_class1+samples_class2,x)
    result_val = [a[1] for a in result]  # keeps second element of each tuple in result
    result_index = [a[0] for a in result] # keeps first element of each tuple in result
    label_real = [1]*len(samples_class1)+[2]*len(samples_class2) # real label of data
    label_knn = [label_real[n] for n in result_index] # label of the points obtained with knnsearch

    count1 = len([n for n in label_knn if n==1])
    count2 = len([n for n in label_knn if n==2])

    if count1 > count2:
        return 1

    else:
        return 2

if __name__ == "__main__":
    # Generamos muestras con las distribuciones
    N = int(10e4)
    x_samples_1 = np.random.uniform(2, 10, N).tolist() # arrays
    x_samples_2 = np.random.normal(2, 4, N).tolist()
    X = np.linspace(-10, 20, 100)
    #p_estimate = knn_estimate(k,x_samples_normal, X)

    #plt.plot(X, p_estimate, 'r-')
    #plt.show()

    #class_result = knn_classify(5,x_samples_normal,x_samples_uniform,)
    #print(class_result)

