import matplotlib.pyplot as plt

def plot_with_labels(x,label_real,label_test):
    n = 0
    for lb_test, lb_real in zip(label_test, label_real):
        if lb_test == 1:
            plt.plot(x[n], 'r.')
        else:
            plt.plot(x[n], 'b.')
        if lb_test != lb_real:
            plt.plot(x[n], 'yx')
        n = n + 1

    plt.show()


def plot_distributions(x,px_1,px_2):
    plt.plot(x,px_1,'.r')
    plt.plot(x,px_2,'.b')
    plt.show()


# Error
# TODO: sacar de aca
def get_error(label_real,label_test):
    label_error = [a - b for a, b in zip(label_real, label_test)]
    n_errors = len([n for n in label_error if n != 0])
    error_rate = n_errors / len(label_error)
    return error_rate
