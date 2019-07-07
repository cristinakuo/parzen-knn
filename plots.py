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
    plt.plot(x,px_1,'.',markersize=3,label='Distribucion 1')
    plt.plot(x,px_2,'.',markersize=3,label='Distribucion 2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    p1 = [1,2,3,4,5]
    p2 = [6,7,8,9,10]
    x = [1,2,3,4,5]
    plot_distributions(x,p1,p2)

