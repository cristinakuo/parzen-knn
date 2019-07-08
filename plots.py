import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

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


def plot_distributions(x,px_1,px_2,style_opt):
    plt.plot(x,px_1,style_opt,markersize=4,label='Distribucion 1')
    plt.plot(x,px_2,style_opt,markersize=4,label='Distribucion 2')
    plt.legend()
    plt.show()

def plot_est_vs_theo(X,px_1,px_2,style_opt,title="no_title"):

    plt.plot(X, px_1, style_opt, markersize=4, label='Experimental 1')
    plt.plot(X, px_2, style_opt, markersize=4, label='Experimental 2')

    normal_pdf = []
    uniform_pdf = []

    X_theo = np.linspace(min(X), max(X), 100)
    for x in X_theo:
        normal_pdf.append(norm.pdf(x, 2, 2))
        if x>=2 and x<=10:
            uniform_pdf.append(1/8)
        else:
            uniform_pdf.append(0)

    plt.plot(X_theo, uniform_pdf, 'b--', linewidth=0.7, label='Teorico 1')
    plt.plot(X_theo,normal_pdf, 'k--', linewidth=0.7, label='Teorico 2')

    if title!="no_title":
        plt.title(title)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    p1 = [1,2,3,4,5]
    p2 = [6,7,8,9,10]
    x = [1,2,3,4,5]

    p1 = np.random.normal(2,4,1000)
    p2 = np.random.normal(-1,2,1000)
    X = np.linspace(-10,15,1000)
    #plot_distributions(x,p1,p2,'.')
    plot_exp_vs_theo(X,p1,p2,'.','hi')

