from math import log

import matplotlib.pyplot as plt
from numpy.linalg import norm as norm


def draw(f, grad, statistics, optimal_x, dataset_name=None, method_name=None, save=False):
    oracles, xs, times, operations = statistics.oracles, statistics.xs, statistics.times, statistics.operations
    size = len(xs)
    f_optimal = f(optimal_x)
    grad_optimal_norm = norm(grad(optimal_x))

    r_k1 = [log(abs(f_optimal - f(x))) for x in xs[:-1]]
    r_k1.append(r_k1[-1])
    r_k2 = [log((norm(grad(x)) / grad_optimal_norm) ** 2) for x in xs]

    operations = [operation / 1e6 for operation in operations]
    enumerator = list(range(size))

    def draw_plot(x, y, title, xlabel, ylabel, save, dataset_name):
        plt.figure(figsize=(5, 5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        plt.title(title)

        if not save:
            plt.show()
        else:
            name = 'Pictures/' + dataset_name + '_' + title + '_' + xlabel + '_' + ylabel + '.png'
            print(name)
            plt.savefig(name)

    draw_plot(oracles, r_k1, xlabel='OracleCalls', ylabel='FunctionDifference', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(oracles, r_k2, xlabel='OracleCalls', ylabel='GradientRatio', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(times, r_k1, xlabel='TimeInSec', ylabel='FunctionDifference', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(times, r_k2, xlabel='TimeaInSec', ylabel='GradientRatio', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(enumerator, r_k1, xlabel='iterations', ylabel='FunctionDifference', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(enumerator, r_k2, xlabel='iterations', ylabel='GradientRatio', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(operations, r_k1, xlabel='Operations-1e6', ylabel='FunctionDifference', title=method_name, save=save,
              dataset_name=dataset_name)
    draw_plot(operations, r_k2, xlabel='Operations-1e6', ylabel='GradientRatio', title=method_name, save=save,
              dataset_name=dataset_name)
