import warnings
import numpy as np


def estimate_bar_x(x, x_old, H, p, T):

    assert x_old.ndim == 2 and x_old.shape == (p, H)
    assert x.ndim == 2 and x.shape == (p, T)

    bar_x_T = np.ones((H * p, T + 1))

    for t in range(T):
        for i in range(H):

            if t - i >= 0:
                bar_x_T[p * i:p * (i + 1), t + 1] = x[:, t - i]
            else:
                bar_x_T[p * i:p * (i + 1), t + 1] = x_old[:, H + t - i]

    for i in range(H):

        bar_x_T[p * i:p * (i + 1), 0] = x_old[:, H - i - 1]

    return bar_x_T


def transposed_OLS(u, y):

    return np.linalg.inv(u @ u.T) @ u @ y.T


def est_G_i(a, b, c, i):

    return c @ np.linalg.matrix_power(a, i - 1) @ b


def matrixwise_norm(Z, X):
    return np.sqrt(Z.T @ X @ Z)


def spectr(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rho = max(np.abs(np.linalg.eigvals(a)))
        fi_a_vect = [np.linalg.norm(np.linalg.matrix_power(a, i), ord=2) /
                     rho ** i for i in range(1, 1000)]
        return max(fi_a_vect)


def populate_dict(algs, dims):
    dct = dict()
    for alg in algs:
        dct[alg] = np.zeros(dims)
    return dct
