import warnings
import numpy as np


def print_system_matrices(a, b, c, d):

    print('Singular values of A:\n', np.linalg.eigvals(a @ a.T),
          '\nA:\n', a, '\nB:\n', b, '\nC:\n', c, '\nD:\n', d)


def system_info(a, b, c):

    ab_contr = np.hstack((b, a @ b, a @ a @ b))
    ab_row_rank = np.linalg.matrix_rank(ab_contr)

    ac_obs = np.hstack((c.T, (c @ a).T, (c @ a @ a).T)).T
    ac_column_rank = np.linalg.matrix_rank(ac_obs)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fi_a_vect = [np.linalg.norm(np.linalg.matrix_power(a, i)) /
                     np.max(np.abs(np.linalg.eigvals(a)))**i
                     for i in range(1, 1000)]

    print('Eigenvalues: ', np.linalg.eigvals(a))
    print('Maximum Eigenvalue: ', np.max(np.abs(np.linalg.eigvals(a))))
    print('Singular Values: ', np.linalg.eigvals(a @ a.T))
    print('AB controllability matrix rank: ', ab_row_rank)
    print('AC observability matrix rank: ', ac_column_rank)
    print('Sup(fi(A)): ', max(fi_a_vect))
