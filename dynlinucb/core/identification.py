import numpy as np
import matplotlib.pyplot as plt


def identify_G_with_D(u_t, y_t, p, m, T, H):
    """identify_G identify the evolution matrix G from previous H inputs
     Input:
         u_t is a (p x T) matrix
         y_t is a (m x T) matrix
         p is the number of input features
         m is the number of output features
         T is the number of samples available
         H is the "look-back" time horizon
     Return:
         G system identification matrix
     """
    bar_u = -1 * np.ones((H * p, T - H + 1))
    target = y_t[:, H - 1:]

    for t in range(T - H + 1):

        for i in range(H):
            bar_u[p * i:p * (i + 1), t] = u_t[:, H + t - i - 1]

    return np.linalg.solve(bar_u @ bar_u.T, bar_u @ target.T).T


def identify_G_closedloop(u_t, y_t, p, m, T, H, lmbd, system_D=False):
    """identify_G_closedloop identifies the evolution matrix G with a closed
       loop approach from previous H inputs and outputs
    Input:
        u_t is a (p x T) matrix
        y_t is a (m x T) matrix
        p is the number of input features
        m is the number of output features
        T is the number of samples available
        H is the "look-back" time horizon
        lmbd is the regularization coefficient
        system_D (default: False) is a flag, True if we want to consider D in the
                code, False otherwise
    Return:
        G_cl system identification matrix
    """

    target = y_t[:, H:]

    if not system_D:

        fi_t = -1 * np.ones((H * (m + p), T - H))

        for t in range(H, T):

            for i in range(1, H + 1):
                fi_t[m * (i - 1):m * i, t - H] = y_t[:, t - i]
                fi_t[m * H + p * (i - 1):m * H + p * i, t - H] = u_t[:, t - i]

        return np.linalg.solve((lmbd * np.eye(fi_t.shape[0])) + (
                fi_t @ fi_t.T), fi_t @ target.T).T

    else:

        fi_t = -1 * np.ones((H * (m + p) + p, T - H))

        for t in range(H, T):

            for i in range(1, H + 1):
                fi_t[m * (i - 1):m * i, t - H] = y_t[:, t - i]
                fi_t[m * H + p * i:m * H + p * (i + 1), t - H] = u_t[:, t - i]

            fi_t[m * H:m * H + p, t - H] = u_t[:, t]

        return np.linalg.solve((lmbd * np.eye(fi_t.shape[0])) + (
                fi_t @ fi_t.T), fi_t @ target.T).T


def hokalman_system_identification(u_t, y_t, mp, T, H, n, lmbd=0.0,
                                   detect_D=True, make_plot=False):
    """ identify_system identifies the system matrices A, B, C, D
            u_t is a (p x T) matrix
            y_t is a (m x T) matrix
            mp is the number of input and output features
            T is the number of samples available
            H is the "look-back" time horizon
            n is the dimension of the system to identify
            lmbd is the regularization coefficient """

    if detect_D:

        cal_gd = identify_G_closedloop(
            u_t, y_t, mp, mp, T, H, lmbd, system_D=True
        )
        cal_g = np.hstack((cal_gd[:, :mp * H], cal_gd[:, mp * (H + 1):]))
        D = cal_gd[:, mp * H:mp * (H + 1)]

    else:

        cal_g = identify_G_closedloop(u_t, y_t, mp, mp, T, H, lmbd)

    H_F = np.zeros(((H // 2) * mp, (H // 2) * mp))
    H_G = np.zeros(((H // 2) * mp, (H // 2) * mp))

    for i in range(H // 2):
        for j in range(H // 2):
            H_F[i * mp:(i + 1) * mp,
            j * mp:(j + 1) * mp] = cal_g[:, (i + j) * mp:(i + j + 1) * mp]
            H_G[i * mp:(i + 1) * mp,
            j * mp:(j + 1) * mp] = cal_g[:, H * mp + (
                    i + j) * mp:H * mp + (i + j + 1) * mp]

    d1_hf, d2_hf = H_F.shape[0] / mp, (H_F.shape[1] - mp) / mp
    d1_hg, d2_hg = H_G.shape[0] / mp, (H_G.shape[1] - mp) / mp

    assert d1_hf + d2_hf + 1 == H and d1_hg + d2_hg + 1 == H
    assert d1_hf == d1_hg and d2_hf == d2_hg

    hankel_ = np.hstack((H_F[:, :-mp], H_G[:, :-mp]))
    hankel_plus = np.hstack((H_F[:, mp:], H_G[:, mp:]))

    u_i, sigma_i, v_i = np.linalg.svd(hankel_, full_matrices=False)

    if make_plot:
        plt.plot(sigma_i)

    sigma_i[n:] = 0
    sigma_t = sigma_i * np.eye(sigma_i.shape[0])

    bold_o_i = u_i @ (sigma_t ** (1 / 2))
    bold_o_i = bold_o_i[:, :n]
    bold_c_i = (sigma_t ** (1 / 2)) @ v_i
    bold_c_i = bold_c_i[:n, :]
    C1 = bold_c_i[:, :bold_c_i.shape[1] // 2]
    C2 = bold_c_i[:, bold_c_i.shape[1] // 2:]

    C = bold_o_i[:mp, :]
    B = C2[:, :mp]
    F = C1[:, :mp]
    bar_A = np.linalg.pinv(bold_o_i) @ hankel_plus @ np.linalg.pinv(bold_c_i)
    A = bar_A + F @ C

    if detect_D:
        return A, B, C, D
    else:
        return A, B, C


def hokalman_system_identification_base_withD(u_t, y_t, mp, T, H, n):
    """ identify_system identifies the system matrices A, B, C, D
            u_t is a (p x T) matrix
            y_t is a (m x T) matrix
            mp is the number of input and output features
            T is the number of samples available
            H is the "look-back" time horizon
            n is the dimension of the system to identify"""

    cal_gd = identify_G_with_D(u_t, y_t, mp, mp, T, H)
    D = cal_gd[:, :mp]
    cal_g = cal_gd[:, mp:]

    H_G = np.zeros(((H // 2) * mp, (H // 2) * mp))

    for i in range(H // 2):
        for j in range(H // 2):
            H_G[i * mp:(i + 1) * mp,
            j * mp:(j + 1) * mp] = cal_g[:, (i + j) * mp:(i + j + 1) * mp]

    d1_hg, d2_hg = H_G.shape[0] / mp, (H_G.shape[1] - mp) / mp

    assert d1_hg + d2_hg + 1 == H

    hankel_ = H_G[:, :-mp]
    hankel_plus = H_G[:, mp:]

    u_i, sigma_i, v_i = np.linalg.svd(hankel_, full_matrices=False)

    sigma_i[n:] = 0
    sigma_t = sigma_i * np.eye(sigma_i.shape[0])

    bold_o_i = u_i @ (sigma_t ** (1 / 2))
    bold_o_i = bold_o_i[:, :n]
    bold_c_i = (sigma_t ** (1 / 2)) @ v_i
    bold_c_i = bold_c_i[:n, :]

    C = bold_o_i[:mp, :]
    B = bold_c_i[:, :mp]
    A = np.linalg.pinv(bold_o_i) @ hankel_plus @ np.linalg.pinv(bold_c_i)

    return A, B, C, D
