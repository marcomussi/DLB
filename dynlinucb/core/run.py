import numpy as np


def run_system(input_mx, A, B, C, T, n, p, m,
               noise=None, out_noise=None):
    """run_system estimates the dynamic and the output of a linear system
    Input:
        input_mx is a (p x T) matrix
        A is a (n x n) matrix
        B is a (n x p) matrix
        C is a (m x n) matrix
        T is the number of samples available
        n is the number of state features
        p is the number of input features
        m is the number of output features
        noise is a (n x Z>=T) matrix (default: None)
        out_noise is a (m x Z>=T) matrix (default: None)
    Return:
        output_mx is a (m x T) matrix
    """

    output_mx = np.zeros((m, T))
    state = np.zeros((n, 1))

    for t in range(T):

        output_mx[:, t] = (C @ state).ravel()

        state = A @ state + B @ input_mx[:, t].reshape(p, 1)

        if noise is not None:
            state = state + noise[:, t].reshape(n, 1)

    if out_noise is not None:
        output_mx += out_noise[:, :T]

    return output_mx


def run_system_complete(input_mx, A, B, C, D, T, n, p, m,
                        noise=None, out_noise=None,
                        prev_state=None, prev_state_ts=0,
                        prev_output=None, return_state=False):
    """run_system estimates the dynamic and the output of a linear system
    Input:
        input_mx is a (p x T) matrix
        A is a (n x n) matrix
        B is a (n x p) matrix
        C is a (m x n) matrix
        D is a (m x p) matrix
        T is the number of samples available
        n is the number of state features
        p is the number of input features
        m is the number of output features
        noise is a (n x Z>=T) matrix (default: None)
        out_noise is a (m x Z>=T) matrix (default: None)
    Return:
        output_mx is a (m x T) matrix
    """

    output_mx = np.zeros((m, T))

    if prev_state is not None:
        assert prev_state.shape == (
            n, 1) and prev_state_ts > 0 and prev_output.shape == (
                   m, prev_state_ts)
        output_mx[:, :prev_state_ts] = prev_output
        state = prev_state
    else:
        state = np.zeros((n, 1))

    for t in range(prev_state_ts, T):

        output_mx[:, t] = (C @ state).ravel() + (D @ input_mx[:, t
                                                     ].reshape(p, 1)).ravel()

        state = A @ state + B @ input_mx[:, t].reshape(p, 1)

        if noise is not None:
            state = state + noise[:, t].reshape(n, 1)

    if out_noise is not None:
        output_mx[:, prev_state_ts:T] += out_noise[:, prev_state_ts:T]

    if return_state:
        return output_mx, state
    else:
        return output_mx


class SystemComplete:

    def __init__(self, a, b, c, d, n, p, m, t_max, noise=None, out_noise=None):

        assert t_max > 0 and n > 0 and p > 0 and m > 0
        assert a.shape == (n, n) and b.shape == (n, p) and c.shape == (
            m, n) and d.shape == (m, p)
        if noise is not None:
            assert noise.shape == (n, t_max)
        if out_noise is not None:
            assert out_noise.shape == (m, t_max)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n
        self.p = p
        self.m = m
        self.noise = noise
        self.out_noise = out_noise
        self.current_t = 0
        self.t_max = t_max
        self.output = np.zeros((m, t_max))
        self.state = np.zeros((n, 1))

    def run(self, input_mx):

        assert input_mx.ndim == 2 and input_mx.shape[0] == self.p
        assert input_mx.shape[1] > self.current_t

        new_t = input_mx.shape[1]

        for t in range(self.current_t, new_t):

            self.output[:, t] = (self.c @ self.state).ravel() + (
                    self.d @ input_mx[:, t
                             ].reshape(self.p, 1)).ravel()

            self.state = self.a @ self.state + self.b @ input_mx[:, t].reshape(
                self.p, 1)

            if self.noise is not None:
                self.state = self.state + self.noise[:, t].reshape(self.n, 1)

        if self.out_noise is not None:
            self.output[:, self.current_t:new_t] += self.out_noise[:,
                                                    self.current_t:new_t]

        self.current_t = new_t

        return self.output[:, :new_t]
