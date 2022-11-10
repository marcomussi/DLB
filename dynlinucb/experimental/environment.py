import numpy as np


class DynLinEnvironment:

    def __init__(self, a, b, c, d, n, p, m, horizon, noise, out_noise,
                 n_trials, output_mapping):
        assert horizon > 0 and n > 0 and p > 0 and m > 0
        assert a.shape == (n, n) and b.shape == (n, p) \
               and c.shape == (m, n) and d.shape == (m, p)
        if noise is not None:
            assert noise.shape == (n_trials, horizon, n)
        if out_noise is not None:
            assert out_noise.shape == (n_trials, horizon, m)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n
        self.p = p
        self.m = m
        self.all_noise = noise
        self.all_out_noise = out_noise
        self.horizon = horizon
        self.n_trials = n_trials
        self.output_mapping = output_mapping
        self.t = None
        self.state = None
        self.noise = None
        self.out_noise = None
        self.reset(0)

    def step(self, action):
        assert action.ndim == 2 and action.shape == (
            self.p, 1), 'error in action input'
        output = self.c @ self.state + self.d @ action + \
            self.out_noise[self.t, :].reshape(self.m, 1)
        self.state = self.a @ self.state + self.b @ action + \
            self.noise[self.t, :].reshape(self.n, 1)
        self.t = self.t + 1
        return self.output_mapping @ output

    def reset(self, i_trials):
        assert 0 <= i_trials < self.n_trials, 'trial not available'
        self.state = np.zeros((self.n, 1))
        self.t = 0
        self.noise = self.all_noise[i_trials, :, :]
        assert self.noise.ndim == 2 and self.noise.shape == (
            self.horizon, self.n), 'error in noise'
        self.out_noise = self.all_out_noise[i_trials, :, :]
        assert self.out_noise.ndim == 2 and self.out_noise.shape == (
            self.horizon, self.m), 'error in output_noise'
