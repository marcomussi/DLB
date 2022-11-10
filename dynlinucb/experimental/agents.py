from abc import ABC, abstractmethod
import numpy as np
import math
from random import Random


class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.last_pull = None
        np.random.seed(random_state)
        self.randgen = Random(random_state)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, reward):
        pass

    def reset(self):
        self.t = 0
        self.last_pull = None


class UCB1Agent(Agent):
    def __init__(self, n_arms, sigma, max_reward=1):
        super().__init__(n_arms)
        self.max_reward = max_reward
        self.sigma = sigma
        self.reset()

    def reset(self):
        super().reset()
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a] + self.max_reward * self.sigma * np.sqrt(
            2 * np.log(self.t) / self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        return new_a

    def update(self, reward):
        self.t += 1
        self.avg_reward[self.last_pull] = (self.avg_reward[self.last_pull] *
                                           self.n_pulls[self.last_pull]
                                           + reward) / (
                                                  self.n_pulls[
                                                      self.last_pull] + 1)
        self.n_pulls[self.last_pull] += 1


class Exp3BaseAgent(Agent):
    def __init__(self, n_arms, gamma=0.1, max_reward=1,
                 add_factor=0, random_state=1):
        super().__init__(n_arms, random_state)
        self.gamma = gamma
        self.max_reward = max_reward
        self.add_factor = add_factor
        self.reset()

    def reset(self):
        super().reset()
        self.w = np.ones(self.n_arms)
        self.est_rewards = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])
        return self

    def pull_arm(self):
        self.last_pull = np.random.choice(self.arms,
                                          p=self.probabilities,
                                          size=None)
        return self.last_pull

    def update(self, reward):
        reward = (reward + self.add_factor) / self.max_reward
        self.est_rewards[self.last_pull] = reward / self.probabilities[
            self.last_pull]
        self.w[self.last_pull] *= np.exp(
            self.gamma * self.est_rewards[self.last_pull] / self.n_arms)
        self.w[~np.isfinite(self.w)] = 0
        self.probabilities = (1 - self.gamma) * self.w / sum(
            self.w) + self.gamma / self.n_arms
        self.probabilities[0] = 1 - sum(self.probabilities[1:])


class Exp3Agent(Agent):
    def __init__(self, n_arms, gamma=0.1, eta=1, max_reward=1,
                 add_factor=0, random_state=1):
        super().__init__(n_arms, random_state)
        self.gamma = gamma
        self.eta = eta
        self.max_reward = max_reward
        self.add_factor = add_factor
        self.reset()

    def reset(self):
        super().reset()
        self.G = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])
        return self

    def pull_arm(self):
        self.probabilities /= sum(self.probabilities)
        self.last_pull = np.random.choice(self.arms, p=self.probabilities)
        return self.last_pull

    def update(self, reward):
        reward = (reward + self.add_factor) / self.max_reward
        reward_vect = np.zeros(self.n_arms)
        reward_vect[self.last_pull] = \
            reward / self.probabilities[self.last_pull]
        self.G = self.G + reward_vect
        div = np.sum(np.array([math.exp(self.eta * self.G[i])
                               for i in range(self.n_arms)]))
        for i in range(self.n_arms):
            self.probabilities[i] = math.exp(self.eta * self.G[i]) / div
        self.probabilities = (1 - self.gamma) * self.probabilities + \
                             self.gamma / self.n_arms


class BatchExp3Agent(Agent):
    def __init__(self, n_arms, gamma=0.1, eta=1, max_reward=1,
                 add_factor=0, batch_size=10, horizon=0,
                 method='mean', random_state=1):
        super().__init__(n_arms, random_state)
        self.gamma = gamma
        self.eta = eta
        self.max_reward = max_reward
        self.add_factor = add_factor
        self.method = method
        self.batch_size = batch_size
        self.horizon = horizon
        assert horizon > 0, 'miss-specification in the horizon'
        assert method == 'mean' or method == 'sum', 'error in method'
        self.reset()
        self.t_newaction = np.arange(0, horizon + batch_size + 1, batch_size)
        self.t_estimate = np.arange(1, horizon + batch_size + 1, batch_size)

    def reset(self):
        super().reset()
        self.G = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])
        self.reward_acc = 0
        self.first = True
        self.newaction_idx = 0
        self.estimate_idx = 0
        return self

    def pull_arm(self):
        if self.first or self.t == self.t_newaction[self.newaction_idx]:
            self.probabilities /= sum(self.probabilities)
            self.last_pull = np.random.choice(self.arms,
                                              p=self.probabilities)
            self.first = False
            self.newaction_idx += 1
        return self.last_pull

    def update(self, reward):
        if self.t == self.t_estimate[self.estimate_idx]:
            reward += self.reward_acc
            if self.method == 'mean':
                reward /= self.batch_size
            reward = (reward + self.add_factor) / self.max_reward
            reward_vect = np.zeros(self.n_arms)
            reward_vect[self.last_pull] = \
                reward / self.probabilities[self.last_pull]
            self.G = self.G + reward_vect
            div = np.sum(np.array([math.exp(self.eta * self.G[i])
                                   for i in range(self.n_arms)]))
            for i in range(self.n_arms):
                self.probabilities[i] = math.exp(
                    self.eta * self.G[i]) / div
            self.probabilities = (1 - self.gamma) * self.probabilities + \
                                 self.gamma / self.n_arms
            self.estimate_idx += 1
            self.reward_acc = 0
        else:
            self.reward_acc += reward
        self.t += 1


class ExpertAgent(Agent):
    def __init__(self, n_arms, exp_policy, min_action=0, max_action=1,
                 max_global=1.5):
        super().__init__(n_arms)
        expert_policy = np.array(exp_policy)
        assert expert_policy.shape == (3,)
        expert_policy = expert_policy / sum(expert_policy)
        min_frac, max_frac = min_action / max_action, max_action / max_global
        mask_alterated = np.ones(len(expert_policy), dtype=bool)
        for i in range(len(mask_alterated)):
            if expert_policy[i] < min_frac:
                expert_policy[i] = min_frac
            elif expert_policy[i] > max_frac:
                expert_policy[i] = max_frac
            else:
                mask_alterated[i] = False
        mask_not_alterated = np.logical_not(mask_alterated)
        rescale_from = np.sum(expert_policy[mask_not_alterated])
        rescale_to = 1 - np.sum(expert_policy[mask_alterated])
        expert_policy[mask_not_alterated] *= (rescale_to / rescale_from)
        self.action_vect = expert_policy * max_global

    def reset(self):
        pass

    def pull_arm(self):
        return self.action_vect

    def update(self, reward):
        pass


class AR2Agent(Agent):
    def __init__(self, n_arms, alpha, epoch_size, c0, sigma):
        super().__init__(n_arms)
        self.alpha = alpha
        self.epoch_size = epoch_size
        self.c1 = 24 * c0
        self.sigma = sigma
        self.reset()

    def reset(self):
        super().reset()
        self.t0 = 1
        self.s = 1
        self.tau_trig = np.ones(self.n_arms) * np.inf
        self.tau = np.ones(self.n_arms) * np.inf
        self.est_rewards = np.ones(self.n_arms) * np.inf
        self.i_sup = None
        self.last_last_pull = None
        self.triggered_arms = []
        return self

    def pull_arm(self):
        if self.t0 <= self.n_arms + (self.s - 1) * self.epoch_size:
            new_a = self.n_arms + (self.s - 1) * self.epoch_size - self.t0
        else:
            if self.t0 == self.n_arms + (self.s - 1) * self.epoch_size:
                self.tau_trig = np.ones(self.n_arms) * np.inf
                self.triggered_arms = []
            if self.est_rewards[self.last_pull] >= self.est_rewards[
                self.last_last_pull]:
                self.i_sup = self.last_pull
            else:
                self.i_sup = self.last_last_pull
            for i in range(self.n_arms):
                if (i != self.i_sup) and (i not in self.triggered_arms) and (
                        self.est_rewards[self.i_sup] -
                        self.est_rewards[i] <= self.c1 * self.sigma * np.sqrt(
                    (self.alpha ** 2 - self.alpha ** (2 * (self.t0 -
                                                           self.tau[
                                                               i] + 1))) / (
                            1 - self.alpha ** 2))):
                    self.triggered_arms.append(i)
                    self.tau_trig[i] = self.t0
            if len(self.triggered_arms) > 0 and self.t0 % 2 == 1:
                new_a = np.random.choice(np.where(self.tau_trig == min(
                    self.tau_trig))[0])
            else:
                new_a = self.i_sup
        self.last_last_pull = self.last_pull
        self.last_pull = new_a
        return new_a

    def update(self, X):
        self.tau[self.last_pull] = self.t0
        if self.t0 < self.n_arms + (self.s - 1) * self.epoch_size:
            self.est_rewards[self.last_pull] = self.alpha ** (
                    self.n_arms - self.t0 + self.tau[
                self.last_pull] - 1) * X
        else:
            self.est_rewards[self.last_pull] = self.alpha * X
            for i in range(self.n_arms):
                if i != self.last_pull:
                    self.est_rewards[i] *= self.alpha
            if self.t0 % self.epoch_size == 0:
                self.s += 1
        self.t0 += 1


class DynLinUCBAgent(Agent):
    def __init__(self, n_arms, action_dim, actions, horizon, lmbd,
                 spectral_rad_ub, omega, theta, u_val, b_val, phi_a_ub,
                 sigma_sq, epsilon=0.000001, random_state=1):
        super().__init__(n_arms, random_state)
        spectral_rad_ub = np.abs(spectral_rad_ub)
        assert lmbd > 0 and spectral_rad_ub < 1
        assert actions.shape == (n_arms, action_dim)
        self.action_dim = action_dim
        self.actions = actions
        self.horizon = horizon
        self.lmbd = lmbd
        self.spectral_rad_ub = spectral_rad_ub
        self.omega = omega
        self.theta = theta
        self.u_val = u_val
        self.b_val = b_val
        self.x_val = u_val / (1 - spectral_rad_ub)
        self.phi_a_ub = phi_a_ub
        self.c1 = theta + (omega * b_val * phi_a_ub) / (1 - spectral_rad_ub)
        self.c2 = u_val * omega * phi_a_ub * ((u_val * b_val) / (
                1 - spectral_rad_ub) + self.x_val)
        self.sigma_sq = sigma_sq
        self.bar_sigma_sq = sigma_sq * (
                1 + (omega ** 2 * phi_a_ub ** 2) / (1 - spectral_rad_ub ** 2))
        if spectral_rad_ub < epsilon:
            self.t_estimate = np.linspace(0, horizon - 1, horizon, dtype=int)
        else:
            self.t_estimate = []
            m, aux_sum = 0, 0
            while aux_sum < horizon:
                m += 1
                aux_sum += 1 + math.floor(
                    np.log(m) / np.log(1 / spectral_rad_ub))
                self.t_estimate.append(aux_sum)
        self.t_newaction = list(np.array(self.t_estimate) + 1)
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.action_dim)
        self.b_vect = np.zeros((self.action_dim, 1))
        self.hat_h_vect = np.zeros((self.action_dim, 1))
        self.first = True
        self.newaction_vect_idx = 0
        self.estimate_vect_idx = 0
        return self

    def pull_arm(self):
        if self.first:
            u_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.first = False
        elif self.t == self.t_newaction[self.newaction_vect_idx]:
            u_t, _ = self._estimate_dynlinucb_action()
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.newaction_vect_idx += 1
        return self.last_pull

    def update(self, reward):
        if self.t == self.t_estimate[self.estimate_vect_idx]:
            self.V_t = self.V_t + (self.last_pull @ self.last_pull.T)
            self.b_vect = self.b_vect + self.last_pull * reward
            self.hat_h_vect = np.linalg.inv(self.V_t) @ self.b_vect
            self.estimate_vect_idx += 1
        self.t += 1

    def _beta_t_fun_dynlinucb(self):
        return self.c1 * np.sqrt(self.lmbd) + \
               self.c2 / np.sqrt(self.lmbd) * np.log(math.e * (self.t + 1)) + \
               np.sqrt(
                   2 * self.bar_sigma_sq * (
                           np.log(self.horizon) + (self.action_dim / 2) *
                           np.log(1 + (self.t * (self.u_val ** 2)) / (
                                   self.action_dim * self.lmbd))
                   )
               )

    def _estimate_dynlinucb_action(self):
        bound = self._beta_t_fun_dynlinucb()
        obj_vals = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(self.action_dim, 1)
            obj_vals[i] = self.hat_h_vect.T @ act_i + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t) @ act_i)
        return self.actions[np.argmax(obj_vals), :], np.argmax(obj_vals)


class LinUCBAgent(Agent):
    def __init__(self, n_arms, action_dim, actions, horizon, lmbd,
                 theta, max_global, sigma, random_state=1):
        super().__init__(n_arms, random_state)
        assert lmbd > 0
        assert actions.shape == (n_arms, action_dim)
        self.action_dim = action_dim
        self.actions = actions
        self.lmbd = lmbd
        self.horizon = horizon
        self.theta = theta
        self.max_global = max_global
        self.sigma = sigma
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.action_dim)
        self.b_vect = np.zeros((self.action_dim, 1))
        self.hat_h_vect = np.zeros((self.action_dim, 1))
        self.first = True
        return self

    def pull_arm(self):
        if self.first:
            u_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.first = False
        else:
            u_t, _ = self._estimate_linucb_action()
            self.last_pull = u_t.reshape(self.action_dim, 1)
        return self.last_pull

    def update(self, reward):
        self.V_t = self.V_t + (self.last_pull @ self.last_pull.T)
        self.b_vect = self.b_vect + self.last_pull * reward
        self.hat_h_vect = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _beta_t_fun_linucb(self):
        return self.theta * np.sqrt(self.lmbd) + \
               np.sqrt(
                   2 * np.log(self.horizon) + (
                           self.action_dim * np.log(
                       (self.action_dim * self.lmbd +
                        self.horizon * (self.max_global ** 2)
                        ) / (self.action_dim * self.lmbd)
                   )
                   )
               )

    def _estimate_linucb_action(self):
        bound = self._beta_t_fun_linucb()
        obj_vals = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(self.action_dim, 1)
            obj_vals[i] = self.hat_h_vect.T @ act_i + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t) @ act_i)
        return self.actions[np.argmax(obj_vals), :], np.argmax(obj_vals)


class DLinUCBAgent(Agent):
    def __init__(self, n_arms, action_dim, actions, horizon, lmbd,
                 theta, max_global, discount_factor, sigma, random_state=1):
        super().__init__(n_arms, random_state)
        assert lmbd > 0
        assert actions.shape == (n_arms, action_dim)
        self.action_dim = action_dim
        self.actions = actions
        self.lmbd = lmbd
        self.discount_factor = discount_factor
        self.reg_param = \
            lambda time: self.discount_factor ** (-time) * self.lmbd
        self.horizon = horizon
        self.theta = theta
        self.sigma = sigma
        self.max_global = max_global
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.action_dim)
        self.Vtilde_t = self.lmbd * np.eye(self.action_dim)
        self.b_vect = np.zeros((self.action_dim, 1))
        self.hat_h_vect = np.zeros((self.action_dim, 1))
        self.first = True
        return self

    def pull_arm(self):
        if self.first:
            u_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.first = False
        else:
            u_t, _ = self._estimate_dlinucb_action()
            self.last_pull = u_t.reshape(self.action_dim, 1)
        return self.last_pull

    def update(self, reward):
        self.V_t = self.discount_factor * self.V_t + (
                self.last_pull @ self.last_pull.T
        ) + (1 - self.discount_factor
             ) * self.reg_param(self.t) * np.eye(self.action_dim)
        self.Vtilde_t = self.discount_factor ** 2 * self.Vtilde_t + (
                self.last_pull @ self.last_pull.T
        ) + (1 - self.discount_factor ** 2
             ) * self.reg_param(self.t) * np.eye(self.action_dim)
        self.b_vect = self.discount_factor * self.b_vect + \
                      self.last_pull * reward
        self.hat_h_vect = np.linalg.inv(self.V_t) @ self.b_vect
        self.t += 1

    def _beta_t_fun_dlinucb(self):
        return self.theta * np.sqrt(self.reg_param(self.t)) + self.sigma * \
               np.sqrt(2 * np.log(self.horizon) + (self.action_dim * np.log(
                   1 + ((self.max_global ** 2 * (1 - self.discount_factor ** (
                           2 * self.t)))) /
                   (self.action_dim * self.reg_param(self.t) * (
                           1 - self.discount_factor ** 2))
               )))

    def _estimate_dlinucb_action(self):
        bound = self._beta_t_fun_dlinucb()
        obj_vals = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(self.action_dim, 1)
            obj_vals[i] = act_i.T @ self.hat_h_vect + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t
                                        ) @ self.Vtilde_t @ np.linalg.inv(
                    self.V_t) @ act_i)
        return self.actions[np.argmax(obj_vals), :], np.argmax(obj_vals)


class LinearAgent(Agent):
    def __init__(self, n_arms, action_dim, actions, horizon, lmbd,
                 spectral_rad_ub, omega, theta, u_val, b_val, phi_a_ub,
                 sigma_sq, delta, gamma=0.9999, epsilon=0.0001, random_state=1,
                 dynlinucb=False, linucb=False, dlinucb=False):
        super().__init__(n_arms, random_state)
        spectral_rad_ub = np.abs(spectral_rad_ub)
        assert lmbd > 0 and spectral_rad_ub < 1
        assert actions.shape == (n_arms, action_dim)
        assert int(dynlinucb) + int(linucb) + int(dlinucb) == 1, \
            'Algorithm not correctly selected'
        assert (not dlinucb) or (0 < gamma < 1)
        self.action_dim = action_dim
        self.actions = actions
        self.horizon = horizon
        self.lmbd = lmbd
        self.delta = delta
        self.dynlinucb = dynlinucb
        self.linucb = linucb
        self.dlinucb = dlinucb
        self.gamma = gamma
        self.spectral_rad_ub = spectral_rad_ub
        self.omega = omega
        self.theta = theta
        self.u_val = u_val
        self.b_val = b_val
        self.x_val = u_val / (1 - spectral_rad_ub)
        self.phi_a_ub = phi_a_ub
        self.c1 = theta + (omega * b_val * phi_a_ub) / (1 - spectral_rad_ub)
        self.c2 = u_val * omega * phi_a_ub * ((u_val * b_val) / (
                1 - spectral_rad_ub) + self.x_val)
        self.sigma_sq = sigma_sq
        self.bar_sigma_sq = sigma_sq * (
                1 + (omega ** 2 * phi_a_ub ** 2) / (1 - spectral_rad_ub ** 2))
        if spectral_rad_ub < epsilon or (not self.dynlinucb):
            self.t_estimate = np.linspace(0, horizon - 1, horizon, dtype=int)
        else:
            self.t_estimate = []
            m, aux_sum = 0, 0
            while aux_sum < horizon:
                m += 1
                aux_sum += 1 + math.floor(
                    np.log(m) / np.log(1 / spectral_rad_ub))
                self.t_estimate.append(aux_sum)
        self.t_newaction = list(np.array(self.t_estimate) + 1)
        self.reset()

    def reset(self):
        super().reset()
        self.V_t = self.lmbd * np.eye(self.action_dim)
        self.b_vect = np.zeros((self.action_dim, 1))
        self.hat_h_vect = np.zeros((self.action_dim, 1))
        self.first = True
        self.newaction_vect_idx = 0
        self.estimate_vect_idx = 0
        return self

    def pull_arm(self):
        if self.first:
            u_t = self.actions[int(np.random.uniform(high=self.n_arms)), :]
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.first = False
        elif self.t == self.t_newaction[self.newaction_vect_idx]:
            u_t, _ = self._estimate_action()
            self.last_pull = u_t.reshape(self.action_dim, 1)
            self.newaction_vect_idx += 1
        return self.last_pull

    def update(self, reward):
        if self.t == self.t_estimate[self.estimate_vect_idx]:
            self.V_t = self.V_t + (self.last_pull @ self.last_pull.T)
            self.b_vect = self.b_vect + self.last_pull * reward
            self.hat_h_vect = np.linalg.inv(self.V_t) @ self.b_vect
            self.estimate_vect_idx += 1
        self.t += 1

    def _beta_t(self):
        if self.dynlinucb:
            return self.c1 * np.sqrt(self.lmbd) + \
                   self.c2 / np.sqrt(self.lmbd) * np.log(
                math.e * (self.t + 1)) + \
                   np.sqrt(2 * self.bar_sigma_sq * (
                           np.log(1 / self.delta) + (self.action_dim / 2) *
                           np.log(1 + (self.t * (self.u_val ** 2)) / (
                                   self.action_dim * self.lmbd))))
        elif self.linucb:
            return self.c1 * np.sqrt(self.lmbd) + \
                   np.sqrt(2 * self.bar_sigma_sq * (
                           np.log(1 / self.delta) + (self.action_dim / 2) *
                           np.log(1 + (self.t * (self.u_val ** 2)) / (
                                   self.action_dim * self.lmbd))))
        elif self.dlinucb:
            return self.c1 * np.sqrt(self.lmbd) + \
                   np.sqrt(2 * self.bar_sigma_sq * (
                           np.log(1 / self.delta) + (self.action_dim / 2) *
                           np.log(1 + ((self.t * (self.u_val ** 2)) / (
                                   self.action_dim * self.lmbd)) * (
                                          1 - self.gamma ** (2 * self.t)) / (
                                          1 - self.gamma ** 2))))
        else:
            raise ValueError('Error in bounds')

    def _estimate_action(self):
        bound = self._beta_t()
        obj_vals = np.zeros(self.n_arms)
        for i, act_i in enumerate(self.actions):
            act_i = act_i.reshape(self.action_dim, 1)
            obj_vals[i] = self.hat_h_vect.T @ act_i + bound * np.sqrt(
                act_i.T @ np.linalg.inv(self.V_t) @ act_i)
        return self.actions[np.argmax(obj_vals), :], np.argmax(obj_vals)
