import numpy as np
from tqdm.auto import tqdm


class Runner:

    def __init__(self, environment, agent, n_trials, horizon,
                 action_size, n_actions, actions=None):
        self.environment = environment
        self.agent = agent
        self.n_trials = n_trials
        self.horizon = horizon
        self.action_size = action_size
        self.n_actions = n_actions
        self.actions = actions
        if action_size > 1:
            assert actions is not None, 'Provide actions'
            assert actions.shape == (n_actions, action_size)

    def perform_simulations(self):
        all_actions = np.zeros((self.n_trials, self.horizon, self.action_size))
        for sim_i in tqdm(range(self.n_trials)):
            self.environment.reset(sim_i)
            self.agent.reset()
            action_vect = self._run_simulation()
            assert action_vect.shape == (self.horizon, self.action_size)
            all_actions[sim_i, :, :] = action_vect
        return all_actions

    def _run_simulation(self):
        action_vect = np.zeros((self.horizon, self.action_size))
        for t in range(self.horizon):
            action = self.agent.pull_arm()
            if self.action_size > 1:
                if isinstance(action, np.ndarray):
                    reward = self.environment.step(action.reshape(
                        self.action_size, 1))
                else:
                    reward = self.environment.step(self.actions[action, :
                                ].reshape(self.action_size, 1))
            else:
                reward = self.environment.step(action)
            if isinstance(reward, np.ndarray):
                self.agent.update(reward[0, 0])
            else:
                self.agent.update(reward)
            if isinstance(action, np.ndarray):
                action_vect[t, :] = action.ravel()
            else:
                action_vect[t, :] = self.actions[action, :]
        return action_vect
