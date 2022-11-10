import sys
import itertools
import math
import numpy as np
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import tikzplotlib as tkz

sys.path.append('.')

from dynlinucb.core.display import system_info
from dynlinucb.experimental.runner import Runner
from dynlinucb.experimental.agents import Exp3Agent, BatchExp3Agent, \
    LinUCBAgent, \
    DLinUCBAgent, DynLinUCBAgent, AR2Agent, ExpertAgent, LinearAgent
from dynlinucb.experimental.environment import DynLinEnvironment
from dynlinucb.core.utils import spectr

plt.rcParams.update(
    {
        "font.size": 20,
        "figure.figsize": (18, 12),
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.loc": "upper right"
    }
)

np.set_printoptions(formatter={'float': lambda x: "{0:0.02f}".format(x)})

warnings.filterwarnings("ignore")

config_id = sys.argv[1]

f = open(f'config/{config_id}.json')
param_dict = json.load(f)

print(f'Parameters: {param_dict}')

exp3 = '\\expthree'
batchexp3 = '\\batchexpthree'
optimal = 'Clairvoyant'
dynlinucb = '\\algnameone'
dynlinucblog = '\\algnamelogt'
linucb = '\\linucbone'
linucblog = '\\linucblogt'
dlinucb = '\\dlinucbone'
dlinucblog = '\\dlinucblogt'
ar2 = '\\artwo'
ucb1 = '\\ucbone'
expert = '\\manualexpert'

alg_list = [optimal, dynlinucblog, dynlinucb, linucblog, linucb, dlinucblog,
            dlinucb, ar2, exp3, batchexp3, ucb1, expert]
line_types = ['-', '-', '-', '--', '--', ':', ':', '-.', '-.', '-.', '-.',
              '-.']
assert len(alg_list) == len(line_types), 'Line types not provided correctly'

horizon = param_dict['horizon']
n_trials = param_dict['n_trials']
sigma = param_dict['noise_sigma']

n = param_dict['n']
p = param_dict['p']
m = param_dict['m']

A = np.array(param_dict['A'])
B = np.array(param_dict['B'])
C = np.array(param_dict['C'])
D = np.array(param_dict['D'])

max_daily_budget = param_dict['max_daily_budget']
max_camp_budget = param_dict['max_camp_budget']
min_camp_budget = param_dict['min_camp_budget']

system_info(A, B, C)

if len(param_dict['expert']) == p:
    do_expert = True
    exp_policy = param_dict['expert']
else:
    do_expert = False

output_mapping = np.ones((1, p))

actions = {alg: np.zeros((n_trials, horizon, p)) for alg in alg_list}
obj_fun = {alg: np.zeros((n_trials, horizon)) for alg in alg_list}
done_flag = {alg: False for alg in alg_list}

np.random.seed(1)
noise = np.random.normal(0, sigma, (n_trials, horizon, n))
out_noise = np.random.normal(0, sigma, (n_trials, horizon, m))

ll = []
for pi in list(itertools.permutations(range(p))):
    budget = max_daily_budget - p * min_camp_budget
    u = np.ones(p) * min_camp_budget
    for i in range(p):
        available = max_camp_budget - u[i]
        delta = min(budget, available)
        u[pi[i]] = u[pi[i]] + delta
        budget = budget - delta
    ll.append(np.copy(u))
discrete_actions = np.unique(np.array(ll), axis=0)
n_arms = discrete_actions.shape[0]

# Optimal Action

MARKOV = (D + C @ np.linalg.inv(np.eye(A.shape[0]) - A) @ B)

steadyvals = np.zeros(n_arms)
for i, act_i in enumerate(discrete_actions):
    steadyvals[i] = output_mapping @ MARKOV @ act_i.reshape(-1, 1)

for i in range(p):
    actions[optimal][:, :, i] = discrete_actions[np.argmax(steadyvals), i]

done_flag[optimal] = True

print('Markov Param: ' + str(MARKOV))
print(
    'Optimal Action: ' + str(list(discrete_actions[np.argmax(steadyvals), :]))
)

# DynLinUCB
spectral_rad_ub = max(np.linalg.eigvals(A))
omega = np.linalg.norm(output_mapping @ C, 2)
u_val = max([np.linalg.norm(np.array(u), 2) for u in discrete_actions])
b_val = np.linalg.norm(B, 2)
x_val = u_val / (1 - spectral_rad_ub)
phi_a_ub = spectr(A)

# DynLinUCB and LinUCB
theta = np.linalg.norm(output_mapping @ D, 2)

# DLinUCB
nonstat_discfactor = 0.999999999

# AR2
ar2_alpha = max(np.linalg.eigvals(A))
ar2_sigma = sigma
ar2_epochs = math.ceil(n_arms * ar2_alpha ** (-3) * ar2_sigma ** (-3))
ar2_c0 = np.sqrt(4 * np.log(1 / (ar2_alpha * sigma)) + 4 * np.log(
    ar2_epochs) + 2 * np.log(4 * n_arms))

# EXP3
exp3_lr = min(1, np.sqrt(n_arms * np.log(n_arms) / (
        (math.e - 1) * 2 * horizon / 3)))
exp3_M = (theta + ((omega * b_val) / (1 - spectral_rad_ub))) * u_val
exp3_eta = exp3_lr / n_arms

# BATCH EXP3
aux, aux_sum = 0, 0
while aux_sum < horizon:
    aux += 1
    aux_sum += 1 + math.floor(np.log(aux) / np.log(1 / spectral_rad_ub))
batchexp3_batchsize = math.ceil(np.log(aux) / np.log(1 / spectral_rad_ub))
batchexp3_lr = min(1, np.sqrt(n_arms * np.log(n_arms) / (
        (math.e - 1) * 2 * (horizon / batchexp3_batchsize) / 3)))
batchexp3_M = (theta + ((omega * b_val) / (1 - spectral_rad_ub))) * u_val
batchexp3_method = 'mean'  # 'mean' or 'sum'
batchexp3_eta = batchexp3_lr / n_arms

to_run_list = [exp3, batchexp3, dynlinucblog, linucblog, dlinucblog, dynlinucb,
               linucb, dlinucb, ar2, expert]

for agent_name in to_run_list:

    if agent_name == exp3:
        agent = Exp3Agent(n_arms, gamma=exp3_lr, eta=exp3_eta,
                          max_reward=4 * exp3_M, add_factor=2 * exp3_M)
    elif agent_name == batchexp3:
        agent = BatchExp3Agent(n_arms, gamma=batchexp3_lr,
                               eta=batchexp3_eta,
                               max_reward=4 * batchexp3_M,
                               add_factor=2 * batchexp3_M,
                               batch_size=batchexp3_batchsize,
                               horizon=horizon,
                               method=batchexp3_method)
    elif agent_name == dynlinucblog:
        agent = LinearAgent(n_arms, p, discrete_actions, horizon,
                            np.log(horizon),
                            spectral_rad_ub, omega, theta, u_val, b_val,
                            phi_a_ub,
                            sigma ** 2, 1.0 / horizon, dynlinucb=True)
    elif agent_name == linucblog:
        agent = LinearAgent(n_arms, p, discrete_actions, horizon,
                            np.log(horizon),
                            spectral_rad_ub, omega, theta, u_val, b_val,
                            phi_a_ub,
                            sigma ** 2, 1.0 / horizon, linucb=True)
    elif agent_name == dlinucblog:
        agent = LinearAgent(n_arms, p, discrete_actions, horizon,
                            np.log(horizon),
                            spectral_rad_ub, omega, theta, u_val, b_val,
                            phi_a_ub,
                            sigma ** 2, 1.0 / horizon, gamma=0.999999,
                            dlinucb=True)
    elif agent_name == dynlinucb:
        agent = DynLinUCBAgent(n_arms, p, discrete_actions, horizon,
                               1, spectral_rad_ub, omega, theta, u_val, b_val,
                               phi_a_ub, sigma ** 2)
    elif agent_name == linucb:
        agent = LinUCBAgent(n_arms, p, discrete_actions, horizon, 1,
                            theta, max_daily_budget, sigma)
    elif agent_name == dlinucb:
        agent = DLinUCBAgent(n_arms, p, discrete_actions, horizon, 1, theta,
                             max_daily_budget, nonstat_discfactor, sigma)
    elif agent_name == ar2:
        agent = AR2Agent(n_arms, ar2_alpha, ar2_epochs, ar2_c0, ar2_sigma)
    elif agent_name == expert:
        if do_expert:
            agent = ExpertAgent(
                n_arms, exp_policy, min_action=0, max_action=1, max_global=1.5
            )
        else:
            continue
    else:
        raise ValueError(agent_name + ' algorithm still not exists!')

    env = DynLinEnvironment(A, B, C, D, n, p, m, horizon, noise, out_noise, n_trials, output_mapping)
    print('Running: ' + agent_name[1:])
    runner = Runner(env, agent, n_trials, horizon, p, n_arms, discrete_actions)
    actions[agent_name] = runner.perform_simulations()
    done_flag[agent_name] = True

inst_regret, cum_regret, cum_regret_mean, cum_regret_std = {}, {}, {}, {}

path = 'results/test_' + str(config_id) + datetime.now().strftime(
    '_%Y_%m_%d_%H_%M')

plt.figure(figsize=(16, 12))

n_plt_samples = 200
hq_plt_samples = 200
split_at = 0.2
assert horizon % 10000 == 0, \
    'Horizon is valid for plot only if can be divided for 10000'

indexer = np.concatenate((
    np.linspace(0, horizon * split_at, hq_plt_samples + 1).astype(int)[:-1],
    np.linspace(horizon * split_at - 1, horizon - 1, n_plt_samples).astype(int)
))

assert done_flag[optimal] and alg_list[0] == optimal, \
    'Error in optimal policy estimate'

for alg_i, alg in enumerate(alg_list):

    if done_flag[alg]:

        for trial_i in range(n_trials):
            obj_fun[alg][trial_i, :] = output_mapping @ MARKOV @ \
                                       actions[alg][trial_i, :, :].T

        inst_regret[alg] = obj_fun[optimal].mean() - obj_fun[alg].T
        cum_regret[alg] = inst_regret[alg].cumsum(axis=0)
        cum_regret_mean[alg] = cum_regret[alg].mean(axis=1)
        cum_regret_std[alg] = cum_regret[alg].std(axis=1) / np.sqrt(n_trials)

        if alg != optimal:
            plt.plot(
                indexer, cum_regret_mean[alg][indexer],
                label=alg, linestyle=line_types[alg_i]
            )
            plt.fill_between(
                indexer,
                cum_regret_mean[alg][indexer] - cum_regret_std[alg][indexer],
                cum_regret_mean[alg][indexer] + cum_regret_std[alg][indexer],
                alpha=0.1
            )

plt.legend(loc='upper left')
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.xlim([0, horizon])
plt.ylim(bottom=0, top=cum_regret_mean[dynlinucblog][-1] * 3)

plt.savefig(path + '.jpg')
tkz.save(path + '.tex')
