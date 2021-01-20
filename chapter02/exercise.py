from ten_armed_testbed import Bandit, simulate

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class NonstationaryBandit(Bandit):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self,
                 k_arm=10, epsilon=0.,
                 initial=0., step_size=0.1, sample_averages=False,
                 UCB_param=None, gradient=False, gradient_baseline=False, true_reward=0.):

        super().__init__(k_arm, epsilon, initial, step_size, sample_averages,
                         UCB_param, gradient, gradient_baseline, true_reward)

    def reset(self):
        super().reset()
        # real reward for each action
        self.q_true = np.zeros(self.k)+self.true_reward

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward -
                                          self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * \
                (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * \
                (reward - self.q_estimation[action])

        # non stationary setting
        self.q_true = self.q_true+np.random.normal(0, 0.01, self.k)
        return reward


def exercise_2_5(runs=10000, time=1000):
    epsilons = [0.1]
    # Sample averaging
    bandits = [NonstationaryBandit(
        epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 2, 3)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Step size
    alpha = 0.1
    bandits = [NonstationaryBandit(
        epsilon=eps, step_size=alpha) for eps in epsilons]

    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.subplot(2, 2, 2)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='alpha = %.02f' % (alpha))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 2, 4)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='alpha = %.02f' % (alpha))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')


    plt.legend()
    plt.savefig('../images/exercise_2_5.png')
    plt.close()


if __name__ == '__main__':
    exercise_2_5()
