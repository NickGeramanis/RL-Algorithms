import random

import numpy as np
from gym import Env

from src.features.discretizer import Discretizer
from src.rl_algorithms.rl_algorithm import RLAlgorithm


class TabularMonteCarlo(RLAlgorithm):
    __env: Env
    __discount_factor: float
    __discretizer: Discretizer
    __q_table: np.ndarray
    __returns: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 discretizer: Discretizer):
        RLAlgorithm.__init__(self, "info.log")
        self.__env = env
        self.__discount_factor = discount_factor
        self.__discretizer = discretizer
        self.__q_table = np.random.random(
            (discretizer.n_bins + (self.__env.action_space.n,)))
        self.__returns = np.empty(self.__q_table.shape, dtype=object)

        self._logger.info(self)

    def train(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0.1

            samples = []
            done = False
            state = self.__env.reset()
            state = self.__discretizer.discretize(state)
            while not done:
                if random.random() <= epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.__q_table[state])

                state, reward, done, _ = self.__env.step(action)
                state = self.__discretizer.discretize(state)
                episode_reward += reward
                episode_actions += 1

                samples.append((state, action, reward))

            self._logger.info(f"episode={episode_i}|reward={episode_reward}|"
                              f"actions={episode_actions}")

            return_ = 0
            processed_samples = []
            for sample in reversed(samples):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                return_ = self.__discount_factor * return_ + reward

                if (state, action) in processed_samples:
                    continue

                processed_samples.append((state, action))
                if self.__returns[state + (action,)] is None:
                    self.__returns[state + (action,)] = [return_]
                else:
                    self.__returns[state + (action,)].append(return_)

                self.__q_table[state + (action,)] = (
                    np.mean(self.__returns[state + (action,)]))

    def run(self, n_episodes: int, render: bool = False) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                state = self.__discretizer.discretize(state)
                action = np.argmax(self.__q_table[state])
                state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1

            self._logger.info(f"episode={episode_i}|reward={episode_reward}|"
                              f"actions={episode_actions}")

    def __str__(self) -> str:
        return ("Tabular Monte Carlo: "
                f"discount factor = {self.__discount_factor}|"
                f"{self.__discretizer}")
