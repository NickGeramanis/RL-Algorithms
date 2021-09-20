import math
import random

import numpy as np
from gym import Env

from src.features.discretizer import Discretizer
from src.rl_algorithms.rl_algorithm import RLAlgorithm


class TabularQLearning(RLAlgorithm):
    __env: Env
    __discount_factor: float
    __initial_learning_rate: float
    __learning_rate_midpoint: int
    __learning_rate_steepness: float
    __discretizer: Discretizer
    __q_table: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_midpoint: int,
                 learning_rate_steepness: float,
                 discretizer: Discretizer) -> None:
        RLAlgorithm.__init__(self, "info.log")
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__learning_rate_steepness = learning_rate_steepness
        self.__discretizer = discretizer
        self.__q_table = np.random.random(
            (discretizer.n_bins + (self.__env.action_space.n,)))

        self._logger.info(self)

    def train(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                exponent = (self.__learning_rate_steepness
                            * (episode_i - self.__learning_rate_midpoint))
                learning_rate = (self.__initial_learning_rate
                                 / (1 + math.exp(exponent)))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            observation = self.__env.reset()
            current_state = self.__discretizer.discretize(observation)

            while not done:
                if random.random() <= epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(self.__q_table[current_state])

                next_state, reward, done, _ = self.__env.step(action)
                next_state = self.__discretizer.discretize(next_state)
                episode_reward += reward
                episode_actions += 1

                td_target = reward
                if not done:
                    td_target += (self.__discount_factor
                                  * max(self.__q_table[next_state]))

                td_error = (td_target
                            - self.__q_table[current_state + (action,)])
                self.__q_table[current_state + (action,)] += (
                        learning_rate * td_error)

                current_state = next_state

            self._logger.info(f"episode={episode_i}|reward={episode_reward}"
                              f"|actions={episode_actions}")

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

            self._logger.info(f"episode={episode_i}|reward={episode_reward}"
                              f"|actions={episode_actions}")

    def __str__(self) -> str:
        return ("Tabular Q-Learning:"
                f"discount factor = {self.__discount_factor}|"
                f"initial learning rate = {self.__initial_learning_rate}|"
                f"learning rate midpoint = {self.__learning_rate_midpoint}|"
                f"learning rate steepness = {self.__learning_rate_steepness}|"
                f"{self.__discretizer}")
