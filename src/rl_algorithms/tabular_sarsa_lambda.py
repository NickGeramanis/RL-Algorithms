import math
import random

import numpy as np
from gym import Env

from src.features.discretizer import Discretizer
from src.rl_algorithms.rl_algorithm import RLAlgorithm


class TabularSARSALambda(RLAlgorithm):
    __env: Env
    __discount_factor: float
    __initial_learning_rate: float
    __learning_rate_midpoint: int
    __learning_rate_steepness: float
    __discretizer: Discretizer
    __lambda: float
    __q_table: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_midpoint: int,
                 learning_rate_steepness: float,
                 discretizer: Discretizer,
                 lambda_: float) -> None:
        RLAlgorithm.__init__(self, "info.log")
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__learning_rate_steepness = learning_rate_steepness
        self.__discretizer = discretizer
        self.__lambda = lambda_
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
            eligibility_traces = np.zeros(
                (self.__discretizer.n_bins + (self.__env.action_space.n,)))
            current_state = self.__env.reset()
            current_state = self.__discretizer.discretize(current_state)

            if random.random() <= epsilon:
                current_action = self.__env.action_space.sample()
            else:
                current_action = np.argmax(self.__q_table[current_state])

            while not done:
                next_state, reward, done, _ = self.__env.step(current_action)
                next_state = self.__discretizer.discretize(next_state)
                episode_reward += reward
                episode_actions += 1

                if random.random() <= epsilon:
                    next_action = self.__env.action_space.sample()
                else:
                    next_action = np.argmax(self.__q_table[next_state])

                td_target = reward
                if not done:
                    td_target += (
                            self.__discount_factor
                            * self.__q_table[next_state + (next_action,)])

                td_error = (
                        td_target
                        - self.__q_table[current_state + (current_action,)])
                eligibility_traces[current_state + (current_action,)] += 1

                self.__q_table += learning_rate * td_error * eligibility_traces
                eligibility_traces *= self.__discount_factor * self.__lambda

                current_state = next_state
                current_action = next_action

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
        return ("Tabular SARSA(lambda):"
                f"discount factor = {self.__discount_factor}|"
                f"initial learning rate = {self.__initial_learning_rate}"
                f"learning rate midpoint = {self.__learning_rate_midpoint}|"
                f"learning rate steepness = {self.__learning_rate_steepness}|"
                f"{self.__discretizer}|"
                f"lambda = {self.__lambda}")
