import math
import random

import numpy as np
from gym import Env

from src.features.feature_constructor import FeatureConstructor
from src.rl_algorithms.rl_algorithm import RLAlgorithm


class LFAQLearning(RLAlgorithm):
    __env: Env
    __discount_factor: float
    __initial_learning_rate: float
    __learning_rate_midpoint: int
    __learning_rate_steepness: float
    __feature_constructor: FeatureConstructor
    __weights: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_midpoint: int,
                 learning_rate_steepness: float,
                 feature_constructor: FeatureConstructor) -> None:
        RLAlgorithm.__init__(self, "info.log")
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__learning_rate_steepness = learning_rate_steepness
        self.__feature_constructor = feature_constructor
        self.__weights = np.random.random((feature_constructor.n_features,))

        self._logger.info(self)

    def train(self, n_episodes: int) -> None:
        for i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                exponent = (self.__learning_rate_steepness
                            * (i - self.__learning_rate_midpoint))
                learning_rate = (self.__initial_learning_rate
                                 / (1 + math.exp(exponent)))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            current_state = self.__env.reset()
            current_q = self.__feature_constructor.calculate_q(self.__weights,
                                                               current_state)

            while not done:
                if random.random() <= epsilon:
                    action = self.__env.action_space.sample()
                else:
                    action = np.argmax(current_q)

                next_state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1
                next_q = self.__feature_constructor.calculate_q(self.__weights,
                                                                next_state)

                td_target = reward
                if not done:
                    td_target += self.__discount_factor * np.max(next_q)

                td_error = td_target - current_q[action]

                features = self.__feature_constructor.get_features(
                    current_state,
                    action)
                self.__weights += learning_rate * td_error * features

                current_state = next_state
                current_q = next_q

            self._logger.info(f'episode={i}|reward={episode_reward}'
                              f'|actions={episode_actions}')

    def run(self, n_episodes: int, render: bool = False) -> None:
        for i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                q = self.__feature_constructor.calculate_q(self.__weights,
                                                           state)
                action = np.argmax(q)
                state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1

            self._logger.info(f'episode={i}|reward={episode_reward}'
                              f'|actions={episode_actions}')

    def __str__(self) -> str:
        return ("Q-Learning with Linear Function Approximation:"
                f"discount factor={self.__discount_factor}|"
                f"initial learning rate = {self.__initial_learning_rate}|"
                f"learning rate midpoint = {self.__learning_rate_midpoint}|"
                f"learning rate steepness = {self.__learning_rate_steepness}|"
                f"{self.__feature_constructor}")
