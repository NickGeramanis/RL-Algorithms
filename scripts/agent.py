import gym
import numpy as np

from src.features.discretizer import Discretizer
from src.features.fourier_basis import FourierBasis
from src.features.polynomials import Polynomials
from src.features.radial_basis_functions import RadialBasisFunctions
from src.features.tile_coding import TileCoding
from src.rl_algorithms.lfa_q_lambda import LFAQLambda
from src.rl_algorithms.lfa_q_learning import LFAQLearning
from src.rl_algorithms.lfa_sarsa import LFASARSA
from src.rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from src.rl_algorithms.lspi import LSPI
from src.rl_algorithms.tabular_monte_carlo import TabularMonteCarlo
from src.rl_algorithms.tabular_q_lambda import TabularQLambda
from src.rl_algorithms.tabular_q_learning import TabularQLearning
from src.rl_algorithms.tabular_sarsa import TabularSARSA
from src.rl_algorithms.tabular_sarsa_lambda import TabularSARSALambda


def main():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    n_episodes = 100

    discount_factor = 0.99
    state_space_low = env.observation_space.low
    state_space_high = env.observation_space.high
    n_dimensions = len(state_space_low)

    initial_learning_rate = 0.1
    learning_rate_steepness = 0.01
    learning_rate_midpoint = 50
    lambda_ = 0.5

    n_bins = (20, 20)
    discretizer = Discretizer(n_bins, state_space_low, state_space_high)

    tabular_monte_carlo = TabularMonteCarlo(env, discount_factor, discretizer)
    tabular_monte_carlo.train(n_episodes)

    tabular_sarsa = TabularSARSA(env, discount_factor, initial_learning_rate,
                                 learning_rate_midpoint,
                                 learning_rate_steepness, discretizer)
    tabular_sarsa.train(n_episodes)

    tabular_q_learning = TabularQLearning(env, discount_factor,
                                          initial_learning_rate,
                                          learning_rate_midpoint,
                                          learning_rate_steepness, discretizer)
    tabular_q_learning.train(n_episodes)

    tabular_sarsa_lambda = TabularSARSALambda(env, discount_factor,
                                              initial_learning_rate,
                                              learning_rate_midpoint,
                                              learning_rate_steepness,
                                              discretizer, lambda_)
    tabular_sarsa_lambda.train(n_episodes)

    tabular_q_lambda = TabularQLambda(env, discount_factor,
                                      initial_learning_rate,
                                      learning_rate_midpoint,
                                      learning_rate_steepness, discretizer,
                                      lambda_)
    tabular_q_lambda.train(n_episodes)

    n_tiles_per_dimension = [14, 14]
    displacement_vector = np.array([1, 1])
    n_tilings = 8
    initial_learning_rate1 = 0.1 / n_tilings
    feature_constructor1 = TileCoding(env.action_space.n, n_tilings,
                                      n_tiles_per_dimension, state_space_low,
                                      state_space_high, displacement_vector)

    order = 2
    feature_constructor2 = Polynomials(env.action_space.n, order, n_dimensions)

    order = 2
    feature_constructor3 = FourierBasis(env.action_space.n, order,
                                        state_space_low, state_space_high)

    standard_deviation = 0.25
    centers_per_dimension = [
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8]
    ]
    feature_constructor4 = RadialBasisFunctions(env.action_space.n,
                                                state_space_low,
                                                state_space_high,
                                                centers_per_dimension,
                                                standard_deviation)

    lfa_sarsa = LFASARSA(env, discount_factor, initial_learning_rate1,
                         learning_rate_midpoint, learning_rate_steepness,
                         feature_constructor1)
    lfa_sarsa.train(n_episodes)

    lfa_q_learning = LFAQLearning(env, discount_factor, initial_learning_rate,
                                  learning_rate_midpoint,
                                  learning_rate_steepness,
                                  feature_constructor2)
    lfa_q_learning.train(n_episodes)

    lfa_sarsa_lambda = LFASARSALambda(env, discount_factor,
                                      initial_learning_rate,
                                      learning_rate_midpoint,
                                      learning_rate_steepness,
                                      feature_constructor3, lambda_)
    lfa_sarsa_lambda.train(n_episodes)

    lfa_q_lambda = LFAQLambda(env, discount_factor, initial_learning_rate,
                              learning_rate_midpoint, learning_rate_steepness,
                              feature_constructor4, lambda_)
    lfa_q_lambda.train(n_episodes)

    tolerance = 0
    delta = 0.1
    n_samples = 1000
    lspi = LSPI(env, discount_factor, feature_constructor2, tolerance, delta)
    lspi.gather_samples(n_samples)
    lspi.train(n_episodes)
    lspi.run(n_episodes)


if __name__ == '__main__':
    main()
