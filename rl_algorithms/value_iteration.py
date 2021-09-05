from rl_algorithms.rl_algorithm import RLAlgorithm


class ValueIteration(RLAlgorithm):

    def __init__(self) -> None:
        RLAlgorithm.__init__(self)

    def train(self, training_episodes: int) -> None:
        pass

    def run(self, episodes: int, render: bool) -> None:
        pass
