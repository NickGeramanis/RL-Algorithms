from src.rl_algorithms.rl_algorithm import RLAlgorithm


class REINFORCE(RLAlgorithm):

    def __init__(self) -> None:
        super().__init__("info.log")

    def train(self, n_episodes: int) -> None:
        raise NotImplementedError

    def run(self, n_episodes: int, render: bool = False) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return "REINFORCE"
