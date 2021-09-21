import logging
from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    _logger: logging.Logger

    def __init__(self, lof_filename: str) -> None:
        self.__setup_logger(lof_filename)

    def __setup_logger(self, log_filename: str) -> None:
        self._logger = logging.getLogger(__name__)

        if not self._logger.hasHandlers():
            log_formatter = logging.Formatter(
                fmt='%(asctime)s %(levelname)s %(message)s',
                datefmt='%d-%m-%Y %H:%M:%S')

            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self._logger.addHandler(console_handler)

            self._logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, n_episodes: int) -> None:
        pass

    @abstractmethod
    def run(self, n_episodes: int, render: bool = False) -> None:
        pass
