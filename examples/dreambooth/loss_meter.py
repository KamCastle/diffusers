from collections import UserDict
from shared import TrainingPair


class LossMeter(UserDict):
    def add_loss_for_pair(pair: TrainingPair):
        pass
