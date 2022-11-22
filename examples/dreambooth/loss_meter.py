from collections import UserDict
from shared import TrainingPair


class LossMeter(UserDict):
    def add_loss_for_pair(self, pair: TrainingPair, loss: float):
        if pair not in self.data:
            self[pair] = [0]

        self[pair].append(loss)

    def get_pairs_with_highest_losses(self,
                                      num_pairs=0) -> list[TrainingPair]:
        if num_pairs == 0:
            num_pairs = 10

        return sorted(self,
                      key=self.get,
                      reverse=True
                      )[:round(num_pairs)]


a = TrainingPair(1, 1)
b = TrainingPair(2, 2)
c = TrainingPair(3, 3)


l = [a, b, c]

lm = LossMeter({pair: 0 for pair in l})
lm.add_loss_for_pair(TrainingPair(5, 5), 0.1)
print(lm.data)
