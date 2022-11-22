from shared import TrainingPair


class PairProvider:
    def __init__(
        self,
        num_instance_imgs: int,
        num_class_imgs: int,
        retrain_percentage: int
    ) -> None:
        self._num_instance_imgs = num_instance_imgs
        self._num_class_imgs = num_class_imgs
        self._retrain_percentage = retrain_percentage

        self._pairs = self._build_cross_product_pairs()
        self._last_pair = (0, 0)
        self._last_pair_index = 0

    def get_next_pair(self) -> TrainingPair:
        pass

    def get_pairs_for_caching(
            self,
            pairs_ordered_by_loss_desc: list[TrainingPair]
    ) -> list[TrainingPair]:

        list_start = self._last_pair_index
        list_end = self._last_pair_index + self._num_instance_imgs

        if len(pairs_ordered_by_loss_desc) == 0:
            return self._pairs[list_start:list_end]
        else:
            num_retrain_pairs = \
                self._get_number_of_retrain_pairs(pairs_ordered_by_loss_desc)

            retrain_pairs = pairs_ordered_by_loss_desc[0:num_retrain_pairs]
            self._pairs[
                self._last_pair_index:self._last_pair_index] = retrain_pairs

    def _build_cross_product_pairs(self) -> list[TrainingPair]:
        result = [TrainingPair(instance_index, class_index)
                  for instance_index in range(self._num_instance_imgs)
                  for class_index in range(self._num_class_imgs)]

        return result

    def _get_number_of_retrain_pairs(
            self,
            trained_pairs: list[TrainingPair]
    ) -> int:
        if self.retrain_percentage == 0:
            return 0
        else:
            return round(len(trained_pairs) / 100 * self._retrain_percentage)
