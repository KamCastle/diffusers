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

        self._internal_init()

    def get_next_pair(self) -> TrainingPair:
        self._last_pair_index += 1

        return self.get_last_pair()

    def get_last_pair(self) -> TrainingPair:
        return self._pairs[self._last_pair_index]

    def get_pairs_for_caching(
            self,
            pairs_ordered_by_loss_desc: list[TrainingPair]
    ) -> list[TrainingPair]:

        list_start = self._last_pair_index
        list_end = self._last_pair_index + self._num_instance_imgs

        if len(pairs_ordered_by_loss_desc) > 0:
            num_retrain_pairs = \
                self._get_number_of_retrain_pairs(pairs_ordered_by_loss_desc)

            retrain_pairs = pairs_ordered_by_loss_desc[0:num_retrain_pairs]
            self._pairs[
                list_start:list_start] = retrain_pairs

        result = self._pairs[list_start:list_end]
        if (list_end >= len(self._pairs)) or (list_start >= (len(self._pairs))):
            self._internal_init()

        return result

    def get_all_pairs(self):
        return self._pairs

    def get_loss_dict(self):
        return {pair: [0] for pair in self._pairs}

    def _build_cross_product_pairs(self) -> list[TrainingPair]:
        result = [TrainingPair(instance_index, class_index)
                  for instance_index in range(self._num_instance_imgs)
                  for class_index in range(self._num_class_imgs)]

        return result

    def _get_number_of_retrain_pairs(
            self,
            trained_pairs: list[TrainingPair]
    ) -> int:
        if self._retrain_percentage == 0:
            return 0
        else:
            return round(len(trained_pairs) / 100 * self._retrain_percentage)

    def _internal_init(self):
        self._pairs = self._build_cross_product_pairs()
        self._last_pair_index = 0
