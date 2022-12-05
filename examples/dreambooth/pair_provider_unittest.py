import unittest
from shared import TrainingPair
from pair_provider import PairProvider


HIGHEST_LOSS_PAIRS = [TrainingPair(2, 2),
                      TrainingPair(2, 3),
                      TrainingPair(1, 4)]

LOSS_PAIRS = [TrainingPair(2, 2),
              TrainingPair(2, 3),
              TrainingPair(1, 4),
              TrainingPair(7, 22),
              TrainingPair(13, 11)]

ONE_INST_ONE_CLASS_0_PCT_RETRAIN = (1, 1, 0)
EXPECTED_PAIR_1_1_0 = TrainingPair(0, 0)
EXPECTED_PAIRS_1_1_0 = [EXPECTED_PAIR_1_1_0]


THREE_INST_THREE_CLASS_0_PCT_RETRAIN = (3, 3, 0)
EXPECTED_PAIRS_3_3_0 = [TrainingPair(0, 0),
                        TrainingPair(0, 1),
                        TrainingPair(0, 2)]

THREE_INST_THREE_CLASS_30_PCT_RETRAIN = (3, 3, 30)
EXPECTED_PAIRS_3_3_30 = [TrainingPair(2, 2),
                         TrainingPair(0, 0),
                         TrainingPair(0, 1)]

THREE_INST_THREE_CLASS_100_PCT_RETRAIN = (3, 3, 100)
TWENTY_INST_200_CLASS_10_PCT_RETRAIN = (20, 200, 10)
TWENTY_INST_200_CLASS_100_PCT_RETRAIN = (20, 200, 100)

LAST_PAIR_INDEX_1000 = 1000
EXPECTED_LAST_PAIR_INDEX_1000_20_INST_200_CLASS = TrainingPair(5, 0)


class TestPairProviderGetPairsForCaching(unittest.TestCase):
    def test_last_index_near_to_number_of_pairs(self):
        pp = PairProvider(*TWENTY_INST_200_CLASS_100_PCT_RETRAIN)
        pp._last_pair_index = 3998
        len_pairs = len(pp.get_pairs_for_caching(LOSS_PAIRS))
        expected_len = 7
        self.assertEqual(len_pairs, expected_len)


class TestPairProvider(unittest.TestCase):
    def test_get_next_pair(self):
        pp = PairProvider(*TWENTY_INST_200_CLASS_10_PCT_RETRAIN)
        next_pair = pp.get_next_pair()

        self.assertEqual(next_pair, TrainingPair(0, 1))

    def test_get_all_pairs(self):
        pp = PairProvider(*ONE_INST_ONE_CLASS_0_PCT_RETRAIN)

        self.assertEqual(pp.get_all_pairs(), EXPECTED_PAIRS_1_1_0)

    def test_get_last_pair(self):
        pp = PairProvider(*ONE_INST_ONE_CLASS_0_PCT_RETRAIN)

        self.assertEqual(pp.get_last_pair(), EXPECTED_PAIR_1_1_0)

    def test_3_inst_3_class_0_retrain_get_pairs_for_caching(self):
        pp = PairProvider(*THREE_INST_THREE_CLASS_0_PCT_RETRAIN)
        pairs_for_caching = pp.get_pairs_for_caching(HIGHEST_LOSS_PAIRS)

        self.assertEqual(pairs_for_caching, EXPECTED_PAIRS_3_3_0)

    def test_3_inst_3_class_50_retrain_get_pairs_for_caching(self):
        pp = PairProvider(*THREE_INST_THREE_CLASS_30_PCT_RETRAIN)
        pairs_for_caching = pp.get_pairs_for_caching(HIGHEST_LOSS_PAIRS)

        self.assertEqual(pairs_for_caching, EXPECTED_PAIRS_3_3_30)

    def test_3_inst_3_class_100_retrain_get_pairs_for_caching(self):
        pp = PairProvider(*THREE_INST_THREE_CLASS_100_PCT_RETRAIN)
        pairs_for_caching = pp.get_pairs_for_caching(HIGHEST_LOSS_PAIRS)

        self.assertEqual(pairs_for_caching, HIGHEST_LOSS_PAIRS)

    def test_20_inst_200_class_10_retrain_get_number_of_retrain_pairs(self):
        pp = PairProvider(*TWENTY_INST_200_CLASS_10_PCT_RETRAIN)
        num_retrain_pairs = pp._get_number_of_retrain_pairs(HIGHEST_LOSS_PAIRS)

        self.assertEqual(0, num_retrain_pairs)

    def test_20_inst_200_class_10_retrain_get_all_pairs(self):
        pp = PairProvider(*TWENTY_INST_200_CLASS_10_PCT_RETRAIN)
        num_all_pairs = len(pp.get_all_pairs())

        self.assertEqual(4000, num_all_pairs)

    def test_20_inst_200_class_10_retrain_index_400_get_all_pairs(self):
        pp = PairProvider(*TWENTY_INST_200_CLASS_10_PCT_RETRAIN)
        pp._last_pair_index = LAST_PAIR_INDEX_1000
        last_pair = pp.get_last_pair()

        self.assertEqual(EXPECTED_LAST_PAIR_INDEX_1000_20_INST_200_CLASS,
                         last_pair)


if __name__ == '__main__':
    unittest.main()
