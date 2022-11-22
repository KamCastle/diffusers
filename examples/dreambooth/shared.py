from collections import namedtuple
from enum import Enum, auto


TrainingPair = namedtuple("TrainingPair",
                          ["instance_image_index", "class_image_index"])


class DataLoaderType(Enum):
    CACHED_LATENTS = auto()
    REGULAR = auto()
