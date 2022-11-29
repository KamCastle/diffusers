from collections import namedtuple
from enum import Enum, auto
from typing import Any


class TrainingObject:
    args: Any


class ImageType(Enum):
    CLASS = auto()
    INSTANCE = auto()


TrainingPair = namedtuple("TrainingPair",
                          ["instance_image_index", "class_image_index"])

ParsedConcepts = namedtuple('ParsedConcepts',
                            ['instance_images', 'class_images'])


class DataLoaderType(Enum):
    CACHED_LATENTS = auto()
    REGULAR = auto()
