import enum

class Explanationtype(enum.Enum):
    OUTPUT_PROBABILITY = 1,
    ATTENTION_WEIGHTS = 2,
    LIME = 3,
    ALL = 4