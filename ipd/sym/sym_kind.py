import dataclasses
from enum import Enum

class ShapeKind(Enum):
    SPARSE = 7
    ONEDIM = 13
    TWODIM = 4135667696
    MAPPING = 2
    SEQUENCE = 345
    SCALAR = 186282

class ValueKind(Enum):
    PAIR = 26
    BASIC = 196883
    XYZ = 163
    INDEX = 691
    MIXED = 314159

@dataclasses.dataclass
class SymKind:
    shapekind: ShapeKind
    valuekind: ValueKind
