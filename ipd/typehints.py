from typing import *  # noqa
import numpy as np

KW = dict[str, Any]

class Frames44(np.ndarray):

    def __instancecheck__(self, obj):
        return hasattr(self, 'shape') and obj.shape[-2:] == (4, 4)

class FramesN44(np.ndarray):

    def __instancecheck__(self, obj):
        return hasattr(self, 'shape') and len(obj.shape) == 3 and obj.shape[-2:] == (4, 4)
