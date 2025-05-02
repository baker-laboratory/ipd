import copy


def shallow_copy(obj):
    origcopy = getattr(obj.__class__, '__copy__', None)
    try:
        if hasattr(obj.__class__, '__copy__'):
            delattr(obj.__class__, '__copy__')
        return copy.copy(obj)
    finally:
        if origcopy:
            setattr(obj.__class__, '__copy__', origcopy)
