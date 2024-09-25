import numpy as np

def guess_symmetry(xyz):
    'ca only, nchain x nres x p'
    match len(xyz):
        case 1:
            return 'C1'
        case 2:
            return 'C2'
        case 3:
            return 'C3'
        case 4:
            return 'C4'
        case 5:
            return 'C5'
        case 6:
            return 'C6'
        case 7:
            return 'C7'
        case 8:
            return 'C8'
        case 9:
            return 'C9'
        case 10:
            return 'C10'
        case 11:
            return 'C11'
        case 12:
            return 'tet'
        case 24:
            return 'oct'
        case 30:
            return 'icos'
        case 60:
            return 'icos'
        case _:
            return 'unknown'
