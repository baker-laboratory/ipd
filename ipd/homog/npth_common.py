import sys
import ipd

th, xr, np = ipd.lazyimports('torch', 'xarray', 'numpy')

if ipd.not_installed('torch'): Tensor = np.ndarray
else: Tensor = ipd.Union[np.ndarray, "th.Tensor"]

def splitlastdim(x, i):
    return x[..., :i], x[..., i:]

def get_tensor_libraries_for(x):
    h, npth = (ipd.homog.hgeom, np) if isinstance(x, np.ndarray) else (ipd.h, th)
    return h, npth

def is_tensor(thing):
    return any([
        isinstance(thing, np.ndarray),
        'torch' in sys.modules and th.is_tensor(thing),
        'xarray' in sys.modules and isinstance(thing, xr.DataArray),
    ])

def is_xform_stack(thing):
    return is_tensor(thing) and thing.ndim == 3 and thing.shape[-2:] == (4, 4)

def numel(tensor):
    if 'torch' in sys.modules and th.is_tensor(tensor): return tensor.numel
    return tensor.size

def tensor_summary(tensor, maxnumel=24):
    if numel(tensor) <= maxnumel: return str(tensor)
    return f'{tensor.__class__.__name__}{list(tensor.shape)}'

def all_lte(a, b):
    return np.all(np.asarray(a) <= np.asarray(b))

def np_array(val):
    try:
        return np.asarray(val)
    except ValueError:
        return np.asarray(val, dtype=object)
