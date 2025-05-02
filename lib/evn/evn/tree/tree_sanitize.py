from pathlib import Path
from enum import Enum
from datetime import datetime, date
from collections.abc import Mapping, Sequence
import evn


def sanitize(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return sorted(sanitize(x) for x in obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [sanitize(x) for x in obj]
    if isinstance(obj, Mapping):
        return {sanitize(k): sanitize(v) for k, v in obj.items()}
    if evn.installed.numpy:
        np = evn.lazyimport('numpy')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if evn.installed.torch:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    if evn.installed.biotite:
        from biotite.structure import AtomArray, AtomArrayStack

        if isinstance(obj, (AtomArray, AtomArrayStack)):
            return obj.as_array().tolist()
    if evn.installed.omegaconf:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    return repr(obj)
