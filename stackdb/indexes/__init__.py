from .base_index import BaseIndex
from .flat_index import FlatIndex
from .lsh_index import LSHIndex
from .ivf_index import IVFIndex
from typing import Dict


def get_index(index_type: str, dimension: int, index_params: Dict):
    if index_type == "flat":
        return FlatIndex(dimension=dimension, **index_params)
    elif index_type == "lsh":
        return LSHIndex(dimension=dimension, **index_params)
    elif index_type == "ivf":
        return IVFIndex(dimension=dimension, **index_params)
    else:
        raise ValueError(f"Invalid index type: {index_type}")


__all__ = ["BaseIndex", "FlatIndex", "LSHIndex", "IVFIndex", "get_index"]
