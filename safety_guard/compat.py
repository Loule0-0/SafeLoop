from __future__ import annotations


def torch_load_compat(path, torch_load):
    try:
        return torch_load(path, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch_load(path)
