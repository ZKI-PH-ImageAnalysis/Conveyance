"""Simple registries for models, losses, and datasets."""
from __future__ import annotations

from typing import Callable, Dict

MODEL_REGISTRY: Dict[str, Callable] = {}
LOSS_REGISTRY: Dict[str, Callable] = {}
DATASET_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    def decorator(fn: Callable):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def register_loss(name: str):
    def decorator(fn: Callable):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def register_dataset(name: str):
    def decorator(fn: Callable):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def create_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def create_loss(name: str, **kwargs):
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


def create_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
