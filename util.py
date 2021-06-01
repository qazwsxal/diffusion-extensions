import torch
from collections.abc import Iterable

def to_device(device, *objects, non_blocking=False):
    gpu_objects = []
    for object in objects:
        if isinstance(object, torch.Tensor):
            gpu_objects.append(object.to(device, non_blocking=non_blocking))
        elif isinstance(object, Iterable):
            gpu_objects.append(to_device(device, *object))
        else:
            raise RuntimeError(f"Cannot move object of type {type(object)} to {device}")
    return gpu_objects
