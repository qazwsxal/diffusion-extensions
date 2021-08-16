import torch
from collections.abc import Iterable
import inspect

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

def init_from_dict(argdict, *classes):
    objs = []
    for cls in classes:
        sig = inspect.signature(cls)
        args = [k for k, v in sig.parameters.items() if
                v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        class_kwargs = {k:v for k,v in argdict.items() if k in args}
        objs += cls(**class_kwargs)