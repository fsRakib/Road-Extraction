"""
Finds every model automatically.

Drop a new file in models/ that defines a class inheriting RoadModel,
and it becomes available immediately. No other file needs editing.
Files starting with '_' are ignored.
"""
import importlib
import pkgutil
from pathlib import Path

import models
from models._base import RoadModel


def discover():
    """Return {model_name: model_class} for every model file in models/."""
    found = {}
    pkg_dir = Path(models.__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"models.{info.name}")
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, RoadModel)
                    and obj is not RoadModel):
                found[obj.name] = obj
    return found
