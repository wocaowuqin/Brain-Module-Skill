from __future__ import annotations
from typing import Any
from .exceptions import LayerViolationError

class _Registry:
    def __init__(self):
        # Singleton registries call ``__init__`` on every access; do not erase
        # decorators registered during module import.
        if not hasattr(self, "_items"):
            self._items = {}
    def register(self, name, factory):
        if name in self._items and self._items[name] is not factory: raise ValueError(f"duplicate registration: {name}")
        self._items[name] = factory; return factory
    def get(self, name, *args, **kwargs):
        if name not in self._items: raise KeyError(name)
        value = self._items[name]
        return value(*args, **kwargs) if isinstance(value, type) else value
    def names(self): return tuple(sorted(self._items))

class SkillRegistry(_Registry):
    _instance = None
    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls); cls._instance._items = {}
        return cls._instance

class ModuleRegistry(_Registry):
    _instance = None
    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls); cls._instance._items = {}
        return cls._instance

def register_skill(name):
    return lambda cls: SkillRegistry().register(name, cls)

def register_module(name):
    return lambda cls: ModuleRegistry().register(name, cls)
