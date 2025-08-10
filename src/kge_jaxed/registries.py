from typing import Any, Callable, Dict, Type


class Registry:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def deco(obj):
            key = name or obj.__name__.lower()
            if key in self._store:
                raise ValueError(f"Duplicate registry key: {key}")
            self._store[key] = obj
            return obj

        return deco

    def get(self, name: str):
        if name not in self._store:
            raise KeyError(f"{name} not found. Available: {list(self._store)}")
        return self._store[name]

    def __contains__(self, name):
        return name in self._store

    def __iter__(self):
        return iter(self._store.items())


MODELS = Registry()
LOSSES = Registry()
NEGATIVE_SAMPLERS = Registry()
