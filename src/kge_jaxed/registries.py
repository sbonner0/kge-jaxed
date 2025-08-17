from typing import Any, Callable, Dict


class Registry:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        """
        Decorator to register a class or function.

        Usage:
            @MODELS.register("transe")
            class TransE(...):
                ...

            @LOSSES.register("mrl")
            def margin_ranking_loss(...):
                ...
        """

        def deco(obj: Any) -> Any:
            key = name or obj.__name__.lower()
            if key in self._store:
                raise ValueError(f"Duplicate key '{key}' in registry")
            self._store[key] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        if name not in self._store:
            raise KeyError(f"{name} not found. Available: {list(self._store)}")
        return self._store[name]

    def available(self) -> list[str]:
        return list(self._store.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self):
        return iter(self._store.items())


MODELS = Registry()
LOSSES = Registry()
NEGATIVE_SAMPLERS = Registry()
