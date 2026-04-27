"""Shared registry primitives.

This module deliberately imports no concrete KGE-JAXed components. Keeping the
registry definitions dependency-light prevents circular imports between model
classes and registry population.
"""

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


class Registry[T]:
    """Map user-facing names and aliases to registered objects."""

    def __init__(self, component_type: str) -> None:
        self.component_type = component_type
        self._items: dict[str, T] = {}
        self._loader: Callable[[], None] | None = None
        self._loading = False

    def set_loader(self, loader: Callable[[], None]) -> None:
        """Attach a lazy population hook."""
        self._loader = loader

    def register(self, name: str, value: T, *, aliases: Iterable[str] = ()) -> T:
        """Register a value under a canonical name and optional aliases."""
        self._register_one(name, value)
        for alias in aliases:
            self._register_one(alias, value)
        return value

    def get(self, name: str) -> T:
        """Return a registered value by name or alias."""
        self._ensure_loaded()
        if name not in self._items:
            raise ValueError(f"Unknown {self.component_type} '{name}'. Available: {self.names()}")
        return self._items[name]

    def build(self, name: str, **kwargs):
        """Call the registered value with keyword arguments."""
        factory = self.get(name)
        if not callable(factory):
            raise TypeError(f"Registered {self.component_type} '{name}' is not callable")
        return factory(**kwargs)

    def names(self) -> list[str]:
        """Return registered names and aliases in registration order."""
        self._ensure_loaded()
        return list(self._items.keys())

    def as_dict(self, *, load: bool = True) -> dict[str, T]:
        """Return a shallow copy of the registry mapping."""
        if load:
            self._ensure_loaded()
        return dict(self._items)

    def _ensure_loaded(self) -> None:
        if self._loader is None or self._loading:
            return
        self._loading = True
        try:
            self._loader()
        finally:
            self._loading = False

    def _register_one(self, name: str, value: T) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.component_type} registry names must be non-empty strings")

        existing = self._items.get(name)
        if existing is not None and existing is not value:
            raise ValueError(f"{self.component_type} '{name}' is already registered")
        self._items[name] = value


models: Registry[type] = Registry("model")
losses: Registry[Callable] = Registry("loss")
negative_samplers: Registry[Callable] = Registry("negative sampler")
optimizers: Registry[Callable] = Registry("optimizer")
regularizers: Registry[type] = Registry("regularizer")
constrainers: Registry[Callable] = Registry("constrainer")
initializers: Registry[Callable] = Registry("embedding initializer")
