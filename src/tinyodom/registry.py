from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ComponentRegistry(Generic[T]):
    """Registry for explicitly named component classes.

    Parameters
    ----------
    component_kind : str
        Human-readable component label used in error messages.
    """

    component_kind: str
    _items: dict[str, type[T]] = field(default_factory=dict, init=False, repr=False)

    def register(self, name: str, component_cls: type[T]) -> None:
        """Register a component class under a stable string key.

        Parameters
        ----------
        name : str
            Stable registration key.
        component_cls : type[T]
            Component class associated with ``name``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``name`` is empty or already registered.
        """

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError(f"{self.component_kind} registration names must be non-empty.")
        if normalized_name in self._items:
            raise ValueError(
                f"{self.component_kind} '{normalized_name}' is already registered."
            )
        self._items[normalized_name] = component_cls

    def get(self, name: str) -> type[T]:
        """Return the component class registered under ``name``.

        Parameters
        ----------
        name : str
            Stable registration key.

        Returns
        -------
        type[T]
            Registered component class.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """

        normalized_name = name.strip()
        try:
            return self._items[normalized_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown {self.component_kind} '{normalized_name}'. "
                f"Known {self.component_kind}s: {', '.join(self.names()) or '<none>'}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered names in sorted order.

        Returns
        -------
        tuple[str, ...]
            Sorted registered component names.
        """

        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is registered.

        Parameters
        ----------
        name : object
            Potential registration key.

        Returns
        -------
        bool
            ``True`` when ``name`` is a registered string key.
        """

        return isinstance(name, str) and name in self._items

    def __getitem__(self, name: str) -> type[T]:
        """Proxy item access to :meth:`get`.

        Parameters
        ----------
        name : str
            Stable registration key.

        Returns
        -------
        type[T]
            Registered component class.
        """

        return self.get(name)


dataset_registry: ComponentRegistry[object] = ComponentRegistry("dataset")
task_registry: ComponentRegistry[object] = ComponentRegistry("task")
model_family_registry: ComponentRegistry[object] = ComponentRegistry("model family")
