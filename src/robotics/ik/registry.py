"""Process-wide name → IK class. One registry; many controller *instances*."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics.ik.base import IKController


class IKRegistry:
    """Singleton table of IK solver names.

    ``IKRegistry()`` and ``IKRegistry.instance()`` are the same object.
    Controllers themselves are *not* singletons — each ``make_ik`` call
    builds a fresh instance with its own last ``dq``.
    """

    _instance: IKRegistry | None = None

    def __new__(cls) -> IKRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._solvers: dict[str, type[IKController]] = {}
            cls._instance = inst
        return cls._instance

    @classmethod
    def instance(cls) -> IKRegistry:
        return cls()

    def add(self, name: str, solver: type[IKController]) -> None:
        if name in self._solvers:
            raise ValueError(f"IK solver already registered: {name!r}")
        self._solvers[name] = solver

    def get(self, name: str) -> type[IKController]:
        try:
            return self._solvers[name]
        except KeyError as exc:
            raise KeyError(f"unknown IK solver {name!r}; know {sorted(self)}") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._solvers

    def __iter__(self):
        return iter(self._solvers)

    def __len__(self) -> int:
        return len(self._solvers)


def register(name: str):
    """Class decorator: ``@register("dls")`` on an :class:`IKController` subclass."""

    def decorator(cls: type[IKController]) -> type[IKController]:
        IKRegistry.instance().add(name, cls)
        return cls

    return decorator


def make_ik(name: str, **kwargs) -> IKController:
    return IKRegistry.instance().get(name)(**kwargs)
