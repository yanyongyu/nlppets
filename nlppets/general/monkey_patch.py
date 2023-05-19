import os
import sys
import inspect
import importlib
from contextlib import suppress, contextmanager
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Optional,
    Generator,
    MutableMapping,
    overload,
)

K = TypeVar("K")
V = TypeVar("V")


class Notset:
    def __repr__(self) -> str:
        return "<notset>"


notset = Notset()


def resolve(name: str) -> object:
    parts = name.split(".")
    used = parts.pop(0)
    found: object = importlib.import_module(used)
    for part in parts:
        used += "." + part
        try:
            found = getattr(found, part)
        except AttributeError:
            pass
        else:
            continue

        try:
            importlib.import_module(used)
        except ImportError as ex:
            expected = str(ex).split()[-1]
            if expected == used:
                raise
            else:
                raise ImportError(f"import error in {used}: {ex}") from ex
        found = annotated_getattr(found, part, used)
    return found


def annotated_getattr(obj: object, name: str, ann: str) -> object:
    try:
        obj = getattr(obj, name)
    except AttributeError as e:
        raise AttributeError(
            f"{type(obj).__name__!r} object at {ann} has no attribute {name!r}"
        ) from e
    return obj


def derive_importpath(import_path: str, raising: bool) -> Tuple[str, object]:
    if not isinstance(import_path, str) or "." not in import_path:
        raise TypeError(f"must be absolute import path string, not {import_path!r}")
    module, attr = import_path.rsplit(".", 1)
    target = resolve(module)
    if raising:
        annotated_getattr(target, attr, ann=module)
    return attr, target


class MonkeyPatch:
    def __init__(self) -> None:
        self._setattr: List[Tuple[object, str, object]] = []
        self._setitem: List[Tuple[Mapping[Any, Any], object, object]] = []

    @classmethod
    @contextmanager
    def context(cls) -> Generator["MonkeyPatch", None, None]:
        m = cls()
        try:
            yield m
        finally:
            m.undo()

    @overload
    def setattr(
        self,
        target: str,
        name: object,
        value: Notset = ...,
        raising: bool = ...,
    ) -> None:
        ...

    @overload
    def setattr(
        self,
        target: object,
        name: str,
        value: object,
        raising: bool = ...,
    ) -> None:
        ...

    def setattr(
        self,
        target: Union[str, object],
        name: Union[object, str],
        value: object = notset,
        raising: bool = True,
    ) -> None:
        if isinstance(value, Notset):
            if not isinstance(target, str):
                raise TypeError(
                    "use setattr(target, name, value) or "
                    "setattr(target, value) with target being a dotted "
                    "import string"
                )
            value = name
            name, target = derive_importpath(target, raising)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    "use setattr(target, name, value) with name being a string or "
                    "setattr(target, value) with target being a dotted "
                    "import string"
                )
        oldval = getattr(target, name, notset)
        if raising and oldval is notset:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        # avoid class descriptors like staticmethod/classmethod
        if inspect.isclass(target):
            oldval = target.__dict__.get(name, notset)
        self._setattr.append((target, name, oldval))
        setattr(target, name, value)

    def delattr(
        self,
        target: Union[object, str],
        name: Union[str, Notset] = notset,
        raising: bool = True,
    ) -> None:
        __tracebackhide__ = True
        import inspect

        if isinstance(name, Notset):
            if not isinstance(target, str):
                raise TypeError(
                    "use delattr(target, name) or "
                    "delattr(target) with target being a dotted "
                    "import string"
                )
            name, target = derive_importpath(target, raising)
        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
        else:
            oldval = getattr(target, name, notset)
            if inspect.isclass(target):
                oldval = target.__dict__.get(name, notset)
            self._setattr.append((target, name, oldval))
            delattr(target, name)

    def setitem(self, dic: Mapping[K, V], name: K, value: V) -> None:
        """Set dictionary entry ``name`` to value."""
        self._setitem.append((dic, name, dic.get(name, notset)))
        dic[name] = value  # type: ignore

    def delitem(self, dic: Mapping[K, V], name: K, raising: bool = True) -> None:
        if name not in dic:
            if raising:
                raise KeyError(name)
        else:
            self._setitem.append((dic, name, dic.get(name, notset)))
            del dic[name]  # type: ignore

    def setenv(self, name: str, value: str, prepend: Optional[str] = None) -> None:
        if prepend and name in os.environ:
            value = value + prepend + os.environ[name]
        self.setitem(os.environ, name, value)

    def delenv(self, name: str, raising: bool = True) -> None:
        environ: MutableMapping[str, str] = os.environ
        self.delitem(environ, name, raising=raising)

    def undo(self) -> None:
        for obj, name, value in reversed(self._setattr):
            if value is not notset:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self._setattr[:] = []
        for dictionary, key, value in reversed(self._setitem):
            if value is notset:
                with suppress(KeyError):
                    del dictionary[key]  # type: ignore
            else:
                dictionary[key] = value  # type: ignore
        self._setitem[:] = []
