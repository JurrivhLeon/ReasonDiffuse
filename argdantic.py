"""Minimal local subset of argdantic used by the copied dataset scripts."""

from __future__ import annotations

import argparse
from types import UnionType
from typing import Any, Callable, Optional, Union, get_args, get_origin


class ArgParser:
    def __init__(self) -> None:
        self._command: Callable[..., Any] | None = None

    def command(self, singleton: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._command = fn
            return fn

        return decorator

    def __call__(self) -> Any:
        if self._command is None:
            raise RuntimeError("No command registered")
        annotations = self._command.__annotations__
        if len(annotations) != 1:
            raise RuntimeError("This compatibility shim supports one config argument")
        config_cls = next(iter(annotations.values()))
        parser = argparse.ArgumentParser()

        for name, field in _iter_model_fields(config_cls):
            annotation = _field_annotation(field)
            default = _field_default(field)
            required = default is ...
            arg_type, nargs = _argparse_type(annotation)

            flags = {"dest": name, "required": required}
            if nargs is not None:
                flags["nargs"] = nargs
            if not required:
                flags["default"] = default

            if arg_type is bool:
                if default is True:
                    parser.add_argument(f"--no-{name.replace('_', '-')}", action="store_false", dest=name)
                else:
                    parser.add_argument(f"--{name.replace('_', '-')}", action="store_true", dest=name)
            else:
                parser.add_argument(f"--{name.replace('_', '-')}", type=arg_type, **flags)

        return self._command(config_cls(**vars(parser.parse_args())))


def _iter_model_fields(config_cls: type) -> list[tuple[str, Any]]:
    if hasattr(config_cls, "model_fields"):
        return list(config_cls.model_fields.items())
    return list(config_cls.__fields__.items())


def _field_annotation(field: Any) -> Any:
    return getattr(field, "annotation", getattr(field, "outer_type_", str))


def _field_default(field: Any) -> Any:
    if getattr(field, "is_required", lambda: False)():
        return ...
    default = getattr(field, "default", ...)
    if str(default) == "PydanticUndefined":
        return ...
    return default


def _argparse_type(annotation: Any) -> tuple[type, str | None]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        return _argparse_type(non_none[0] if non_none else str)

    if origin in (list, tuple):
        item_type = args[0] if args else str
        return item_type, "+"

    if annotation in (str, int, float, bool):
        return annotation, None

    return str, None
