#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""app.core package.

Keep package import side effects minimal; callers should import the
submodules they actually use.
"""

from importlib import import_module

__all__ = [
    "config",
    "file_upload",
    "headers",
    "http_client",
    "models",
    "openai",
    "request_signing",
    "response_handler",
    "retry_policy",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
