#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

logger = logging.getLogger("zai2api")
logger.setLevel(logging.INFO)
logger.propagate = False

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
))
logger.addHandler(_handler)


def setup_logger(debug_mode=False):
    """Reconfigure logger (used by admin hot-reload)."""
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        if debug_mode
        else "%(asctime)s | %(levelname)-8s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S" if debug_mode else "%H:%M:%S"
    for h in logger.handlers:
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
