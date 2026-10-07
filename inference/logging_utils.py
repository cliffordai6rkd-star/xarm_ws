"""Hide routine hardware banners while preserving warnings and errors."""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import logging


class _WarningsOnly(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


@contextmanager
def quiet_hardware_output(enabled):
    if not enabled:
        yield
        return
    # The SDK prints these banners directly instead of using logging.
    original_print = builtins.print
    prefixes = ("SDK_VERSION:", "ROBOT_IP:", "change protocol identifier to ")
    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith(prefixes):
            return
        original_print(*args, **kwargs)
    filter_ = _WarningsOnly()
    loggers = [logging.getLogger(name) for name in
               ("nero_collection.cameras", "origin.print", "websockets.client")]
    for logger in loggers:
        logger.addFilter(filter_)
    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print
        for logger in loggers:
            logger.removeFilter(filter_)
