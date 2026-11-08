"""Make our code available to Spark Python workers regardless of how it was installed.

Custom data sources and UDFs are cloudpickled on the driver and unpickled in worker
processes. If `tax_rag` isn't importable there (it isn't, unless the cluster env installed
the package), unpickling fails with `ModuleNotFoundError: No module named 'tax_rag'`.
Registering the package for pickle-by-value ships the code with each task instead.
"""

from __future__ import annotations

import contextlib
import logging

log = logging.getLogger(__name__)


def ship_package_by_value() -> bool:
    try:
        from pyspark import cloudpickle
    except ImportError:  # pragma: no cover
        log.warning("pyspark.cloudpickle unavailable; relying on the package being installed")
        return False
    import tax_rag

    with contextlib.suppress(ValueError):  # already registered
        cloudpickle.register_pickle_by_value(tax_rag)
    return True
