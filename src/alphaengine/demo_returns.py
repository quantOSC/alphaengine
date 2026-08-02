"""A built-in project module: one name's return series, for `size_position` and `monitor_sleeve`.

    alphaengine run <workflow> --project alphaengine.demo_returns

`--project` reads `data` off a module, and the three non-sweep workflows want
different SHAPES of it. Rather than teach the loader which workflow wants which
— workflow knowledge does not belong in the client, and that rule is what keeps
the loop honest — each shape gets a module of its own and the caller names the
one they want.

Everything here is a fixed seed. Offline, reproducible, no data licence.
"""

from __future__ import annotations

from .demo import returns as data

__all__ = ["data"]
