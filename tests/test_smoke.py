"""The public surface: what a user gets from `import alphaengine`.

These are cheap tests guarding the first thirty seconds of somebody's
experience with the package, which is the part no internal test naturally
covers because internal tests import from the module that defines a thing.
"""

import alphaengine


def test_version_is_exposed():
    assert isinstance(alphaengine.__version__, str)
    assert alphaengine.__version__


def test_documented_entry_point_is_callable():
    """`from alphaengine import sweep` must give the FUNCTION.

    Pinned because it silently was not. The subpackage `alphaengine.sweep`
    shadows the function unless __init__ re-exports it, so the exact line in
    our own README and module docstring raised "'module' object is not
    callable". It was caught by installing the built wheel and running the
    documented example, which is the only place a packaging-shaped bug like
    this shows up.
    """
    from alphaengine import sweep

    assert callable(sweep), "the documented top-level import must be the function"


def test_advertised_names_are_all_importable():
    """Everything in __all__ resolves. A promise in __all__ that raises on
    import is worse than not making the promise."""
    for name in alphaengine.__all__:
        assert hasattr(alphaengine, name), f"__all__ advertises {name!r}, which is not there"


def test_submodule_stays_reachable_explicitly():
    """Shadowing the subpackage name must not make the module unreachable."""
    from alphaengine.sweep import SweepResult
    from alphaengine.sweep.runner import sweep as runner_sweep

    assert SweepResult is alphaengine.SweepResult
    assert runner_sweep is alphaengine.sweep


def test_import_makes_no_network_call():
    """Importing must never reach the network.

    Asserted as a test rather than trusted as a convention: the offline
    guarantee is the reason a researcher can install this without asking
    anyone, and it would be easy to break accidentally by adding a client at
    module scope.
    """
    import sys

    assert "alphaengine" in sys.modules
    # No HTTP client should have been pulled in as a side effect of importing.
    # A transitive dependency importing one of these is the realistic way the
    # offline claim gets quietly broken.
    for mod in ("requests", "httpx", "urllib3", "aiohttp"):
        assert mod not in sys.modules, f"importing alphaengine dragged in {mod}"
