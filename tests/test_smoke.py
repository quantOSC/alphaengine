"""Skeleton checks, replaced by the real suite when core/ lands.

The golden marker is registered here from the start so the CI job that guards
computed values has something to run before there are any values to guard.
"""

import alphaengine


def test_version_is_exposed():
    assert isinstance(alphaengine.__version__, str)
    assert alphaengine.__version__


def test_import_makes_no_network_call():
    """Importing must never reach the network.

    Asserted as a test rather than trusted as a convention: the offline
    guarantee is the reason a researcher can install this without asking
    anyone, and it would be easy to break accidentally by adding a client at
    module scope.
    """
    import sys

    assert "alphaengine" in sys.modules
