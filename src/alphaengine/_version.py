"""Single source of the version.

Read by hatchling at build time (see [tool.hatch.version] in pyproject.toml), so
this file and the published artifact can never disagree.

SEMVER, WITH ONE LOCAL RULE: a MAJOR bump is required whenever a computed value
changes. Once this is on PyPI the numbers are a public contract, somebody's
saved study has to reproduce in two years, so a different result from the same
inputs is a breaking change even when the signature is untouched.

While the leading digit is 0, the MINOR position carries that rule: 0.1 -> 0.2
is what a changed figure costs. The API may still move underneath it.
"""

__version__ = "0.2.1"
