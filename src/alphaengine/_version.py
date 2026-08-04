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

# 0.3.0 IS THE RULE ABOVE BEING OBEYED, not a marketing number. The sweep
# addressed its result matrix by TRIAL index while failed trials contribute no
# column, so any grid with a failure read the wrong survivor's column (or
# raised). Fixing it changes the figure a 0.2 install would have produced from
# the same inputs — which is exactly what the MINOR position costs here.
__version__ = "0.3.0"
