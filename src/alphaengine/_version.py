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
#
# 0.4.0 IS THE SAME RULE AGAIN, and it is the harder case to spot because it
# reads as "we added some charts". The wire's figure cap went 64 -> 512 and the
# per-period cap in `core.signals` went with it, so the SAME INPUTS NOW PRODUCE
# DIFFERENT OUTPUT: `ic_by_period` returns up to 512 readings where it returned
# 64, `best_curve` and `drawdown_curve` carry a curve rather than a sketch of
# one, a grid of up to 512 configurations records its full surface instead of
# reporting `trials_recorded: false`, and `compute.overlap` emits a rolling
# correlation and a scatter it did not emit before.
#
# Not one of those is a bug fix, and every one changes what a saved study
# reproduces. That costs the MINOR position while the leading digit is 0.
#
# 0.5.0 IS THE RULE READ THE OTHER WAY: no computed value moved, and the SURFACE
# broke. `--sleeve` and `--thesis` are removed along with `Session.sleeves()`,
# `.theses()` and `.propose_sleeve()`. Their three routes were part of the thesis
# object model, which came out of the platform with the agent desk it was built
# around, so the flags could only have failed at the END of a run — after the
# work, naming a server error rather than a missing feature.
#
# A removed flag is a breaking change even though every figure is byte-identical,
# and while the leading digit is 0 the MINOR position carries that too.
#
# Added in the same release, and the reason the removal is not a loss: `gaps` and
# `tonight`. The record already knew which workflow closes each open gap and
# nothing traversed those edges; these two read that derivation, so the terminal
# can now answer "what have I NOT tried" and "what will run while I sleep".
__version__ = "0.5.0"
