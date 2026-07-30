# Security policy

## Reporting a vulnerability

Report privately through [GitHub Security Advisories](https://github.com/quantos/alphaengine/security/advisories/new),
or email **security@quantos.dev**. Please don't open a public issue for a
suspected vulnerability.

We'll acknowledge within **3 working days** and give an assessment within **10**.
If we accept the report we'll agree a disclosure timeline with you and credit
you in the advisory unless you'd rather we didn't.

## What's in scope

This repository: the library, its build, and its release pipeline. The hosted
platform is a separate system — report issues there to the same address and say
so.

## How this package is built and shipped

Worth stating plainly, because a library that computes numbers you defend to
investors deserves a supply chain you can inspect:

- **No API tokens exist.** Releases publish to PyPI through OIDC trusted
  publishing from GitHub Actions, so there is no long-lived credential in this
  repository or its secrets to steal. Exfiltrated publish tokens are the usual
  way a package gets hijacked.
- **Two runtime dependencies**, numpy and scipy, and that number is defended.
  Most supply-chain risk arrives through dependency count.
- **Releases are built once and published from that exact artifact**, so what CI
  tested and what PyPI serves are the same object.
- **Tagged releases only.** Publishing runs on a version tag, never on a branch.

## Reproducibility is a security property here

A changed computed value is treated as a breaking change and requires a major
version bump. CI fails if a golden value moves. This matters beyond correctness:
if a result silently changed between versions, a firm's saved study would stop
reproducing and nobody would know why.
