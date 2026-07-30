# AlphaEngine

**Is this an edge, or did you try enough things that something was bound to look good?**

```python
from alphaengine import sweep

r = sweep(backtest_fn, grid, data=prices)   # runs every combination
r.surface()                                 # a stable plateau, or a knife edge
r.verdict()                                 # deflated for the trials you actually ran
r.save("study.json")                        # the artifact, on your disk
```

```
pip install alphaengine
```

---

## The problem

A strategy has knobs. Nobody knows the right values, so you test combinations —
four lookbacks by three thresholds by three holding periods is thirty-six
variants — and one comes back with a Sharpe of 2.1.

**If you try thirty-six things, the best one looks good by luck alone.** Try five
hundred and a beautiful backtest is guaranteed. This is most of the reason
strategies work on paper and die live.

The statistics that correct for it are published and well understood. They all
need one input: **how many things did you try?** That number is almost never
recorded, because nobody counts what they threw away.

## What this does about it

It runs the grid, so the count is `len(grid)` and nobody has to be asked.

The deflation is a by-product. The thing you get back is the **neighbourhood** —
whether your result sits on a broad plateau or a knife edge, and where the
plateau's centre is. That's the output worth having: you end the session with a
better strategy, not just a worse number.

## What it is not

**Not a backtester.** `sweep()` takes *your* function. We orchestrate and
measure; you simulate. The engine you already trust stays the engine you trust,
and we never become responsible for its correctness.

**Not a service.** `import alphaengine` makes no network call and needs no
account. numpy and scipy, nothing else — both already in your environment.
Everything above runs with the wifi off, and your data never leaves the machine.

## What's inside

| | |
|---|---|
| `alphaengine.core` | deflated Sharpe, PBO/CSCV, CPCV, minimum track record length |
| `alphaengine.sweep` | the grid runner and the sensitivity surface |
| `alphaengine.study` | the portable artifact and its schema |

## The numbers are a contract

Once published, a study written today has to reproduce in two years. So a
**changed computed value is a breaking change** and requires a major version
bump, even when nothing about the signature moved. CI fails on a moved golden.

The methods are from the published literature — Bailey and López de Prado on
deflated Sharpe and backtest overfitting, Harvey and Liu on multiple testing.
Nothing here is a proprietary formula, and it shouldn't be: a referee whose
reasoning you can't inspect isn't a referee.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

Built by [QuantOS](https://quantos.dev). The library is free and always will be.
The hosted platform — where studies persist, workflows run, and a record
accumulates across a firm — is the commercial product.
