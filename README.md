# AlphaEngine

Validated research tooling for investment strategies. Run a parameter search,
get back the shape of the result and an honest read on whether it survives the
number of things you tried.

```bash
pip install alphaengine
```

```python
from alphaengine import sweep

r = sweep(backtest_fn, {"fast": [5, 10, 20], "slow": [50, 100, 200]}, data=prices)

r.surface()    # is the result a broad plateau or a single lucky configuration?
r.verdict()    # deflated for the 9 trials that were actually run
r.save()       # study.json, on your disk
```

Everything above runs offline, with no account and no key.

## Try it without writing anything

The repo ships a runnable project module, so you can see the whole offline half
work before deciding whether any of this is for you:

```bash
git clone https://github.com/quantOSC/alphaengine && cd alphaengine
pip install -e .
python -m examples.momentum
```

```
trials     9  (derived_from_grid)
verdict    marginal
surface    ridge
dsr        0.6352
```

That is a moving-average crossover on **synthetic prices from a fixed seed** —
no download, no data licence, same numbers on every machine. It is a
demonstration of the wiring, not a strategy. A crossover on a random walk has no
edge, and the verdict says so rather than flattering it. **That is the example
working, not failing.**

## Writing your own `backtest_fn`

Two rules, both easy to get wrong the first time, and the reason the example
above exists to copy:

**Return a bare 1-D return series.** Not a dict, not a stats object — the
per-period returns themselves. `sweep` does `np.asarray(list(raw))`, so a dict
of results iterates its *keys* and fails on the first string.

**Return the same length for every combination.** PBO splits the trial matrix
into time blocks and compares configurations within each block, which only means
anything if they line up in time. Ragged output is refused rather than truncated,
because silently trimming produces a confident number over series that do not
correspond. In practice: pick a warm-up long enough for the slowest window in
your grid and start every configuration there.

```python
WARMUP = 200   # covers the slowest `slow` in the grid

def backtest_fn(*, data, fast, slow):
    close = data["close"]
    return [
        (close[i + 1] - close[i]) / close[i] * (1 if sma(close, fast, i) > sma(close, slow, i) else 0)
        for i in range(WARMUP, len(close) - 1)
    ]
```

`data` is whatever you want it to be — a DataFrame, a dict of series, an array.
The package never inspects it and it never leaves your machine.

## What it does

**Runs your parameter grid.** `sweep()` calls your backtest function once per
combination. It does not backtest anything itself, so the engine you already
trust stays the engine you trust.

**Counts the trials for you.** The statistics that correct a Sharpe ratio for
multiple testing need to know how many variants were tested. That number is
almost never recorded, because nobody counts what they discarded. Running the
grid makes it `len(grid)`, so it never has to be asked for or asserted.

**Refuses to flatter an unrecorded count.** Since 0.2.0, omitting `n_trials`
means `not_recorded` — not `1`. The trial count comes back `null`,
`n_trials_source` travels beside it, and **a verdict of `edge` is unreachable
without a recorded denominator.** A deflated Sharpe is a ratio; deflating by a
denominator nobody wrote down does not produce a weaker claim, it produces a
claim about nothing.

**Shows you the neighbourhood.** The output is whether your result sits on a
broad plateau or a knife edge, and where the robust region is centred. A single
spike surrounded by failures is a result fitted to its own parameters.

**Produces a portable study.** A JSON artifact holding what was tried, what came
back, and a content hash of the data it ran on. Readable in a text editor,
diffable, and versioned so it still parses in two years.

## Running a workflow from the terminal

The offline half above is complete on its own. A *workflow* adds a sequence — 
what to run, in what order, and what stops the run — and that sequence lives on
a QuantOS workflow server, so this part needs an account.

```bash
alphaengine                                                # a session
alphaengine workflows                                      # what your workspace offers
alphaengine run validate_study --project examples.momentum
```

### "command not found"

The console script lives in your venv's `Scripts/` (Windows) or `bin/` (POSIX),
which joins your PATH only while that venv is **activated**. A fresh install
followed by `alphaengine` therefore says *command not found*, which reads as a
broken install and is not one.

```powershell
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate        # macOS / Linux
alphaengine
```

Or skip activation entirely — **`python -m alphaengine` always works**, because
the interpreter that can import the package can always run it:

```bash
.venv/Scripts/python.exe -m alphaengine      # Windows, unactivated
.venv/bin/python -m alphaengine              # POSIX, unactivated
```

This is the price of in-process data access, and it is the one cost we will not
engineer around: a launcher that worked from anywhere would have to run in its
own environment, which is exactly the isolation that makes `compute.*` unable to
see your DataFrames.

The run narrates itself, because a loop you cannot watch is a loop you cannot
trust. The server names an op, this machine executes it and hands back figures:

```
validate_study · <your workflow server>
  server →  compute.sweep
  local  ·  done
  server →  compute.deflated_sharpe
  local  ·  done
  server →  Stop: the surface is a ridge, not a plateau
```

`--project` names an ordinary module of yours exposing `data` and, if the
workflow sweeps, `backtest_fn`. Authenticate with a key from the portal:

```bash
export QUANTOS_API_KEY=ae_live_...     # QUANTOS_API_URL for self-hosted or VPC
```

**A stop exits 0.** "This did not clear the bar" is the system working, not a
broken build — a non-zero exit there would make every CI pipeline treat an
honest refusal as a failure, which is exactly the pressure that gets honesty
controls switched off. Only a run that could not execute a step exits non-zero.

> **Install this into the venv you do research in — not with `pipx` or
> `uv tool install`.** The compute steps execute in-process against your own
> DataFrames, so tool isolation, which is normally the right way to install a
> CLI, is the one thing that cannot work here. You cannot have isolation and
> in-process data access, and your data not moving is the point.

**No LLM dependency and no key field.** `AgentDriver` takes a callable, so you
bring your own model by passing a function. There is nowhere in this tool to put
a model key, ours or yours, and that is how "runs under your own account" is
satisfied structurally rather than promised.

## What is in it

| Module | Contents |
|---|---|
| `alphaengine.core` | deflated Sharpe, PSR, PBO via CSCV, CPCV, minimum track record length, performance and risk statistics |
| `alphaengine.sweep` | the grid runner and the sensitivity surface |
| `alphaengine.study` | the study artifact and its schema |
| `alphaengine.client` | the workflow client and the step executor |
| `alphaengine.cli` | the `alphaengine` terminal entry point |

Two runtime dependencies, numpy and scipy, both already present in a typical
research environment. `import alphaengine` makes no network call and needs no
account. Factor decomposition and cointegration testing need statsmodels and
are available as `pip install 'alphaengine[factors]'`.

## Getting a study to somebody else

`save()` writes to your disk and needs no account. When the work has to reach
the PM who will act on it, `report()` sends the study — and only the study.

```python
import os
from alphaengine import Study, sweep

os.environ["QUANTOS_API_KEY"] = "ae_live_..."   # created in the portal

r = sweep(backtest_fn, grid, data=prices)
r.save()                                        # yours, on your disk, always

Study.from_sweep(r, label="momentum, 9 configs").report()
```

What crosses is an explicit allowlist: the trial count and how it was obtained,
a content hash of the data, the verdict, the shape of the neighbourhood, the
performance figures. Your returns, your prices and your parameter grid stay on
the machine, and a guard keyed on length rather than field name refuses to send
anything series-shaped whatever it is called.

Reporting is the only part of this package that touches a network, so it is the
only part that is not imported until you call it. `import alphaengine` still
makes no network call.

## Where this sits in QuantOS

AlphaEngine is the open research layer of the [QuantOS](https://github.com/quantOSC)
platform. It is the piece that runs on your machine, against your data, and it
is complete on its own: everything above works offline and forever, at no cost.

The QuantOS platform builds on it. Studies produced here can be persisted to a
firm's record, referenced when an idea becomes a position, and assembled into
the reports that go to an investment committee or an allocator. The library
computes; the platform remembers and reports. The two halves are separated so
that the part touching your data has no reason to phone home.

## The methods

Everything in `core` comes from the published literature. Nothing here is a
proprietary formula, which is deliberate: a referee whose reasoning you cannot
inspect is not a referee.

**Deflated Sharpe Ratio, Probabilistic Sharpe Ratio, minimum track record length**
Bailey, D. H., and López de Prado, M. (2012). "The Sharpe Ratio Efficient
Frontier." *Journal of Risk* 15(2), 3 to 44.
Bailey, D. H., and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
*Journal of Portfolio Management* 40(5), 94 to 107.

**Probability of Backtest Overfitting via CSCV**
Bailey, D. H., Borwein, J., López de Prado, M., and Zhu, Q. J. (2017). "The
Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4),
39 to 69.

**Combinatorial purged cross-validation**
López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley,
chapters 7 and 12.

**Multiple testing in asset pricing**
Harvey, C. R., Liu, Y., and Zhu, H. (2016). "... and the Cross-Section of
Expected Returns." *Review of Financial Studies* 29(1), 5 to 68.
Harvey, C. R., and Liu, Y. (2015). "Backtesting." *Journal of Portfolio
Management* 42(1), 13 to 28.

**Downside deviation**
Sortino, F. A., and Price, L. N. (1994). "Performance Measurement in a Downside
Risk Framework." *Journal of Investing* 3(3), 59 to 64.

**Factor regression standard errors** (in the `factors` extra)
Newey, W. K., and West, K. D. (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3), 703 to 708.

**Unit root testing for cointegration** (in the `factors` extra)
Dickey, D. A., and Fuller, W. A. (1979). "Distribution of the Estimators for
Autoregressive Time Series with a Unit Root." *Journal of the American
Statistical Association* 74(366), 427 to 431.

## Reproducibility

The values these functions return are treated as a public contract. A study
written today has to reproduce in two years, so a change to a computed value is
a breaking change requiring a version bump even when the signature is unchanged.
While the leading digit is 0 the minor position carries that rule — 0.1 → 0.2 is
what a changed figure costs — so every 0.2.x release produces identical numbers.
CI fails if a pinned value moves.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,factors]'
```

CI runs five gates. Run all of them before pushing — `ruff check` and
`ruff format` are different tools with different opinions, and passing one
says nothing about the other:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pytest
python -m pytest tests -m golden -q      # the frozen figures
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
