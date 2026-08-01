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

## What it does

**Runs your parameter grid.** `sweep()` calls your backtest function once per
combination. It does not backtest anything itself, so the engine you already
trust stays the engine you trust.

**Counts the trials for you.** The statistics that correct a Sharpe ratio for
multiple testing need to know how many variants were tested. That number is
almost never recorded, because nobody counts what they discarded. Running the
grid makes it `len(grid)`, so it never has to be asked for or asserted.

**Shows you the neighbourhood.** The output is whether your result sits on a
broad plateau or a knife edge, and where the robust region is centred. A single
spike surrounded by failures is a result fitted to its own parameters.

**Produces a portable study.** A JSON artifact holding what was tried, what came
back, and a content hash of the data it ran on. Readable in a text editor,
diffable, and versioned so it still parses in two years.

## What is in it

| Module | Contents |
|---|---|
| `alphaengine.core` | deflated Sharpe, PSR, PBO via CSCV, CPCV, minimum track record length, performance and risk statistics |
| `alphaengine.sweep` | the grid runner and the sensitivity surface |
| `alphaengine.study` | the study artifact and its schema |

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
a breaking change requiring a major version bump even when the signature is
unchanged. CI fails if a pinned value moves.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
