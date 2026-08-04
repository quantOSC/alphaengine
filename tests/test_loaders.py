"""Getting data into a run from a file instead of a Python module.

THE GAP THIS CLOSES. Data reached a workflow exactly one way: `--project
<module>`, importing something that exposes `data` and `backtest_fn`. That is the
right door for a quant with a research package and the wrong one for the same
quant on the day somebody emails them a CSV. Three of the four workflows need
numbers, not code, and "write a Python module first" was a real barrier in front
of them.

THE TESTS THAT MATTER ARE THE SHAPE ONES. A loader that guesses wrong does not
raise -- it returns a plausible structure, and a screen that ranked returns as
prices reports a shortlist nobody can tell is wrong. So detection is by HEADER,
and anything undecidable raises naming the three shapes.
"""

from __future__ import annotations

import pytest

from alphaengine.loaders import DataShapeError, coverage, load_csv, load_delimited

WIDE = """date,AAPL,MSFT,NVDA
2026-01-01,100.0,200.0,300.0
2026-01-02,101.0,201.5,305.0
2026-01-03,102.5,199.0,310.0
"""

LONG = """date,symbol,close
2026-01-02,AAPL,101.0
2026-01-01,AAPL,100.0
2026-01-01,MSFT,200.0
2026-01-02,MSFT,201.5
"""

SERIES = """date,close
2026-01-01,0.004
2026-01-02,-0.002
2026-01-03,0.011
"""


# ── the three shapes ───────────────────────────────────────────────────────


def test_a_wide_file_is_a_universe_and_keeps_its_dates():
    """`{symbol: {date: close}}`, not `{symbol: [closes]}`. Dropping the dates
    here made a dated file profile as undated two steps later — a capability
    the caller paid for, thrown away in transit."""
    got = load_delimited(WIDE)
    assert set(got) == {"AAPL", "MSFT", "NVDA"}
    assert got["AAPL"]["2026-01-01"] == 100.0
    assert len(got["NVDA"]) == 3


def test_a_long_file_is_a_universe_and_is_sorted_by_date():
    """The file's row order is the file's business. Every metric downstream
    reads the mapping sorted by its date key, latest last."""
    got = load_delimited(LONG)
    assert set(got) == {"AAPL", "MSFT"}
    assert list(got["AAPL"]) == ["2026-01-01", "2026-01-02"]
    assert got["AAPL"]["2026-01-02"] == 101.0


def test_a_two_column_file_is_a_single_series():
    assert load_delimited(SERIES) == [0.004, -0.002, 0.011]


def test_one_bare_column_of_numbers_is_a_series():
    """No header, because a returns file exported from a notebook often has
    none. The first row being a number is what says so."""
    assert load_delimited("0.01\n-0.02\n0.03\n") == [0.01, -0.02, 0.03]


# ── the dates survive the load ─────────────────────────────────────────────


def test_a_dated_universe_profiles_with_its_calendar_intact():
    """End to end at the unit level: the loader keeps the dates, and
    `profile_data` can therefore run the checks that need a calendar —
    `dates_supplied` is EARNED, not defaulted."""
    from alphaengine.core import profile_data

    out = profile_data(load_delimited(WIDE))
    assert out["dates_supplied"] is True
    assert out["usable"] == 3
    assert out["panel_first"] == "2026-01-01"
    assert out["panel_last"] == "2026-01-03"


def test_a_dateless_file_keeps_the_old_list_shape_byte_for_byte():
    """A bare series has no dates to keep, and inventing a calendar for it
    would be the loader guessing. The list shape is the contract."""
    assert load_delimited("0.01\n-0.02\n0.03\n") == [0.01, -0.02, 0.03]
    assert load_delimited("returns\n0.01\n-0.02\n") == [0.01, -0.02]
    # A dated SINGLE series also stays a list: one series feeds the
    # return-measuring workflows, which read values, not calendars.
    assert load_delimited(SERIES) == [0.004, -0.002, 0.011]


def test_a_named_single_column_drops_its_header():
    assert load_delimited("returns\n0.01\n-0.02\n") == [0.01, -0.02]


# ── detection is by header, and refuses rather than guesses ────────────────


def test_a_long_file_is_never_mistaken_for_a_two_name_wide_file():
    """Checked first for exactly this reason: date+symbol+close has three
    columns, and so does a wide file with two names."""
    got = load_delimited(LONG)
    assert "SYMBOL" not in got and "CLOSE" not in got


def test_an_undecidable_file_names_the_three_shapes():
    """A loader that guesses eventually guesses wrong on somebody's book, and
    the failure is a plausible number rather than an error."""
    with pytest.raises(DataShapeError) as e:
        load_delimited("alpha,beta,gamma\n1,2,3\n4,5,6\n")
    for shape in ("wide", "long", "series"):
        assert shape in str(e.value)


def test_an_empty_file_says_so():
    with pytest.raises(DataShapeError):
        load_delimited("")
    with pytest.raises(DataShapeError):
        load_delimited("date,close\n")


# ── the awkward real-world bits ────────────────────────────────────────────


def test_semicolons_and_tabs_work_without_being_asked_about():
    assert set(load_delimited(WIDE.replace(",", ";"))) == {"AAPL", "MSFT", "NVDA"}
    assert set(load_delimited(WIDE.replace(",", "\t"))) == {"AAPL", "MSFT", "NVDA"}


def test_a_gap_is_a_gap_and_is_not_filled():
    """Filling it would invent observations the file does not contain, and every
    coverage figure downstream would be a lie about data nobody supplied."""
    got = load_delimited("date,AAPL,MSFT\n2026-01-01,100,\n2026-01-02,101,200\n")
    assert len(got["AAPL"]) == 2
    assert len(got["MSFT"]) == 1


def test_the_usual_spellings_of_missing_are_all_missing():
    got = load_delimited("date,AAPL\n2026-01-01,NA\n2026-01-02,n/a\n2026-01-03,100\n")
    assert len(got["AAPL"]) == 1


def test_thousands_separators_and_whitespace_survive():
    """Quoted, because an unquoted `1,200.50` in a comma-delimited file is three
    fields and no parser can rescue it. Excel quotes them; this handles that."""
    got = load_delimited('date,AAPL\n2026-01-01," 1,200.50 "\n')
    assert got["AAPL"]["2026-01-01"] == 1200.50


def test_a_single_name_file_is_a_universe_of_one_not_a_series():
    """`date,AAPL` is ambiguous until you look at the second header. A series
    file names that column `close` or `returns`; anything else is a ticker."""
    assert load_delimited("date,AAPL\n2026-01-01,100\n2026-01-02,101\n") == {
        "AAPL": {"2026-01-01": 100.0, "2026-01-02": 101.0}
    }
    assert load_delimited("date,close\n2026-01-01,100\n") == [100.0]


def test_a_symbol_is_upper_cased_so_a_universe_can_match_it():
    assert "AAPL" in load_delimited("date,symbol,close\n2026-01-01,aapl,100\n")
    assert "AAPL" in load_delimited("date,aapl\n2026-01-01,100\n")


def test_alternative_header_names_are_understood():
    for date in ("date", "timestamp", "dt", "as_of"):
        got = load_delimited(f"{date},symbol,price\n2026-01-01,AAPL,100\n")
        assert "AAPL" in got, date


# ── the file wrapper ───────────────────────────────────────────────────────


def test_load_csv_reads_a_real_file(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text(WIDE, encoding="utf-8")
    assert set(load_csv(p)) == {"AAPL", "MSFT", "NVDA"}


def test_a_byte_order_mark_does_not_hide_the_date_column(tmp_path):
    """Excel writes one. Without this the header reads as `\\ufeffdate` and the
    whole file becomes undecidable."""
    p = tmp_path / "excel.csv"
    p.write_bytes(b"\xef\xbb\xbf" + WIDE.encode())
    assert set(load_csv(p)) == {"AAPL", "MSFT", "NVDA"}


def test_a_missing_file_says_which_one(tmp_path):
    with pytest.raises(DataShapeError) as e:
        load_csv(tmp_path / "nope.csv")
    assert "nope.csv" in str(e.value)


def test_a_bad_file_names_itself_in_the_error(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("alpha,beta,gamma\n1,2,3\n", encoding="utf-8")
    with pytest.raises(DataShapeError) as e:
        load_csv(p)
    assert "wrong.csv" in str(e.value)


# ── narrowing to a registered universe ─────────────────────────────────────


def test_coverage_keeps_the_named_symbols_and_reports_the_rest():
    """Missing names are RETURNED, not logged and dropped. A universe of 500
    screened against a file holding 40 is a different result from a universe of
    40, and this is the last place that difference is visible."""
    data = load_delimited(WIDE)
    kept, missing = coverage(data, ["AAPL", "NVDA", "TSLA", "AMZN"])
    assert set(kept) == {"AAPL", "NVDA"}
    assert missing == ["TSLA", "AMZN"]


def test_coverage_matches_case_insensitively():
    kept, missing = coverage(load_delimited(WIDE), ["aapl", " msft "])
    assert set(kept) == {"AAPL", "MSFT"}
    assert missing == []


def test_a_universe_definition_against_a_single_series_is_refused():
    with pytest.raises(DataShapeError) as e:
        coverage([0.01, 0.02], ["AAPL"])
    assert "single series" in str(e.value)
