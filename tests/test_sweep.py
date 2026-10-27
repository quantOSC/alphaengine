"""The sweep runner and the study artifact."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alphaengine.study import SCHEMA_VERSION, Study, load, save
from alphaengine.sweep import sweep


def prices(seed: int = 42, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))


def ma_cross(data=None, fast: int = 10, slow: int = 50):
    """A plain moving-average rule, standing in for a user's own backtest."""
    p = np.asarray(data, dtype=float)
    r = np.diff(p, prepend=p[0]) / p
    f = np.convolve(p, np.ones(fast) / fast, mode="same")
    s = np.convolve(p, np.ones(slow) / slow, mode="same")
    pos = (f > s).astype(float)
    return (np.r_[0.0, pos[:-1]] * r).tolist()


# ── the trial count ────────────────────────────────────────────────────────
def test_n_trials_is_derived_from_the_grid():
    """The reason this module exists. The count is the grid, not an argument."""
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    assert r.n_trials == 6
    assert r.verdict()["n_trials"] == 6
    assert r.verdict()["n_trials_source"] == "derived_from_grid"


def test_there_is_no_way_to_assert_a_trial_count():
    """`sweep()` must not accept an n_trials override.

    Pinned as a test because it is a design guarantee rather than an
    implementation detail: the moment a caller can supply this number, the
    deflation becomes self-reported and the library stops being worth using.
    """
    import inspect

    params = inspect.signature(sweep).parameters
    assert "n_trials" not in params
    assert not any("trial" in p.lower() for p in params)


def test_larger_grid_deflates_harder():
    p = prices()
    small = sweep(ma_cross, {"fast": [10, 15], "slow": [50, 200]}, data=p)
    large = sweep(ma_cross, {"fast": [5, 8, 10, 12, 15, 20], "slow": [40, 50, 80, 100, 150, 200]}, data=p)
    assert large.n_trials > small.n_trials
    # Same underlying rule family, more search, so the honest verdict cannot be
    # more flattering. This is the property a user is relying on.
    assert large.verdict()["deflated_sharpe"] <= small.verdict()["deflated_sharpe"] + 1e-9


# ── the surface ────────────────────────────────────────────────────────────
def test_surface_classifies_the_neighbourhood():
    r = sweep(ma_cross, {"fast": [5, 10, 15, 20], "slow": [50, 100, 200]}, data=prices())
    s = r.surface()
    assert s["shape"] in {"plateau", "ridge", "knife_edge"}
    assert s["n_ok"] + s["n_failed"] == r.n_trials
    assert 0.0 <= s["share_within_20pct_of_best"] <= 1.0
    assert s["reading"]


def test_matrix_is_the_full_trial_set():
    """PBO needs every column. A single-point run structurally cannot produce it."""
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    assert r.matrix.shape[1] == 6
    assert r.verdict()["selection"]["n_configs"] == 6


def test_single_configuration_reports_no_selection_reading():
    """One config means the choice among configs was not a choice."""
    r = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices())
    assert "selection" not in r.verdict()


# ── parameters are treated as IP ───────────────────────────────────────────
def test_params_are_not_stored_by_default():
    """The grid is frequently bigger IP than the returns."""
    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50, 100]}, data=prices())
    v = r.verdict()
    assert "best_params" not in v
    assert v["best_params_hash"]  # comparable without being disclosed

    r2 = sweep(ma_cross, {"fast": [5, 10], "slow": [50, 100]}, data=prices(), store_params=True)
    assert r2.verdict()["best_params"]


def test_data_identity_is_a_hash_not_a_label():
    a = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=1))
    b = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=1))
    c = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=2))
    assert a.data_hash == b.data_hash, "same data must be the same segment"
    assert a.data_hash != c.data_hash, "different data must not collide"


# ── failure handling ───────────────────────────────────────────────────────
def test_one_bad_combination_does_not_lose_the_rest():
    def flaky(data=None, fast: int = 10, slow: int = 50):
        if fast == 15:
            raise ValueError("deliberate failure")
        return ma_cross(data=data, fast=fast, slow=slow)

    r = sweep(flaky, {"fast": [5, 10, 15], "slow": [50]}, data=prices())
    assert r.n_trials == 3
    assert r.surface()["n_failed"] == 1
    assert r.surface()["n_ok"] == 2


def test_all_failing_raises_rather_than_returning_a_verdict():
    def broken(data=None, **_):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError, match="every one of"):
        sweep(broken, {"fast": [5, 10]}, data=prices())


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="grid is empty"):
        sweep(ma_cross, {}, data=prices())


# ── the study artifact ─────────────────────────────────────────────────────
def test_study_roundtrip(tmp_path):
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    p = r.save(tmp_path / "study.json", label="ma cross", data_description="synthetic")

    back = load(p)
    assert back.label == "ma cross"
    assert back.n_trials == 6
    assert back.n_trials_source == "derived_from_grid"
    assert back.data_hash == r.data_hash
    assert back.schema_version == SCHEMA_VERSION
    assert back.surface["shape"] in {"plateau", "ridge", "knife_edge"}


def test_study_holds_no_series(tmp_path):
    """A study is derived facts and references. The inputs stay with their owner."""
    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50]}, data=prices())
    raw = json.loads(r.save(tmp_path / "s.json").read_text(encoding="utf8"))

    blob = json.dumps(raw)
    assert "matrix" not in raw
    assert "returns" not in raw
    assert "prices" not in raw
    # A price series would be a long run of floats; nothing here should be one.
    for value in raw.values():
        if isinstance(value, list):
            assert len(value) < 50, "a study should not carry a series"
    assert len(blob) < 20_000


def test_study_refuses_an_unknown_major_schema(tmp_path):
    """Refuse rather than half-parse. These figures get quoted to investors."""
    p = tmp_path / "future.json"
    save(Study(label="from the future"), p)
    raw = json.loads(p.read_text(encoding="utf8"))
    raw["schema_version"] = "99.0"
    p.write_text(json.dumps(raw), encoding="utf8")

    with pytest.raises(ValueError, match="cannot be read"):
        load(p)


def test_study_tolerates_unknown_minor_fields(tmp_path):
    """A newer minor version must stay readable by an older build."""
    p = tmp_path / "newer.json"
    save(Study(label="x", n_trials=3), p)
    raw = json.loads(p.read_text(encoding="utf8"))
    raw["schema_version"] = f"{SCHEMA_VERSION.split('.')[0]}.99"
    raw["a_field_from_the_future"] = {"anything": True}
    p.write_text(json.dumps(raw), encoding="utf8")

    back = load(p)
    assert back.label == "x"
    assert back.n_trials == 3


def test_study_schema_1_0_still_loads(tmp_path):
    """1.1 is a minor bump: optional process/charts. A 1.0 file must still parse."""
    payload = Study(label="old", n_trials=2).to_dict()
    payload["schema_version"] = "1.0"
    payload.pop("process", None)
    payload.pop("charts", None)
    p = tmp_path / "v10.json"
    p.write_text(json.dumps(payload), encoding="utf8")
    back = load(p)
    assert back.process is None
    assert back.charts is None
    assert back.n_trials == 2


def test_study_1_1_keeps_optional_process_and_charts(tmp_path):
    s = Study(
        label="p",
        n_trials=4,
        process={"dgp": "ou"},
        charts=[{"kind": "hist", "key": "sharpe_hist", "title": "Sharpe under the DGP"}],
    )
    back = load(save(s, tmp_path / "p.json"))
    assert back.process == {"dgp": "ou"}
    assert back.charts[0]["kind"] == "hist"
    assert back.schema_version == SCHEMA_VERSION


def test_study_report_payload_includes_all_ingest_fields_and_guards_length(monkeypatch):
    import io
    import urllib.request

    from alphaengine.study.report import MAX_FIGURE_LIST, _guard, _payload

    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50, 100]}, data=prices(), store_params=True)
    study = Study.from_sweep(
        r,
        label="full ingest payload",
        data_description="synthetic",
        notes="checked",
        assignment_id="asgn_123",
        thesis_id="th_456",
    )
    payload = _payload(study)

    # Every field the quantOS ingest model accepts:
    for expected in (
        "label",
        "schema_version",
        "engine_version",
        "n_trials",
        "n_trials_source",
        "grid_keys",
        "n_failed_trials",
        "data_hash",
        "data_description",
        "verdict",
        "deflated_sharpe",
        "psr_vs_zero",
        "sr0_expected_max",
        "min_track_record_length",
        "selection",
        "surface",
        "performance",
        "notes",
        "assignment_id",
        "thesis_id",
    ):
        assert expected in payload, f"missing {expected} from _payload"

    assert payload["grid_keys"] == ["fast", "slow"]
    assert payload["n_failed_trials"] == 0
    assert payload["assignment_id"] == "asgn_123"
    assert payload["thesis_id"] == "th_456"
    assert payload["selection"]["n_configs"] == 4
    # Allowlist still excludes best_params
    assert "best_params" not in payload
    assert _guard(payload) is payload

    # Length guard still rejects any > 512 list inside any ingest field
    bad = dict(payload)
    bad["selection"] = {"logits": list(range(MAX_FIGURE_LIST + 1))}
    with pytest.raises(ValueError, match="is a series"):
        _guard(bad)

    # End-to-end report() sends the full payload and supports overriding assignment_id/thesis_id
    sent: list[dict] = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data.decode("utf-8")))

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(b'{"id": "run_1", "study_id": "st_1"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    res = study.report(
        api_key="ae_live_test",
        base_url="https://api.example.invalid",
        assignment_id="asgn_override",
        thesis_id="th_override",
    )
    assert res["id"] == "run_1"
    assert len(sent) == 1
    assert sent[0]["assignment_id"] == "asgn_override"
    assert sent[0]["thesis_id"] == "th_override"
    assert sent[0]["psr_vs_zero"] == study.psr_vs_zero
    assert sent[0]["sr0_expected_max"] == study.sr0_expected_max
    assert sent[0]["min_track_record_length"] == study.min_track_record_length
    assert sent[0]["selection"] == study.selection


def test_study_seal_full_data_hash_trial_ledger_and_1_1_backward_compat(tmp_path):
    def flaky(data=None, fast: int = 10, slow: int = 50):
        if fast == 15:
            raise ValueError("bad window")
        return ma_cross(data=data, fast=fast, slow=slow)

    r = sweep(flaky, {"fast": [5, 10, 15], "slow": [50]}, data=prices())
    # Full 64-hex data hash
    assert len(r.data_hash) == 64
    int(r.data_hash, 16)

    # Optional per-trial ledger export
    ledger = r.ledger()
    assert len(ledger) == 3
    assert ledger[0]["params_hash"] and isinstance(ledger[0]["sharpe"], float)
    assert ledger[2]["failed"] is not None and ledger[2]["sharpe"] is None

    study = Study.from_sweep(r, label="sealed study", include_ledger=True)
    assert study.schema_version == "1.2"
    assert study.trial_ledger is not None and len(study.trial_ledger) == 3
    assert study.content_sha256 is not None and len(study.content_sha256) == 64
    assert study.verify() is True

    # Backward compatibility with reading 1.1
    v11 = study.to_dict()
    v11["schema_version"] = "1.1"
    v11.pop("content_sha256", None)
    v11.pop("trial_ledger", None)
    v11_path = tmp_path / "study_v11.json"
    v11_path.write_text(json.dumps(v11), encoding="utf-8")
    loaded_v11 = load(v11_path)
    assert loaded_v11.schema_version == "1.1"
    assert loaded_v11.content_sha256 is None
    assert loaded_v11.verify() is False


def test_cli_seal_and_report_default(tmp_path, monkeypatch, capsys):
    import io
    import urllib.request

    from alphaengine import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    monkeypatch.setenv("QUANTOS_API_KEY", "ae_live_test")

    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50]}, data=prices())
    study = Study.from_sweep(r, label="cli study", notes="ready")
    # Write unsealed study first so `alphaengine seal` writes the hash, then re-run to verify
    raw = study.to_dict()
    raw["content_sha256"] = None
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    rc1 = cli.main(["seal", str(study_path)])
    assert rc1 == 0
    out1 = capsys.readouterr().out
    assert "Sealed." in out1

    rc2 = cli.main(["seal", str(study_path)])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "Verified." in out2

    # Tamper with the sealed file and verify `seal` fails
    tampered = json.loads(study_path.read_text(encoding="utf-8"))
    tampered["n_trials"] = 999
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert cli.main(["seal", str(tampered_path)]) == 1
    capsys.readouterr()

    # CLI report -> /api/me/runs/ingest
    requests_seen: list[tuple[str, dict]] = []

    def fake_urlopen(req, timeout=None):
        requests_seen.append((req.full_url, json.loads(req.data.decode("utf-8"))))

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(b'{"id": "rec_1"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert (
        cli.main(
            [
                "report",
                str(study_path),
                "--key",
                "ae_live_test",
                "--url",
                "https://api.example.invalid",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert len(requests_seen) == 1
    assert requests_seen[0][0] == "https://api.example.invalid/api/me/runs/ingest"
    assert requests_seen[0][1]["content_sha256"] == study.content_sha256

    # Verify `--retail` flag is rejected by parser
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["report", str(study_path), "--retail"])
