"""The checkpoint override must be opt-in and confined to offline evaluation.

get_nowcast_service() honours STORMSENSE_CKPT / STORMSENSE_CONFIG so an offline
backtest can drive a CANDIDATE model through the real production inference path
instead of a reimplementation. That is only safe if the variables are absent in
normal operation, so these tests pin the default.
"""
import os

import src.inference.nowcast_service as ns


def test_override_env_vars_are_not_set_by_default():
    """Nothing in the repo may set these; they are for an explicit operator."""
    assert not os.environ.get("STORMSENSE_CKPT"), (
        "STORMSENSE_CKPT leaked into the environment -- the served model would "
        "silently change"
    )
    assert not os.environ.get("STORMSENSE_CONFIG")


def test_service_reads_override_only_from_environment(monkeypatch):
    """With the vars unset the service must request the default checkpoint."""
    captured = {}

    class _Stub:
        def __init__(self, checkpoint_path=None, config_path=None):
            captured["ckpt"] = checkpoint_path
            captured["cfg"] = config_path

    monkeypatch.setattr(ns, "NowcastService", _Stub)
    monkeypatch.setattr(ns, "_SERVICE_INSTANCE", None)
    monkeypatch.delenv("STORMSENSE_CKPT", raising=False)
    monkeypatch.delenv("STORMSENSE_CONFIG", raising=False)

    ns.get_nowcast_service()
    # With no override the service resolves the ACTIVE model explicitly.
    #
    # This previously asserted {"ckpt": None, "cfg": None}, i.e. that the
    # service deferred to NowcastService.__init__'s own search. The application
    # names the active model in one place -- ACTIVE_MODEL_CHECKPOINT /
    # ACTIVE_MODEL_CONFIG -- so the default arrives here as an explicit path
    # instead of None. The CONTRACT this test exists to pin is unchanged and is
    # asserted below: the environment variables must not be what selects the
    # model in normal operation, whichever checkpoint is currently active.
    assert captured["ckpt"] == ns.ACTIVE_MODEL_CHECKPOINT
    assert captured["cfg"] == ns.ACTIVE_MODEL_CONFIG


def test_service_honours_override_when_set(monkeypatch):
    captured = {}

    class _Stub:
        def __init__(self, checkpoint_path=None, config_path=None):
            captured["ckpt"] = checkpoint_path
            captured["cfg"] = config_path

    monkeypatch.setattr(ns, "NowcastService", _Stub)
    monkeypatch.setattr(ns, "_SERVICE_INSTANCE", None)
    monkeypatch.setenv("STORMSENSE_CKPT", "some/candidate.pt")
    monkeypatch.setenv("STORMSENSE_CONFIG", "configs/candidate.yaml")

    ns.get_nowcast_service()
    assert captured["ckpt"] == "some/candidate.pt"
    assert captured["cfg"] == "configs/candidate.yaml"


def test_production_checkpoint_is_the_default_choice():
    """The V2 checkpoint must remain the shipped FALLBACK, and is currently
    also the ACTIVE model (switched back from V4 on 2026-09-22)."""
    import inspect
    src = inspect.getsource(ns.NowcastService.__init__)
    assert "v2_calibrated_best.pt" in src
    assert "v2_calibrated_best.pt" in ns.FALLBACK_MODEL_CHECKPOINT


def test_active_model_is_v2():
    """The application's declared active model is the V2 checkpoint.

    Switched back from V4 to V2 on 2026-09-22 (user request), after a live
    side-by-side comparison on the same real GFS analysis showed V2's raw
    probability output was substantially higher and better spatially located
    inside West Bengal than V4's. V4 remains on disk, untouched, and stays
    reachable via STORMSENSE_CKPT/STORMSENSE_CONFIG for backtests and its own
    saved benchmark report.
    """
    assert ns.ACTIVE_MODEL_CHECKPOINT.endswith("v2_calibrated_best.pt")
    assert "checkpoints_v4" not in ns.ACTIVE_MODEL_CHECKPOINT
    assert ns.ACTIVE_MODEL_CONFIG is None  # resolves to configs/default.yaml


def test_demo_horizon_mapping_is_the_specified_one():
    """NOW/+2h/+4h/+6h are served by real V2 leads 3/4/5/6."""
    assert ns.DEMO_HORIZON_MAP == {0: 3, 2: 4, 4: 5, 6: 6}
    assert ns.DEMO_HORIZON_LABELS == {0: "NOW", 2: "+2h", 4: "+4h", 6: "+6h"}
