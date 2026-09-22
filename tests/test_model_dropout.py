"""Spatial dropout must be strictly opt-in.

Dropout was added to SevereWeatherNetV2 because the long-lead (8-16h) training
run overfits from roughly epoch 3: train_loss fell monotonically 0.0155 ->
0.0082 while val_loss bottomed at epoch 2 (0.0159) and rose to 0.0214.

The production checkpoint was trained WITHOUT it, so the default must be a
no-op in every observable way -- same parameter count, same state_dict keys,
same outputs. These tests pin that, so a future default change cannot silently
alter the served model.
"""
import torch

from src.models.v2_model import SevereWeatherNetV2


def _batch(b=2):
    torch.manual_seed(0)
    return {
        "surface": torch.randn(b, 6, 15, 33, 25),
        "pressure_wind": torch.randn(b, 6, 2, 3, 33, 25),
        "pressure_thermo": torch.randn(b, 6, 3, 4, 33, 25),
        "dem": torch.randn(b, 3, 33, 25),
    }


def test_dropout_defaults_to_disabled():
    m = SevereWeatherNetV2(n_lead_times=5)
    assert m.dropout_p == 0.0
    assert isinstance(m.spatial_dropout, torch.nn.Identity), (
        "default must be a true no-op, not Dropout2d(p=0)"
    )


def test_dropout_adds_no_parameters():
    base = SevereWeatherNetV2(n_lead_times=5)
    reg = SevereWeatherNetV2(n_lead_times=5, dropout=0.15)
    assert base.count_parameters() == reg.count_parameters() == 781889
    assert base.state_dict().keys() == reg.state_dict().keys(), (
        "dropout must not change the state_dict, or existing checkpoints break"
    )


def test_existing_checkpoint_loads_strictly():
    """The shipped checkpoint must still load with strict=True."""
    import os
    p = os.path.join("Data", "outputs", "checkpoints", "v2_calibrated_best.pt")
    if not os.path.exists(p):
        import pytest
        pytest.skip("production checkpoint not present")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = SevereWeatherNetV2(n_lead_times=5)
    m.load_state_dict(ck["model"], strict=True)


def test_default_model_is_deterministic_in_train_mode():
    """With dropout off, train() and eval() must agree -- proving no stochastic
    path was introduced into the default configuration."""
    m = SevereWeatherNetV2(n_lead_times=5)
    b = _batch()
    m.eval()
    with torch.no_grad():
        a = m(b)["severe_weather_logit"]
    m.train()
    with torch.no_grad():
        c = m(b)["severe_weather_logit"]
    assert torch.equal(a, c)


def test_enabled_dropout_is_active_in_train_and_inert_in_eval():
    m = SevereWeatherNetV2(n_lead_times=5, dropout=0.5)
    b = _batch()

    m.train()
    torch.manual_seed(1)
    with torch.no_grad():
        x = m(b)["severe_weather_logit"]
    torch.manual_seed(2)
    with torch.no_grad():
        y = m(b)["severe_weather_logit"]
    assert not torch.equal(x, y), "dropout must perturb outputs while training"

    # Inference must never be stochastic.
    m.eval()
    with torch.no_grad():
        p = m(b)["severe_weather_logit"]
        q = m(b)["severe_weather_logit"]
    assert torch.equal(p, q), "dropout must be inert under eval()"
