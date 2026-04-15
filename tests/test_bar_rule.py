import pytest

from src.data_handler.bar_rule import DollarBarRule, TickBarRule, VolumeBarRule


def test_volume_bar_rule_updates_ohlcv_and_threshold():
    rule = VolumeBarRule(threshold=5.0)

    bar = rule.init_bar(ts=1, price=10.0, amount=2.0, previous_bars=[])
    rule.update_bar(bar, ts=2, price=12.0, amount=3.0)

    assert bar == {
        "timestamp": 1,
        "open": 10.0,
        "high": 12.0,
        "low": 10.0,
        "close": 12.0,
        "volume": 5.0,
        "dollar_volume": 56.0,
        "num_trades": 2,
    }
    assert rule.should_close(bar) is True

    rule.reset()
    assert rule.cumulative == 0.0


def test_tick_bar_rule_counts_trades():
    rule = TickBarRule(threshold=3)

    bar = rule.init_bar(ts=1, price=10.0, amount=1.0, previous_bars=[])
    rule.update_bar(bar, ts=2, price=11.0, amount=2.0)
    assert rule.should_close(bar) is False

    rule.update_bar(bar, ts=3, price=9.0, amount=1.5)

    assert bar["high"] == 11.0
    assert bar["low"] == 9.0
    assert bar["close"] == 9.0
    assert bar["num_trades"] == 3
    assert rule.should_close(bar) is True

    rule.reset()
    assert rule.trade_count == 0


def test_dollar_bar_rule_updates_dynamic_threshold_after_four_hours():
    rule = DollarBarRule(threshold=200.0)
    ts = 1_700_000_000_000
    four_hours_ms = 4 * 60 * 60 * 1_000

    rule.last_threshold_update_ts = ts - four_hours_ms
    previous_bars = [
        {"timestamp": ts - 1_000, "dollar_volume": 100.0}
        for _ in range(rule.update_interval)
    ]

    bar = rule.init_bar(ts=ts, price=10.0, amount=1.0, previous_bars=previous_bars)

    expected_threshold = (100.0 * rule.alpha) + (200.0 * (1 - rule.alpha))
    assert rule.threshold == pytest.approx(expected_threshold)
    assert bar["dollar_volume"] == 10.0
    assert rule.cumulative == 10.0
