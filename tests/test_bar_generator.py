from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from src.data_handler.bar_generator import BarGenerator, BarGeneratorUtil
from src.data_handler.bar_rule import TickBarRule, VolumeBarRule


def _write_raw_day(base_dir: Path, symbol: str, date: str, rows: list[dict]) -> None:
    year = date[:4]
    month = date[5:7]
    output_path = base_dir / symbol / "aggTrades" / year / month / f"{symbol}-aggTrades-{date}.csv.zst"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    cctx = zstd.ZstdCompressor(level=1)
    with open(output_path, "wb") as f_out:
        with cctx.stream_writer(f_out, closefd=False) as writer:
            writer.write(df.to_csv(index=False).encode("utf-8"))


@pytest.fixture
def symbol():
    return "BTCUSDT"


def test_get_date_range_returns_contiguous_available_dates(tmp_path, symbol):
    _write_raw_day(tmp_path, symbol, "2026-01-01", [{"timestamp": 1, "price": 10.0, "amount": 1.0}])
    _write_raw_day(tmp_path, symbol, "2026-01-02", [{"timestamp": 2, "price": 11.0, "amount": 1.5}])

    date_range = BarGeneratorUtil.get_date_range(tmp_path, symbol, "2026-01-01", "2026-01-03")

    assert date_range == ["2026-01-01", "2026-01-02"]


def test_generate_volume_bars_can_span_multiple_days(tmp_path, symbol):
    _write_raw_day(
        tmp_path,
        symbol,
        "2026-01-01",
        [
            {"timestamp": 1, "price": 10.0, "amount": 2.0},
            {"timestamp": 2, "price": 12.0, "amount": 3.0},
        ],
    )
    _write_raw_day(
        tmp_path,
        symbol,
        "2026-01-02",
        [
            {"timestamp": 3, "price": 11.0, "amount": 1.0},
            {"timestamp": 4, "price": 13.0, "amount": 4.0},
            {"timestamp": 5, "price": 14.0, "amount": 2.0},
        ],
    )

    generator = BarGenerator(base_dir=tmp_path)
    result = generator.generate(
        bar_rule=VolumeBarRule(threshold=6.0),
        symbol=symbol,
        start_date="2026-01-01",
        end_date="2026-01-02",
        show_progress=False,
    )

    assert len(result) == 2

    first_bar = result.iloc[0].to_dict()
    second_bar = result.iloc[1].to_dict()

    assert first_bar["timestamp"] == 1
    assert first_bar["close_time"] == 3
    assert first_bar["open"] == 10.0
    assert first_bar["high"] == 12.0
    assert first_bar["low"] == 10.0
    assert first_bar["close"] == 11.0
    assert first_bar["volume"] == 6.0
    assert first_bar["num_trades"] == 3

    assert second_bar["timestamp"] == 4
    assert second_bar["close_time"] == 5
    assert second_bar["open"] == 13.0
    assert second_bar["high"] == 14.0
    assert second_bar["low"] == 13.0
    assert second_bar["close"] == 14.0
    assert second_bar["volume"] == 6.0
    assert second_bar["num_trades"] == 2


def test_generate_by_day_yields_one_dataframe_per_period(tmp_path, symbol):
    _write_raw_day(
        tmp_path,
        symbol,
        "2026-01-01",
        [
            {"timestamp": 1, "price": 10.0, "amount": 2.0},
            {"timestamp": 2, "price": 12.0, "amount": 3.0},
        ],
    )
    _write_raw_day(
        tmp_path,
        symbol,
        "2026-01-02",
        [
            {"timestamp": 3, "price": 11.0, "amount": 1.0},
            {"timestamp": 4, "price": 13.0, "amount": 4.0},
        ],
    )

    generator = BarGenerator(base_dir=tmp_path)
    period_results = list(
        generator.generate(
            bar_rule=TickBarRule(threshold=2),
            symbol=symbol,
            start_date="2026-01-01",
            end_date="2026-01-02",
            period="day",
            show_progress=False,
        )
    )

    assert [period_key for period_key, _ in period_results] == ["2026-01-01", "2026-01-02"]
    assert [len(df) for _, df in period_results] == [1, 1]
    assert period_results[0][1].iloc[0]["close_time"] == 2
    assert period_results[1][1].iloc[0]["close_time"] == 4
