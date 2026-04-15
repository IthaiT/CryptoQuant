from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from src.data_handler.raw_data_downloader import RawDataDownloader


def _make_downloader(base_dir: Path, data_type: str) -> RawDataDownloader:
    return RawDataDownloader(
        data_type=data_type,
        symbol="BTCUSDT",
        start_date="2026-01-01",
        end_date="2026-01-02",
        interval="1m",
        base_dir=base_dir,
    )


@pytest.fixture
def aggtrades_downloader(tmp_path):
    return _make_downloader(tmp_path, "aggTrades")


@pytest.fixture
def klines_downloader(tmp_path):
    return _make_downloader(tmp_path, "klines")


def test_standardize_aggtrades_columns(aggtrades_downloader):
    df = pd.DataFrame(
        [
            {
                "trade_id": "10",
                "price": "100.5",
                "amount": "0.25",
                "first_trade_id": "10",
                "last_trade_id": "11",
                "timestamp": "1700000000000",
                "is_buyer_maker": "true",
                "is_best_match": "1",
            },
            {
                "trade_id": "12",
                "price": "101.0",
                "amount": "0.50",
                "first_trade_id": "12",
                "last_trade_id": "12",
                "timestamp": "1700000005000",
                "is_buyer_maker": "0",
                "is_best_match": "1",
            },
        ]
    )

    result = aggtrades_downloader._standardize_columns(df)

    assert list(result.columns) == [
        "timestamp",
        "price",
        "amount",
        "is_buyer_maker",
        "trade_id",
        "first_trade_id",
        "last_trade_id",
        "num_trades",
    ]
    assert result["num_trades"].tolist() == [2, 1]
    assert result["is_buyer_maker"].tolist() == [True, False]
    assert result["timestamp"].dtype.kind in "iu"


def test_standardize_klines_columns(klines_downloader):
    df = pd.DataFrame(
        [
            {
                "open_time": "1700000000000",
                "open": "100.0",
                "high": "102.0",
                "low": "99.0",
                "close": "101.0",
                "volume": "5.0",
                "close_time": "1700000059999",
                "quote_volume": "505.0",
                "count": "3",
                "taker_buy_volume": "2.0",
                "taker_buy_quote_volume": "202.0",
                "ignore": "0",
            }
        ]
    )

    result = klines_downloader._standardize_columns(df)

    assert result.loc[0, "open_time"] == 1_700_000_000_000
    assert result.loc[0, "close_time"] == 1_700_000_059_999
    assert result.loc[0, "open"] == 100.0
    assert result.loc[0, "count"] == 3


def test_save_compressed_writes_roundtrip_csv(tmp_path, aggtrades_downloader):
    df = pd.DataFrame(
        [
            {"timestamp": 1, "price": 10.0, "amount": 2.0},
            {"timestamp": 2, "price": 11.0, "amount": 3.0},
        ]
    )
    output_file = tmp_path / "roundtrip.csv.zst"

    aggtrades_downloader._save_compressed(df, output_file)

    assert output_file.exists()
    assert not output_file.with_suffix(".csv.zst.tmp").exists()

    dctx = zstd.ZstdDecompressor()
    with open(output_file, "rb") as f_in:
        with dctx.stream_reader(f_in) as reader:
            restored = pd.read_csv(BytesIO(reader.read()))

    pd.testing.assert_frame_equal(restored, df)
