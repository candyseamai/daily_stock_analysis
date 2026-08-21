# -*- coding: utf-8 -*-
"""Offline tests for the Yahoo A-share index daily adapter."""

import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_provider.base import IndexDailyProviderError
from data_provider.yfinance_fetcher import YfinanceFetcher


EXPECTED = date(2026, 3, 24)


def _download_frame(*, symbol=None, multi_index=False) -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-03-23", "2026-03-24"], name="Date")
    values = [
        [100.0, 102.0, 99.0, 101.0, 1000.0],
        [101.0, 103.0, 100.0, 102.0, 1100.0],
    ]
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if multi_index:
        columns = pd.MultiIndex.from_product(
            [fields, [symbol]], names=["Price", "Ticker"]
        )
    else:
        columns = fields
    return pd.DataFrame(values, index=index, columns=columns)


def _assert_category(expected_category, callable_):
    with pytest.raises(IndexDailyProviderError) as exc_info:
        callable_()
    assert exc_info.value.category == expected_category


def _run_with_download(index_code, returned_frame, download_mock=None):
    download = download_mock or Mock(return_value=returned_frame)
    fake_yfinance = SimpleNamespace(download=download)
    with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
        result = YfinanceFetcher().get_index_daily_data(
            index_code, EXPECTED, days=60
        )
    return result, download


@pytest.mark.parametrize(
    "index_code,symbol",
    [
        ("sh000001", "000001.SS"),
        ("sh000300", "000300.SS"),
        ("sz399006", "399006.SZ"),
    ],
)
def test_yahoo_uses_explicit_index_symbol_and_exclusive_end(index_code, symbol):
    result, download = _run_with_download(index_code, _download_frame())

    assert result is not None
    kwargs = download.call_args.kwargs
    assert kwargs["tickers"] == symbol
    assert kwargs["end"] == "2026-03-25"
    assert kwargs["progress"] is False
    assert kwargs["auto_adjust"] is True
    assert kwargs["multi_level_index"] is True


def test_yahoo_index_route_never_calls_stock_code_converter(monkeypatch):
    fetcher = YfinanceFetcher()
    monkeypatch.setattr(
        fetcher,
        "_convert_stock_code",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("stock converter must not be called")
        ),
    )
    fake_yfinance = SimpleNamespace(download=Mock(return_value=_download_frame()))

    with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
        result = fetcher.get_index_daily_data("sh000001", EXPECTED)

    assert result is not None


def test_yahoo_plain_columns_are_preliminarily_standardized_without_amount():
    result, _ = _run_with_download("sh000001", _download_frame())

    assert list(result.columns) == [
        "date", "open", "high", "low", "close", "volume", "amount"
    ]
    assert list(result["close"]) == [101.0, 102.0]
    assert result["amount"].isna().all()


def test_yahoo_multiindex_accepts_matching_ticker_and_flattens_price_level():
    symbol = "000300.SS"
    result, _ = _run_with_download(
        "sh000300", _download_frame(symbol=symbol, multi_index=True)
    )

    assert result is not None
    assert list(result["high"]) == [102.0, 103.0]
    assert result["amount"].isna().all()


def test_yahoo_multiindex_rejects_mismatched_ticker():
    _assert_category(
        "ticker_mismatch",
        lambda: _run_with_download(
            "sh000300", _download_frame(symbol="WRONG.SS", multi_index=True)
        ),
    )


def test_yahoo_multiindex_accepts_ticker_first_and_price_second():
    symbol = "399006.SZ"
    original = _download_frame(symbol=symbol, multi_index=True)
    reversed_levels = original.copy()
    reversed_levels.columns = reversed_levels.columns.swaplevel(0, 1)

    result, _ = _run_with_download("sz399006", reversed_levels)

    assert list(result["close"]) == [101.0, 102.0]


def test_yahoo_multiindex_rejects_multiple_tickers_even_when_requested_is_present():
    symbol = "000001.SS"
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product(
        [fields, [symbol, "OTHER.SS"]], names=["Price", "Ticker"]
    )
    frame = pd.DataFrame(
        [[100.0] * len(columns)],
        index=pd.DatetimeIndex(["2026-03-24"], name="Date"),
        columns=columns,
    )

    _assert_category(
        "ticker_mismatch",
        lambda: _run_with_download("sh000001", frame),
    )


def test_yahoo_multiindex_rejects_duplicate_price_fields():
    symbol = "000001.SS"
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", symbol),
            ("Open", symbol),
            ("High", symbol),
            ("Low", symbol),
            ("Close", symbol),
        ],
        names=["Price", "Ticker"],
    )
    frame = pd.DataFrame(
        [[100.0, 100.0, 102.0, 99.0, 101.0]],
        index=pd.DatetimeIndex(["2026-03-24"], name="Date"),
        columns=columns,
    )

    _assert_category(
        "duplicate_columns",
        lambda: _run_with_download("sh000001", frame),
    )


def test_yahoo_three_level_multiindex_accepts_single_ticker_metadata_level():
    symbol = "000300.SS"
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product(
        [fields, [symbol], ["daily"]],
        names=["Price", "Ticker", "Interval"],
    )
    frame = pd.DataFrame(
        [[100.0, 102.0, 99.0, 101.0, 1000.0]],
        index=pd.DatetimeIndex(["2026-03-24"], name="Date"),
        columns=columns,
    )

    result, _ = _run_with_download("sh000300", frame)

    assert result.iloc[0]["close"] == 101.0


@pytest.mark.parametrize(
    "returned,category",
    [
        (None, "empty"),
        (pd.DataFrame(), "empty"),
        (pd.DataFrame({"Unexpected": [1]}), "invalid_schema"),
    ],
)
def test_yahoo_empty_or_invalid_schema_raises_structured_failure(returned, category):
    _assert_category(
        category,
        lambda: _run_with_download("sh000001", returned),
    )


def test_yahoo_network_exception_returns_failure_without_real_network():
    download = Mock(side_effect=ConnectionError("offline"))
    _assert_category(
        "download_error",
        lambda: _run_with_download("sh000001", None, download),
    )
    called = download
    called.assert_called_once()


def test_yahoo_plain_date_column_with_range_index_is_supported():
    frame = _download_frame().reset_index()
    result, _ = _run_with_download("sh000001", frame)
    assert list(result["date"].dt.date) == [date(2026, 3, 23), EXPECTED]


def test_yahoo_unnamed_datetime_index_is_used_as_date():
    frame = _download_frame()
    frame.index.name = None
    result, _ = _run_with_download("sh000001", frame)
    assert result.iloc[-1]["date"].date() == EXPECTED


def test_yahoo_timezone_aware_index_is_preserved_for_public_normalizer():
    frame = _download_frame()
    frame.index = frame.index.tz_localize("UTC")
    result, _ = _run_with_download("sh000001", frame)
    assert str(result["date"].dt.tz) == "UTC"


def test_existing_stock_daily_entry_remains_operational(monkeypatch):
    fetcher = YfinanceFetcher()
    raw = pd.DataFrame({"raw": [1]})
    normalized = pd.DataFrame({
        "date": ["2026-03-24"],
        "open": [100.0],
        "high": [102.0],
        "low": [99.0],
        "close": [101.0],
        "volume": [1000.0],
        "amount": [2000.0],
        "pct_chg": [1.0],
    })
    monkeypatch.setattr(fetcher, "_fetch_raw_data", Mock(return_value=raw))
    monkeypatch.setattr(fetcher, "_normalize_data", Mock(return_value=normalized))

    result = fetcher.get_daily_data(
        "600519", start_date="2026-03-24", end_date="2026-03-24"
    )

    fetcher._fetch_raw_data.assert_called_once_with(
        "600519", "2026-03-24", "2026-03-24"
    )
    assert result.iloc[0]["close"] == 101.0
    assert result.iloc[0]["ma5"] == 101.0


def test_existing_600519_stock_code_conversion_is_unchanged():
    assert YfinanceFetcher()._convert_stock_code("600519") == "600519.SS"
