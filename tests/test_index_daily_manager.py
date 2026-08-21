# -*- coding: utf-8 -*-
"""Offline tests for the dedicated A-share index daily manager route."""

from datetime import date, datetime

import pandas as pd
import pytest

import data_provider.base as base_module
from data_provider.base import (
    BaseFetcher,
    DataFetchError,
    DataFetcherManager,
    IndexDailyProviderError,
)
from data_provider.cn_index_daily import INDEX_DAILY_STANDARD_COLUMNS


EXPECTED = date(2026, 3, 24)


def _provider_bars(session: date = EXPECTED, count: int = 120) -> pd.DataFrame:
    sessions = pd.bdate_range(end=session, periods=count)
    return pd.DataFrame({
        "date": sessions,
        "open": [100.0 + index for index in range(count)],
        "high": [103.0 + index for index in range(count)],
        "low": [99.0 + index for index in range(count)],
        "close": [102.0 + index for index in range(count)],
        "volume": [1000.0 + index for index in range(count)],
    })


class _MinimalFetcher(BaseFetcher):
    def _fetch_raw_data(self, stock_code, start_date, end_date):
        return pd.DataFrame()

    def _normalize_data(self, df, stock_code):
        return df


class _FakeIndexFetcher:
    def __init__(self, name, outcome, calls, *, priority=0):
        self.name = name
        self.priority = priority
        self.outcome = outcome
        self.calls = calls

    def get_index_daily_data(self, index_code, expected_session, days=120):
        self.calls.append((self.name, index_code, expected_session, days))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def get_daily_data(self, *args, **kwargs):
        raise AssertionError("ordinary stock daily route must not be called")


def _manager(*fetchers) -> DataFetcherManager:
    if not fetchers:
        fetchers = (_FakeIndexFetcher("UnrelatedFetcher", None, []),)
    return DataFetcherManager(fetchers=list(fetchers))


def test_base_fetcher_default_index_daily_capability_is_unsupported():
    assert _MinimalFetcher().get_index_daily_data("sh000001", EXPECTED, 60) is None


def test_explicit_empty_fetcher_list_preserves_default_initialization_contract(monkeypatch):
    initialized = []

    def fake_init_default_fetchers(manager):
        initialized.append(manager)
        manager._fetchers = [_FakeIndexFetcher("YfinanceFetcher", None, [])]
        manager._refresh_fetcher_indexes_locked()

    monkeypatch.setattr(
        DataFetcherManager,
        "_init_default_fetchers",
        fake_init_default_fetchers,
    )

    manager = DataFetcherManager(fetchers=[])

    assert initialized == [manager]
    assert [item.name for item in manager._get_fetchers_snapshot()] == [
        "YfinanceFetcher"
    ]


def test_manager_preserves_prefix_and_never_normalizes_or_calls_stock_daily(monkeypatch):
    calls = []
    fetcher = _FakeIndexFetcher("YfinanceFetcher", _provider_bars(), calls)
    monkeypatch.setattr(
        base_module,
        "normalize_stock_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normalize_stock_code must not be called")
        ),
    )

    result, provider = _manager(fetcher).get_index_daily_data(
        "sh000001", EXPECTED, days=90
    )

    assert provider == "YfinanceFetcher"
    assert calls == [("YfinanceFetcher", "sh000001", EXPECTED, 90)]
    assert len(result) == 90
    assert result.attrs["index_code"] == "sh000001"


def test_manager_uses_fixed_provider_order_independent_of_stock_priority():
    calls = []
    yfinance = _FakeIndexFetcher("YfinanceFetcher", None, calls, priority=99)
    tushare = _FakeIndexFetcher(
        "TushareFetcher", _provider_bars(), calls, priority=-10
    )

    _, provider = _manager(tushare, yfinance).get_index_daily_data(
        "sh000300", EXPECTED
    )

    assert provider == "TushareFetcher"
    assert [call[0] for call in calls] == ["YfinanceFetcher", "TushareFetcher"]


def test_manager_continues_after_first_provider_exception():
    calls = []
    yfinance = _FakeIndexFetcher(
        "YfinanceFetcher", RuntimeError("provider failed"), calls
    )
    tushare = _FakeIndexFetcher("TushareFetcher", _provider_bars(), calls)

    _, provider = _manager(yfinance, tushare).get_index_daily_data(
        "sz399006", EXPECTED
    )

    assert provider == "TushareFetcher"
    assert [call[0] for call in calls] == ["YfinanceFetcher", "TushareFetcher"]


@pytest.mark.parametrize(
    "bad_result",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"date": [EXPECTED], "close": [100.0]}),
    ],
)
def test_manager_continues_after_none_empty_or_invalid_schema(bad_result):
    calls = []
    yfinance = _FakeIndexFetcher("YfinanceFetcher", bad_result, calls)
    tushare = _FakeIndexFetcher("TushareFetcher", _provider_bars(), calls)

    _, provider = _manager(yfinance, tushare).get_index_daily_data(
        "sh000001", EXPECTED
    )

    assert provider == "TushareFetcher"


def test_stale_provider_is_rejected_and_fallback_continues():
    calls = []
    yfinance = _FakeIndexFetcher(
        "YfinanceFetcher", _provider_bars(date(2026, 3, 23)), calls
    )
    tushare = _FakeIndexFetcher("TushareFetcher", _provider_bars(), calls)

    result, provider = _manager(yfinance, tushare).get_index_daily_data(
        "sh000001", EXPECTED
    )

    assert provider == "TushareFetcher"
    assert result.iloc[-1]["date"].date() == EXPECTED


def test_insufficient_bars_fall_back_and_success_returns_latest_requested_count():
    calls = []
    yfinance = _FakeIndexFetcher(
        "YfinanceFetcher", _provider_bars(count=59), calls
    )
    tushare = _FakeIndexFetcher(
        "TushareFetcher", _provider_bars(count=80), calls
    )

    result, provider = _manager(yfinance, tushare).get_index_daily_data(
        "sh000001", EXPECTED, days=60
    )

    assert provider == "TushareFetcher"
    assert len(result) == 60
    assert result.iloc[-1]["date"].date() == EXPECTED
    assert [call[0] for call in calls] == ["YfinanceFetcher", "TushareFetcher"]


def test_all_provider_failures_raise_sanitized_provider_summary():
    calls = []
    fetchers = [
        _FakeIndexFetcher(
            "YfinanceFetcher",
            IndexDailyProviderError("token=VERY_SECRET_VALUE"),
            calls,
        ),
        _FakeIndexFetcher(
            "TushareFetcher", IndexDailyProviderError("download_error"), calls
        ),
        _FakeIndexFetcher(
            "AkshareFetcher",
            pd.DataFrame({"date": [EXPECTED], "close": [100.0]}),
            calls,
        ),
    ]

    with pytest.raises(DataFetchError) as exc_info:
        _manager(*fetchers).get_index_daily_data("sh000001", EXPECTED)

    message = str(exc_info.value)
    assert "YfinanceFetcher" in message
    assert "TushareFetcher" in message
    assert "AkshareFetcher" in message
    assert "download_error" in message
    assert "VERY_SECRET_VALUE" not in message


def test_success_returns_standard_columns_identity_attrs_and_provider():
    calls = []
    result, provider = _manager(
        _FakeIndexFetcher("YfinanceFetcher", _provider_bars(), calls)
    ).get_index_daily_data("sh000300", EXPECTED)

    assert tuple(result.columns) == INDEX_DAILY_STANDARD_COLUMNS
    assert provider == "YfinanceFetcher"
    assert result.attrs["asset_type"] == "index"
    assert result.attrs["index_code"] == "sh000300"
    assert result.attrs["index_name"] == "沪深300"
    assert result.attrs["provider"] == provider
    assert result.attrs["provider_symbol"] == "000300.SS"


def test_manager_standardization_removes_future_bars():
    calls = []
    frame = pd.concat(
        [_provider_bars(), _provider_bars(date(2026, 3, 25), count=1)],
        ignore_index=True,
    )

    result, _ = _manager(
        _FakeIndexFetcher("YfinanceFetcher", frame, calls)
    ).get_index_daily_data("sh000001", EXPECTED)

    assert list(result["date"].dt.date) == [EXPECTED]
    assert "future_bars_removed:1" in result.attrs["warnings"]


@pytest.mark.parametrize(
    "index_code,expected_session,days,error",
    [
        ("000001", EXPECTED, 120, ValueError),
        ("unknown", EXPECTED, 120, ValueError),
        ("sh000001", datetime(2026, 3, 24), 120, ValueError),
        ("sh000001", EXPECTED, 0, ValueError),
        ("sh000001", EXPECTED, True, ValueError),
    ],
)
def test_manager_rejects_noncanonical_or_invalid_arguments(
    index_code, expected_session, days, error
):
    with pytest.raises(error):
        _manager().get_index_daily_data(index_code, expected_session, days)
