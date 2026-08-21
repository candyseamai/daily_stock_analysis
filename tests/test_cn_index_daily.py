# -*- coding: utf-8 -*-
"""Offline tests for canonical A-share index daily normalization."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_provider.cn_index_daily import (
    CN_INDEX_IDENTITIES,
    INDEX_DAILY_STANDARD_COLUMNS,
    IndexDailyIdentityError,
    IndexDailyNormalizationError,
    get_cn_index_identity,
    get_cn_index_provider_symbol,
    normalize_cn_index_daily_data,
)


EXPECTED = date(2026, 3, 24)


def _bars() -> pd.DataFrame:
    return pd.DataFrame({
        "date": [date(2026, 3, 23), date(2026, 3, 24)],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000.0, 1100.0],
        "amount": [2000.0, 2100.0],
        "pct_chg": [0.5, 0.99],
    })


def _normalize(frame: pd.DataFrame, **overrides) -> pd.DataFrame:
    kwargs = {
        "index_code": "sh000001",
        "index_name": "上证指数",
        "provider": "YfinanceFetcher",
        "expected_session": EXPECTED,
    }
    kwargs.update(overrides)
    return normalize_cn_index_daily_data(frame, **kwargs)


@pytest.mark.parametrize(
    "code,name,akshare,yahoo,tushare",
    [
        ("sh000001", "上证指数", "sh000001", "000001.SS", "000001.SH"),
        ("sh000300", "沪深300", "sh000300", "000300.SS", "000300.SH"),
        ("sz399006", "创业板指", "sz399006", "399006.SZ", "399006.SZ"),
    ],
)
def test_canonical_identity_and_all_provider_mappings(code, name, akshare, yahoo, tushare):
    identity = get_cn_index_identity(code, name)
    assert (identity.code, identity.name) == (code, name)
    assert get_cn_index_provider_symbol(code, "AkshareFetcher", name) == akshare
    assert get_cn_index_provider_symbol(code, "YfinanceFetcher", name) == yahoo
    assert get_cn_index_provider_symbol(code, "TushareFetcher", name) == tushare


def test_identity_mapping_and_values_are_immutable():
    with pytest.raises(TypeError):
        CN_INDEX_IDENTITIES["sh000001"] = CN_INDEX_IDENTITIES["sh000001"]
    with pytest.raises(FrozenInstanceError):
        CN_INDEX_IDENTITIES["sh000001"].name = "错误名称"


@pytest.mark.parametrize(
    "code",
    ["000001", "000300", "399006", "000001.SH", "000001.SS", "unknown"],
)
def test_bare_provider_and_unknown_codes_are_rejected(code):
    with pytest.raises(IndexDailyIdentityError, match="unsupported canonical"):
        get_cn_index_identity(code)


def test_name_mismatch_and_unknown_provider_are_rejected():
    with pytest.raises(IndexDailyIdentityError, match="identity mismatch"):
        get_cn_index_identity("sh000300", "上证指数")
    with pytest.raises(IndexDailyIdentityError, match="unsupported.*provider"):
        get_cn_index_provider_symbol("sh000001", "UnknownFetcher", "上证指数")


def test_standard_columns_are_strict_and_ordered():
    result = _normalize(_bars())
    assert tuple(result.columns) == INDEX_DAILY_STANDARD_COLUMNS


def test_missing_optional_columns_are_added_without_fabricating_amount():
    result = _normalize(_bars().drop(columns=["volume", "amount", "pct_chg"]))
    assert result["volume"].isna().all()
    assert result["amount"].isna().all()
    assert pd.isna(result.iloc[0]["pct_chg"])
    assert result.iloc[1]["pct_chg"] == pytest.approx((102 / 101 - 1) * 100)
    assert "missing_amount" in result.attrs["warnings"]
    assert "pct_chg_calculated_from_close" in result.attrs["warnings"]


def test_input_dataframe_is_not_modified():
    frame = _bars().drop(columns=["volume", "amount", "pct_chg"])
    original = frame.copy(deep=True)
    _normalize(frame)
    assert_frame_equal(frame, original)


def test_reverse_dates_are_sorted_ascending():
    result = _normalize(_bars().iloc[::-1])
    assert list(result["date"].dt.date) == [date(2026, 3, 23), date(2026, 3, 24)]


def test_duplicate_session_keeps_latest_shanghai_time_then_input_order():
    frame = pd.concat([_bars().iloc[[0]], _bars().iloc[[1]], _bars().iloc[[1]]], ignore_index=True)
    frame.loc[1, "date"] = datetime(2026, 3, 24, 9, 0)
    frame.loc[2, "date"] = datetime(2026, 3, 24, 15, 0)
    frame.loc[1, "close"] = 102.0
    frame.loc[2, "close"] = 102.5
    frame.loc[2, "high"] = 103.5
    result = _normalize(frame)
    assert result.iloc[-1]["close"] == 102.5
    assert "duplicate_sessions_kept_last:1" in result.attrs["warnings"]


def test_utc_timestamp_crossing_shanghai_midnight_uses_shanghai_session():
    frame = _bars().iloc[[0]].copy()
    frame.loc[frame.index[0], "date"] = datetime(2026, 3, 23, 16, 30, tzinfo=timezone.utc)
    result = _normalize(frame)
    assert result.iloc[0]["date"].date() == date(2026, 3, 24)


def test_expected_session_rejects_datetime():
    with pytest.raises(ValueError, match="pure Python date"):
        _normalize(_bars(), expected_session=datetime(2026, 3, 24))


def test_future_bars_are_removed_after_shanghai_date_normalization():
    future = _bars().iloc[[-1]].copy()
    future["date"] = date(2026, 3, 25)
    result = _normalize(pd.concat([_bars(), future], ignore_index=True))
    assert result["date"].dt.date.max() == EXPECTED
    assert "future_bars_removed:1" in result.attrs["warnings"]


@pytest.mark.parametrize(
    "column,value,warning",
    [
        ("high", "bad", "non_finite_ohlc_rows_removed:1"),
        ("high", float("nan"), "non_finite_ohlc_rows_removed:1"),
        ("high", float("inf"), "non_finite_ohlc_rows_removed:1"),
        ("high", float("-inf"), "non_finite_ohlc_rows_removed:1"),
        ("low", 0, "non_positive_ohlc_rows_removed:1"),
        ("high", 50, "invalid_ohlc_relationship_rows_removed:1"),
        ("low", 500, "invalid_ohlc_relationship_rows_removed:1"),
    ],
)
def test_invalid_ohlc_rows_are_removed(column, value, warning):
    frame = _bars()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    result = _normalize(frame)
    assert len(result) == 1
    assert warning in result.attrs["warnings"]


@pytest.mark.parametrize(
    "volume",
    [None, float("nan"), float("inf"), float("-inf"), -1],
)
def test_invalid_volume_does_not_remove_valid_ohlc(volume):
    frame = _bars()
    if volume is None:
        frame = frame.drop(columns="volume")
    else:
        frame.loc[0, "volume"] = volume
    result = _normalize(frame)
    assert len(result) == 2
    assert pd.isna(result.iloc[0]["volume"])


def test_attrs_are_derived_from_validated_identity_and_provider():
    result = _normalize(_bars())
    assert result.attrs["asset_type"] == "index"
    assert result.attrs["index_code"] == "sh000001"
    assert result.attrs["index_name"] == "上证指数"
    assert result.attrs["provider"] == "YfinanceFetcher"
    assert result.attrs["provider_symbol"] == "000001.SS"
    assert isinstance(result.attrs["warnings"], tuple)


def test_all_invalid_rows_raise_instead_of_returning_empty_success():
    frame = _bars()
    frame["close"] = 0
    with pytest.raises(IndexDailyNormalizationError, match="no valid daily bars"):
        _normalize(frame)
