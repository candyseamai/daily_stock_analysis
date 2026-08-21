# -*- coding: utf-8 -*-
"""Offline tests for the premarket three-index batch orchestrator."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.services.premarket_index_service as service_module
from src.services.index_technical_level_service import DataQualityStatus
from src.services.premarket_index_service import (
    PREMARKET_INDEX_CODES,
    PremarketIndexBatchStatus,
    PremarketIndexItemStatus,
    build_premarket_index_batch,
)
from src.services.premarket_window_service import (
    PremarketWindow,
    PremarketWindowStatus,
)


EXPECTED = date(2026, 3, 23)
SHANGHAI = ZoneInfo("Asia/Shanghai")
INDEX_NAMES = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sz399006": "创业板指",
}


def _ready_window(**overrides) -> PremarketWindow:
    values = {
        "status": PremarketWindowStatus.READY,
        "run_at": datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI),
        "session_date": date(2026, 3, 24),
        "previous_trading_date": EXPECTED,
        "window_start": datetime(2026, 3, 23, 15, 0, tzinfo=SHANGHAI),
        "window_end": datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI),
    }
    values.update(overrides)
    return PremarketWindow(**values)


def _frame(code: str, *, name: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": [EXPECTED],
        "open": [100.0],
        "high": [102.0],
        "low": [99.0],
        "close": [101.0],
        "volume": [1000.0],
    })
    frame.attrs.update({
        "asset_type": "index",
        "index_code": code,
        "index_name": name or INDEX_NAMES[code],
    })
    return frame


def _technical(code: str, quality: DataQualityStatus):
    return SimpleNamespace(
        code=code,
        quality_status=quality,
        warnings=("source_warning",) if quality is DataQualityStatus.PARTIAL else (),
        missing_reasons=("insufficient_history_for_ma60",)
        if quality is DataQualityStatus.PARTIAL
        else (),
    )


class _Manager:
    def __init__(self, outcomes, calls):
        self.outcomes = outcomes
        self.calls = calls

    def get_index_daily_data(self, index_code, expected_session, days=120):
        self.calls.append((index_code, expected_session, days))
        outcome = self.outcomes[index_code]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_calculator(monkeypatch, qualities=None, calls=None):
    qualities = qualities or {
        code: DataQualityStatus.OK for code in PREMARKET_INDEX_CODES
    }
    calls = calls if calls is not None else []

    def fake_calculator(code, name, daily_data, expected_session):
        calls.append((code, name, expected_session, daily_data))
        return _technical(code, qualities[code])

    monkeypatch.setattr(
        service_module,
        "calculate_index_technical_levels",
        fake_calculator,
    )
    return calls


def _successful_outcomes():
    return {
        "sh000001": (_frame("sh000001"), "YfinanceFetcher"),
        "sh000300": (_frame("sh000300"), "AkshareFetcher"),
        "sz399006": (_frame("sz399006"), "YfinanceFetcher"),
    }


def test_three_indices_complete_in_fixed_order_and_preserve_providers(monkeypatch):
    manager_calls = []
    calculator_calls = _patch_calculator(monkeypatch)
    result = build_premarket_index_batch(
        _ready_window(),
        manager=_Manager(_successful_outcomes(), manager_calls),
    )

    assert result.status is PremarketIndexBatchStatus.COMPLETE
    assert result.expected_session == EXPECTED
    assert result.success_count == 3
    assert result.failure_count == 0
    assert tuple(item.index_code for item in result.items) == PREMARKET_INDEX_CODES
    assert tuple(item.provider for item in result.items) == (
        "YfinanceFetcher", "AkshareFetcher", "YfinanceFetcher"
    )
    assert [call[0] for call in manager_calls] == list(PREMARKET_INDEX_CODES)
    assert [call[0] for call in calculator_calls] == list(PREMARKET_INDEX_CODES)


def test_expected_session_and_days_are_forwarded_exactly(monkeypatch):
    calls = []
    _patch_calculator(monkeypatch)
    build_premarket_index_batch(
        _ready_window(),
        manager=_Manager(_successful_outcomes(), calls),
        days=77,
    )
    assert calls == [(code, EXPECTED, 77) for code in PREMARKET_INDEX_CODES]


def test_middle_index_failure_does_not_stop_last_index(monkeypatch):
    calls = []
    outcomes = _successful_outcomes()
    outcomes["sh000300"] = RuntimeError("token=VERY_SECRET_VALUE")
    _patch_calculator(monkeypatch)

    result = build_premarket_index_batch(
        _ready_window(), manager=_Manager(outcomes, calls)
    )

    assert result.status is PremarketIndexBatchStatus.PARTIAL
    assert [call[0] for call in calls] == list(PREMARKET_INDEX_CODES)
    assert result.items[1].status is PremarketIndexItemStatus.FAILED
    assert result.items[1].error_category == "data_fetch_failed"
    assert "VERY_SECRET_VALUE" not in repr(result)


@pytest.mark.parametrize(
    "attr_name,bad_value",
    [
        ("asset_type", "stock"),
        ("index_code", "sh000001"),
        ("index_name", "错误名称"),
    ],
)
def test_attrs_identity_mismatch_fails_before_calculation(
    monkeypatch, attr_name, bad_value
):
    outcomes = _successful_outcomes()
    mismatched = _frame("sh000300")
    mismatched.attrs[attr_name] = bad_value
    outcomes["sh000300"] = (mismatched, "AkshareFetcher")
    calculator_calls = _patch_calculator(monkeypatch)

    result = build_premarket_index_batch(
        _ready_window(), manager=_Manager(outcomes, [])
    )

    assert result.items[1].error_category == "identity_mismatch"
    assert [call[0] for call in calculator_calls] == ["sh000001", "sz399006"]


@pytest.mark.parametrize(
    "quality,expected_status,expected_error",
    [
        (DataQualityStatus.OK, PremarketIndexItemStatus.SUCCESS, None),
        (DataQualityStatus.PARTIAL, PremarketIndexItemStatus.SUCCESS, None),
        (
            DataQualityStatus.STALE,
            PremarketIndexItemStatus.FAILED,
            "technical_quality_stale",
        ),
        (
            DataQualityStatus.INSUFFICIENT,
            PremarketIndexItemStatus.FAILED,
            "technical_quality_insufficient",
        ),
    ],
)
def test_quality_gate(monkeypatch, quality, expected_status, expected_error):
    qualities = {code: DataQualityStatus.OK for code in PREMARKET_INDEX_CODES}
    qualities["sh000001"] = quality
    _patch_calculator(monkeypatch, qualities)

    result = build_premarket_index_batch(
        _ready_window(), manager=_Manager(_successful_outcomes(), [])
    )
    item = result.items[0]

    assert item.status is expected_status
    assert item.error_category == expected_error
    if quality is DataQualityStatus.PARTIAL:
        assert "technical_quality_partial" in item.warnings
        assert "sh000001:technical_quality_partial" in result.warnings


@pytest.mark.parametrize(
    "success_count,expected_batch_status",
    [
        (3, PremarketIndexBatchStatus.COMPLETE),
        (2, PremarketIndexBatchStatus.PARTIAL),
        (1, PremarketIndexBatchStatus.PARTIAL),
        (0, PremarketIndexBatchStatus.FAILED),
    ],
)
def test_batch_status_for_success_count(monkeypatch, success_count, expected_batch_status):
    qualities = {
        code: DataQualityStatus.OK if position < success_count else DataQualityStatus.STALE
        for position, code in enumerate(PREMARKET_INDEX_CODES)
    }
    _patch_calculator(monkeypatch, qualities)

    result = build_premarket_index_batch(
        _ready_window(), manager=_Manager(_successful_outcomes(), [])
    )

    assert result.status is expected_batch_status
    assert result.success_count == success_count
    assert result.failure_count == 3 - success_count


@pytest.mark.parametrize(
    "window,expected_status,category",
    [
        (
            _ready_window(status=PremarketWindowStatus.SKIPPED_NON_TRADING_DAY),
            PremarketIndexBatchStatus.SKIPPED,
            "non_trading_day",
        ),
        (
            _ready_window(status=PremarketWindowStatus.CALENDAR_UNAVAILABLE),
            PremarketIndexBatchStatus.FAILED,
            "calendar_unavailable",
        ),
        (
            _ready_window(previous_trading_date=None),
            PremarketIndexBatchStatus.FAILED,
            "incomplete_window",
        ),
    ],
)
def test_invalid_window_never_calls_manager(window, expected_status, category):
    manager = _Manager({}, [])
    result = build_premarket_index_batch(window, manager=manager)
    assert result.status is expected_status
    assert result.error_category == category
    assert manager.calls == []
    assert result.items == ()


def test_loader_injection_is_supported_and_results_are_frozen(monkeypatch):
    calls = []
    _patch_calculator(monkeypatch)

    def loader(code, expected_session, days):
        calls.append((code, expected_session, days))
        return _successful_outcomes()[code]

    result = build_premarket_index_batch(_ready_window(), loader=loader)

    assert len(calls) == 3
    assert isinstance(result.items, tuple)
    assert isinstance(result.warnings, tuple)
    with pytest.raises(FrozenInstanceError):
        result.success_count = 0
    with pytest.raises(FrozenInstanceError):
        result.items[0].provider = "changed"
