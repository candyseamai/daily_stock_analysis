# -*- coding: utf-8 -*-
"""Batch orchestration for premarket A-share index technical levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Optional, Protocol

import pandas as pd

from data_provider.base import DataFetcherManager
from data_provider.cn_index_daily import get_cn_index_identity
from src.services.index_technical_level_service import (
    DataQualityStatus,
    IndexTechnicalLevels,
    calculate_index_technical_levels,
)
from src.services.premarket_window_service import (
    PremarketWindow,
    PremarketWindowStatus,
)


PREMARKET_INDEX_CODES = ("sh000001", "sh000300", "sz399006")


class PremarketIndexItemStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class PremarketIndexBatchStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PremarketIndexItemResult:
    index_code: str
    index_name: str
    status: PremarketIndexItemStatus
    provider: Optional[str]
    technical_levels: Optional[IndexTechnicalLevels]
    error_category: Optional[str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PremarketIndexBatchResult:
    status: PremarketIndexBatchStatus
    expected_session: Optional[date]
    items: tuple[PremarketIndexItemResult, ...]
    success_count: int
    failure_count: int
    error_category: Optional[str] = None
    warnings: tuple[str, ...] = ()


class IndexDailyManager(Protocol):
    def get_index_daily_data(
        self,
        index_code: str,
        expected_session: date,
        days: int = 120,
    ) -> tuple[pd.DataFrame, str]: ...


IndexDailyLoader = Callable[[str, date, int], tuple[pd.DataFrame, str]]


def build_premarket_index_batch(
    window: PremarketWindow,
    *,
    manager: Optional[IndexDailyManager] = None,
    loader: Optional[IndexDailyLoader] = None,
    days: int = 120,
) -> PremarketIndexBatchResult:
    """Build an isolated three-index batch for one resolved premarket window."""
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    if manager is not None and loader is not None:
        raise ValueError("provide either manager or loader, not both")

    invalid_window = _validate_window(window)
    if invalid_window is not None:
        return invalid_window

    expected_session = window.previous_trading_date
    assert expected_session is not None

    if loader is None:
        resolved_manager = manager if manager is not None else DataFetcherManager()

        def resolved_loader(
            index_code: str,
            session: date,
            requested_days: int,
        ) -> tuple[pd.DataFrame, str]:
            return resolved_manager.get_index_daily_data(
                index_code,
                session,
                days=requested_days,
            )
    else:
        resolved_loader = loader

    items = tuple(
        _build_one_index(
            index_code,
            expected_session,
            days,
            resolved_loader,
        )
        for index_code in PREMARKET_INDEX_CODES
    )
    success_count = sum(
        item.status is PremarketIndexItemStatus.SUCCESS for item in items
    )
    failure_count = len(items) - success_count
    if success_count == len(items):
        batch_status = PremarketIndexBatchStatus.COMPLETE
    elif success_count:
        batch_status = PremarketIndexBatchStatus.PARTIAL
    else:
        batch_status = PremarketIndexBatchStatus.FAILED

    batch_warnings = tuple(
        f"{item.index_code}:{warning}"
        for item in items
        for warning in item.warnings
    )
    return PremarketIndexBatchResult(
        status=batch_status,
        expected_session=expected_session,
        items=items,
        success_count=success_count,
        failure_count=failure_count,
        warnings=batch_warnings,
    )


def _validate_window(
    window: PremarketWindow,
) -> Optional[PremarketIndexBatchResult]:
    if not isinstance(window, PremarketWindow):
        return _window_failure("invalid_window")
    if window.status is PremarketWindowStatus.SKIPPED_NON_TRADING_DAY:
        return PremarketIndexBatchResult(
            status=PremarketIndexBatchStatus.SKIPPED,
            expected_session=None,
            items=(),
            success_count=0,
            failure_count=0,
            error_category="non_trading_day",
            warnings=("premarket_window_skipped_non_trading_day",),
        )
    if window.status is PremarketWindowStatus.CALENDAR_UNAVAILABLE:
        return _window_failure("calendar_unavailable")
    if window.status is not PremarketWindowStatus.READY:
        return _window_failure("invalid_window_status")
    if (
        type(window.previous_trading_date) is not date
        or window.window_start is None
        or window.window_end is None
    ):
        return _window_failure("incomplete_window")
    return None


def _window_failure(category: str) -> PremarketIndexBatchResult:
    return PremarketIndexBatchResult(
        status=PremarketIndexBatchStatus.FAILED,
        expected_session=None,
        items=(),
        success_count=0,
        failure_count=0,
        error_category=category,
        warnings=(f"premarket_window_{category}",),
    )


def _build_one_index(
    index_code: str,
    expected_session: date,
    days: int,
    loader: IndexDailyLoader,
) -> PremarketIndexItemResult:
    identity = get_cn_index_identity(index_code)
    provider: Optional[str] = None
    try:
        daily_data, provider = loader(index_code, expected_session, days)
    except Exception:
        return _failed_item(identity.code, identity.name, None, "data_fetch_failed")

    if not isinstance(daily_data, pd.DataFrame) or not isinstance(provider, str) or not provider:
        return _failed_item(
            identity.code, identity.name, provider, "invalid_data_result"
        )
    if (
        daily_data.attrs.get("asset_type") != "index"
        or daily_data.attrs.get("index_code") != identity.code
        or daily_data.attrs.get("index_name") != identity.name
    ):
        return _failed_item(
            identity.code, identity.name, provider, "identity_mismatch"
        )

    try:
        technical_levels = calculate_index_technical_levels(
            identity.code,
            identity.name,
            daily_data,
            expected_session,
        )
    except Exception:
        return _failed_item(
            identity.code, identity.name, provider, "technical_calculation_failed"
        )

    quality = technical_levels.quality_status
    if quality is DataQualityStatus.STALE:
        return _failed_item(
            identity.code, identity.name, provider, "technical_quality_stale"
        )
    if quality is DataQualityStatus.INSUFFICIENT:
        return _failed_item(
            identity.code, identity.name, provider, "technical_quality_insufficient"
        )
    if quality not in {DataQualityStatus.OK, DataQualityStatus.PARTIAL}:
        return _failed_item(
            identity.code, identity.name, provider, "technical_quality_unknown"
        )

    item_warnings = list(technical_levels.warnings)
    if quality is DataQualityStatus.PARTIAL:
        item_warnings.append("technical_quality_partial")
        item_warnings.extend(
            f"missing:{reason}" for reason in technical_levels.missing_reasons
        )
    return PremarketIndexItemResult(
        index_code=identity.code,
        index_name=identity.name,
        status=PremarketIndexItemStatus.SUCCESS,
        provider=provider,
        technical_levels=technical_levels,
        error_category=None,
        warnings=tuple(dict.fromkeys(item_warnings)),
    )


def _failed_item(
    index_code: str,
    index_name: str,
    provider: Optional[str],
    category: str,
) -> PremarketIndexItemResult:
    return PremarketIndexItemResult(
        index_code=index_code,
        index_name=index_name,
        status=PremarketIndexItemStatus.FAILED,
        provider=provider if isinstance(provider, str) and provider else None,
        technical_levels=None,
        error_category=category,
        warnings=(),
    )
