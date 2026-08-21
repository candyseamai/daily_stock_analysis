# -*- coding: utf-8 -*-
"""Pure identity and daily-frame normalization for supported A-share indices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
INDEX_DAILY_STANDARD_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
)


class IndexDailyIdentityError(ValueError):
    """Raised when a canonical index identity or provider is invalid."""


class IndexDailyNormalizationError(ValueError):
    """Raised when provider data cannot produce a valid daily frame."""


@dataclass(frozen=True)
class CNIndexIdentity:
    """Canonical identity and explicit provider symbols for one A-share index."""

    code: str
    name: str
    akshare_symbol: str
    yahoo_symbol: str
    tushare_symbol: str


_CN_INDEX_IDENTITIES = {
    "sh000001": CNIndexIdentity(
        code="sh000001",
        name="上证指数",
        akshare_symbol="sh000001",
        yahoo_symbol="000001.SS",
        tushare_symbol="000001.SH",
    ),
    "sh000300": CNIndexIdentity(
        code="sh000300",
        name="沪深300",
        akshare_symbol="sh000300",
        yahoo_symbol="000300.SS",
        tushare_symbol="000300.SH",
    ),
    "sz399006": CNIndexIdentity(
        code="sz399006",
        name="创业板指",
        akshare_symbol="sz399006",
        yahoo_symbol="399006.SZ",
        tushare_symbol="399006.SZ",
    ),
}
CN_INDEX_IDENTITIES: Mapping[str, CNIndexIdentity] = MappingProxyType(_CN_INDEX_IDENTITIES)

_PROVIDER_SYMBOL_FIELDS = MappingProxyType({
    "AkshareFetcher": "akshare_symbol",
    "YfinanceFetcher": "yahoo_symbol",
    "TushareFetcher": "tushare_symbol",
})


def get_cn_index_identity(index_code: str, index_name: Optional[str] = None) -> CNIndexIdentity:
    """Return a supported canonical identity without normalizing or guessing codes."""
    identity = CN_INDEX_IDENTITIES.get(index_code)
    if identity is None:
        raise IndexDailyIdentityError(f"unsupported canonical A-share index code: {index_code}")
    if index_name is not None and index_name != identity.name:
        raise IndexDailyIdentityError(
            f"index identity mismatch: {index_code} must be named {identity.name}"
        )
    return identity


def get_cn_index_provider_symbol(
    index_code: str,
    provider: str,
    index_name: Optional[str] = None,
) -> str:
    """Return the explicit provider symbol for a validated canonical identity."""
    identity = get_cn_index_identity(index_code, index_name)
    field_name = _PROVIDER_SYMBOL_FIELDS.get(provider)
    if field_name is None:
        raise IndexDailyIdentityError(f"unsupported A-share index daily provider: {provider}")
    return str(getattr(identity, field_name))


def normalize_cn_index_daily_data(
    daily_data: pd.DataFrame,
    *,
    index_code: str,
    index_name: str,
    provider: str,
    expected_session: date,
) -> pd.DataFrame:
    """Normalize already mapped provider bars without accessing a data source.

    ``expected_session`` is an inclusive Shanghai trading-date cutoff and must
    be a pure Python :class:`date`. Provider identity metadata is derived only
    from the canonical mapping above, never from a bare code in returned data.
    """
    identity = get_cn_index_identity(index_code, index_name)
    provider_symbol = get_cn_index_provider_symbol(index_code, provider, index_name)
    if type(expected_session) is not date:
        raise ValueError("expected_session must be a pure Python date, not datetime")
    if not isinstance(daily_data, pd.DataFrame):
        raise TypeError("daily_data must be a pandas DataFrame")

    required = {"date", "open", "high", "low", "close"}
    missing_required = sorted(required.difference(daily_data.columns))
    if missing_required:
        raise IndexDailyNormalizationError(
            f"index daily data missing required columns: {', '.join(missing_required)}"
        )

    frame = daily_data.copy(deep=True)
    warnings: list[str] = []
    normalized_dates = [
        _normalize_date_value(value, original_order)
        for original_order, value in enumerate(frame["date"].tolist())
    ]
    valid_date_mask = [item is not None for item in normalized_dates]
    invalid_date_count = len(valid_date_mask) - sum(valid_date_mask)
    if invalid_date_count:
        warnings.append(f"dropped_invalid_dates:{invalid_date_count}")
    frame = frame.loc[valid_date_mask].copy()
    valid_dates = [item for item in normalized_dates if item is not None]
    frame["_session"] = [item[0] for item in valid_dates]
    frame["_shanghai_sort_time"] = [item[1] for item in valid_dates]
    frame["_original_order"] = [item[2] for item in valid_dates]
    frame = frame.sort_values(
        ["_session", "_shanghai_sort_time", "_original_order"],
        kind="stable",
    )

    duplicate_count = int(frame.duplicated("_session", keep="last").sum())
    if duplicate_count:
        warnings.append(f"duplicate_sessions_kept_last:{duplicate_count}")
        frame = frame.drop_duplicates("_session", keep="last")

    future_count = int((frame["_session"] > expected_session).sum())
    if future_count:
        warnings.append(f"future_bars_removed:{future_count}")
        frame = frame.loc[frame["_session"] <= expected_session].copy()

    ohlc_columns = ["open", "high", "low", "close"]
    for column in ohlc_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_ohlc = frame[ohlc_columns].apply(
        lambda column: column.map(_is_finite_number)
    ).all(axis=1)
    non_finite_ohlc = ~finite_ohlc
    non_positive_ohlc = finite_ohlc & (frame[ohlc_columns] <= 0).any(axis=1)
    invalid_relationship = finite_ohlc & ~non_positive_ohlc & (
        (frame["high"] < frame[["open", "low", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "high", "close"]].min(axis=1))
    )
    if non_finite_ohlc.any():
        warnings.append(f"non_finite_ohlc_rows_removed:{int(non_finite_ohlc.sum())}")
    if non_positive_ohlc.any():
        warnings.append(f"non_positive_ohlc_rows_removed:{int(non_positive_ohlc.sum())}")
    if invalid_relationship.any():
        warnings.append(
            f"invalid_ohlc_relationship_rows_removed:{int(invalid_relationship.sum())}"
        )
    frame = frame.loc[~(non_finite_ohlc | non_positive_ohlc | invalid_relationship)].copy()

    _normalize_optional_numeric_column(frame, "volume", warnings, reject_negative=True)
    _normalize_optional_numeric_column(frame, "amount", warnings, reject_negative=True)
    if "pct_chg" not in frame.columns:
        frame["pct_chg"] = frame["close"].pct_change(fill_method=None) * 100
        warnings.append("pct_chg_calculated_from_close")
    else:
        _normalize_optional_numeric_column(frame, "pct_chg", warnings, reject_negative=False)

    if frame.empty:
        raise IndexDailyNormalizationError(
            f"no valid daily bars for {identity.code} on or before {expected_session.isoformat()}"
        )

    frame["date"] = pd.to_datetime(frame["_session"])
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    for column in INDEX_DAILY_STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    result = frame.loc[:, list(INDEX_DAILY_STANDARD_COLUMNS)].copy()
    result.attrs.update({
        "asset_type": "index",
        "index_code": identity.code,
        "index_name": identity.name,
        "provider": provider,
        "provider_symbol": provider_symbol,
        "warnings": tuple(warnings),
    })
    return result


def _normalize_date_value(
    value: object,
    original_order: int,
) -> Optional[tuple[date, datetime, int]]:
    if type(value) is date:
        local_time = datetime.combine(value, time.min, tzinfo=SHANGHAI_TIMEZONE)
        return value, local_time, original_order
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return None
    if not isinstance(parsed, (datetime, pd.Timestamp)) or pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        parsed = parsed.to_pydatetime()
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            local_time = parsed.replace(tzinfo=SHANGHAI_TIMEZONE)
        else:
            local_time = parsed.astimezone(SHANGHAI_TIMEZONE)
    except (TypeError, ValueError, OverflowError):
        return None
    return local_time.date(), local_time, original_order


def _normalize_optional_numeric_column(
    frame: pd.DataFrame,
    column: str,
    warnings: list[str],
    *,
    reject_negative: bool,
) -> None:
    if column not in frame.columns:
        frame[column] = pd.NA
        warnings.append(f"missing_{column}")
        return
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = ~frame[column].map(_is_finite_number)
    if invalid.any():
        warnings.append(f"invalid_or_non_finite_{column}_values:{int(invalid.sum())}")
        frame.loc[invalid, column] = pd.NA
    if reject_negative:
        negative = ~invalid & (frame[column] < 0)
        if negative.any():
            warnings.append(f"negative_{column}_values:{int(negative.sum())}")
            frame.loc[negative, column] = pd.NA


def _is_finite_number(value: object) -> bool:
    try:
        return not pd.isna(value) and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
