# -*- coding: utf-8 -*-
"""Pure daily-bar technical-level calculations for supported A-share indices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from math import isfinite
from statistics import median
from types import MappingProxyType
from typing import Literal, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


SUPPORTED_INDICES = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sz399006": "创业板指",
}
LOOKBACK_PERIODS = (5, 10, 20, 60)
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


class DataQualityStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    STALE = "stale"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class LevelEvidence:
    kind: str
    value: float
    source_date: Optional[date]
    period: Optional[int]
    description: str


@dataclass(frozen=True)
class SwingPoint:
    kind: Literal["high", "low"]
    date: date
    value: float
    left_bars: int = 2
    right_bars: int = 2


@dataclass(frozen=True)
class DailyGap:
    direction: Literal["up", "down"]
    formed_on: date
    original_lower: float
    original_upper: float
    remaining_lower: float
    remaining_upper: float
    status: Literal["unfilled", "partially_filled"]


@dataclass(frozen=True)
class LevelZone:
    side: Literal["support", "resistance"]
    lower: float
    upper: float
    representative: float
    evidences: tuple[LevelEvidence, ...]


@dataclass(frozen=True)
class IndexTechnicalLevels:
    code: str
    name: str
    expected_session: date
    latest_complete_session: Optional[date]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    ma5: Optional[float]
    ma10: Optional[float]
    ma20: Optional[float]
    ma60: Optional[float]
    boll20_upper: Optional[float]
    boll20_middle: Optional[float]
    boll20_lower: Optional[float]
    atr14_sma: Optional[float]
    range_highs: Mapping[int, Optional[float]]
    range_lows: Mapping[int, Optional[float]]
    recent_swing_high: Optional[SwingPoint]
    recent_swing_low: Optional[SwingPoint]
    unfilled_gaps: tuple[DailyGap, ...]
    support_zones: tuple[LevelZone, ...]
    resistance_zones: tuple[LevelZone, ...]
    neutral_evidences: tuple[LevelEvidence, ...]
    quality_status: DataQualityStatus
    missing_reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def calculate_index_technical_levels(
    code: str,
    name: str,
    daily_data: pd.DataFrame,
    expected_session: date,
) -> IndexTechnicalLevels:
    """Calculate auditable levels from completed bars through ``expected_session``.

    ``expected_session`` must be the pure :class:`date` supplied by the
    premarket-window service. Input rows must already represent completed daily
    bars. This pure calculator can reject later Shanghai dates, but cannot tell
    whether a bar stamped on ``expected_session`` is an unfinished intraday bar.
    """
    _validate_index_identity(code, name)
    if type(expected_session) is not date:
        raise ValueError("expected_session must be a pure Python date, not datetime")

    frame, warnings, quality_issues = _normalize_bars(daily_data, expected_session)
    if frame.empty:
        return _empty_result(
            code, name, expected_session, warnings,
            tuple(quality_issues or ["no_valid_bars_on_or_before_expected_session"]),
        )

    latest_session = frame.iloc[-1]["session"]
    stale = latest_session < expected_session
    if stale:
        quality_issues.append("latest_bar_before_expected_session")

    latest = frame.iloc[-1]
    close = frame["close"]
    mas = {period: _rolling_last(close, period) for period in LOOKBACK_PERIODS}

    boll_middle = _rolling_last(close, 20)
    boll_std = _rolling_last(close, 20, operation="std")
    boll_upper = _finite_or_none(boll_middle + 2 * boll_std) if boll_middle is not None and boll_std is not None else None
    boll_lower = _finite_or_none(boll_middle - 2 * boll_std) if boll_middle is not None and boll_std is not None else None
    atr14 = _atr14_sma(frame)

    range_highs = {period: _window_extreme(frame, period, "high", "max") for period in LOOKBACK_PERIODS}
    range_lows = {period: _window_extreme(frame, period, "low", "min") for period in LOOKBACK_PERIODS}
    swing_high, swing_low = _latest_confirmed_swings(frame)
    gaps = _unfilled_gaps(frame)

    evidences = _build_evidences(
        frame, mas, boll_upper, boll_middle, boll_lower, atr14,
        range_highs, range_lows, swing_high, swing_low, gaps,
    )
    support, resistance, neutral = _classify_evidences(evidences, float(latest["close"]))
    support_zones = _cluster_levels(support, float(latest["close"]), atr14, "support")
    resistance_zones = _cluster_levels(resistance, float(latest["close"]), atr14, "resistance")

    missing = list(quality_issues)
    for field_name, value in (
        ("ma5", mas[5]), ("ma10", mas[10]), ("ma20", mas[20]), ("ma60", mas[60]),
        ("boll20", boll_middle), ("atr14_sma", atr14),
    ):
        if value is None:
            missing.append(f"insufficient_history_for_{field_name}")

    if stale:
        quality = DataQualityStatus.STALE
    elif len(frame) < 5:
        quality = DataQualityStatus.INSUFFICIENT
    elif missing:
        quality = DataQualityStatus.PARTIAL
    else:
        quality = DataQualityStatus.OK

    return IndexTechnicalLevels(
        code=code,
        name=name,
        expected_session=expected_session,
        latest_complete_session=latest_session,
        open=float(latest["open"]),
        high=float(latest["high"]),
        low=float(latest["low"]),
        close=float(latest["close"]),
        ma5=mas[5], ma10=mas[10], ma20=mas[20], ma60=mas[60],
        boll20_upper=boll_upper, boll20_middle=boll_middle, boll20_lower=boll_lower,
        atr14_sma=atr14,
        range_highs=MappingProxyType(range_highs.copy()),
        range_lows=MappingProxyType(range_lows.copy()),
        recent_swing_high=swing_high,
        recent_swing_low=swing_low,
        unfilled_gaps=gaps,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        neutral_evidences=neutral,
        quality_status=quality,
        missing_reasons=tuple(dict.fromkeys(missing)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_index_identity(code: str, name: str) -> None:
    expected_name = SUPPORTED_INDICES.get(code)
    if expected_name is None:
        raise ValueError(f"unsupported canonical index code: {code}")
    if name != expected_name:
        raise ValueError(f"index identity mismatch: {code} must be named {expected_name}")


def _normalize_bars(
    daily_data: pd.DataFrame,
    expected_session: date,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not isinstance(daily_data, pd.DataFrame):
        raise TypeError("daily_data must be a pandas DataFrame")
    required = {"date", "open", "high", "low", "close"}
    missing_columns = sorted(required.difference(daily_data.columns))
    if missing_columns:
        raise ValueError(f"daily_data missing required columns: {', '.join(missing_columns)}")

    frame = daily_data.copy(deep=True)
    warnings: list[str] = []
    quality_issues: list[str] = []
    normalized_dates = [
        _normalize_date_value(value, original_order)
        for original_order, value in enumerate(frame["date"].tolist())
    ]
    valid_date_mask = [item is not None for item in normalized_dates]
    invalid_dates = len(valid_date_mask) - sum(valid_date_mask)
    if invalid_dates:
        warnings.append(f"dropped_invalid_dates:{invalid_dates}")
        quality_issues.append("invalid_dates")
    frame = frame.loc[valid_date_mask].copy()
    valid_normalized_dates = [item for item in normalized_dates if item is not None]
    frame["session"] = [item[0] for item in valid_normalized_dates]
    frame["_shanghai_sort_time"] = [item[1] for item in valid_normalized_dates]
    frame["_original_order"] = [item[2] for item in valid_normalized_dates]

    frame = frame.sort_values(
        ["session", "_shanghai_sort_time", "_original_order"],
        kind="stable",
    )

    duplicate_count = int(frame.duplicated("session", keep="last").sum())
    if duplicate_count:
        warnings.append(f"duplicate_sessions_kept_last:{duplicate_count}")
        frame = frame.drop_duplicates("session", keep="last")

    future_count = int((frame["session"] > expected_session).sum())
    if future_count:
        warnings.append(f"future_bars_removed:{future_count}")
        frame = frame.loc[frame["session"] <= expected_session].copy()

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
        count = int(non_finite_ohlc.sum())
        warnings.append(f"non_finite_ohlc_rows_removed:{count}")
        quality_issues.append("non_finite_ohlc")
    if non_positive_ohlc.any():
        count = int(non_positive_ohlc.sum())
        warnings.append(f"non_positive_ohlc_rows_removed:{count}")
        quality_issues.append("non_positive_ohlc")
    if invalid_relationship.any():
        count = int(invalid_relationship.sum())
        warnings.append(f"invalid_ohlc_relationship_rows_removed:{count}")
        quality_issues.append("invalid_ohlc_relationship")
    invalid_ohlc = non_finite_ohlc | non_positive_ohlc | invalid_relationship
    invalid_count = int(invalid_ohlc.sum())
    if invalid_count:
        warnings.append(f"invalid_ohlc_rows_removed:{invalid_count}")
        quality_issues.append("invalid_ohlc")
        frame = frame.loc[~invalid_ohlc].copy()

    if "volume" not in frame.columns:
        warnings.append("missing_volume_column")
        quality_issues.append("missing_volume")
    else:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        invalid_volume = ~frame["volume"].map(_is_finite_number)
        negative_volume = ~invalid_volume & (frame["volume"] < 0)
        if invalid_volume.any():
            count = int(invalid_volume.sum())
            warnings.append(f"invalid_or_non_finite_volume_values:{count}")
            quality_issues.append("invalid_or_non_finite_volume")
            frame.loc[invalid_volume, "volume"] = pd.NA
        if negative_volume.any():
            count = int(negative_volume.sum())
            warnings.append(f"negative_volume_values:{count}")
            quality_issues.append("negative_volume")
            frame.loc[negative_volume, "volume"] = pd.NA

    frame = frame.sort_values("session", kind="stable").reset_index(drop=True)
    return frame, warnings, quality_issues


def _normalize_date_value(value: object, original_order: int) -> Optional[tuple[date, datetime, int]]:
    """Return Shanghai session date and local sorting time for one scalar value."""
    if type(value) is date:
        local_time = datetime.combine(value, time.min, tzinfo=SHANGHAI_TIMEZONE)
        return value, local_time, original_order
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


def _is_finite_number(value: object) -> bool:
    try:
        return not pd.isna(value) and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_or_none(value: object) -> Optional[float]:
    return float(value) if _is_finite_number(value) else None


def _rolling_last(series: pd.Series, period: int, operation: str = "mean") -> Optional[float]:
    if len(series) < period:
        return None
    rolling = series.rolling(period, min_periods=period)
    value = rolling.mean().iloc[-1] if operation == "mean" else rolling.std(ddof=0).iloc[-1]
    return _finite_or_none(value)


def _atr14_sma(frame: pd.DataFrame) -> Optional[float]:
    if len(frame) < 14:
        return None
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(),
         (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(14, min_periods=14).mean().iloc[-1]
    return _finite_or_none(value)


def _window_extreme(frame: pd.DataFrame, period: int, column: str, operation: str) -> Optional[float]:
    if len(frame) < period:
        return None
    window = frame[column].tail(period)
    return _finite_or_none(window.max() if operation == "max" else window.min())


def _latest_confirmed_swings(frame: pd.DataFrame) -> tuple[Optional[SwingPoint], Optional[SwingPoint]]:
    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []
    for index in range(2, len(frame) - 2):
        row = frame.iloc[index]
        neighbor_highs = pd.concat([frame["high"].iloc[index - 2:index], frame["high"].iloc[index + 1:index + 3]])
        neighbor_lows = pd.concat([frame["low"].iloc[index - 2:index], frame["low"].iloc[index + 1:index + 3]])
        if float(row["high"]) > float(neighbor_highs.max()):
            highs.append(SwingPoint("high", row["session"], float(row["high"])))
        if float(row["low"]) < float(neighbor_lows.min()):
            lows.append(SwingPoint("low", row["session"], float(row["low"])))
    return (highs[-1] if highs else None, lows[-1] if lows else None)


def _unfilled_gaps(frame: pd.DataFrame) -> tuple[DailyGap, ...]:
    gaps: list[DailyGap] = []
    for index in range(1, len(frame)):
        previous = frame.iloc[index - 1]
        current = frame.iloc[index]
        subsequent = frame.iloc[index + 1:]
        if float(current["low"]) > float(previous["high"]):
            lower, upper = float(previous["high"]), float(current["low"])
            later_low = float(subsequent["low"].min()) if not subsequent.empty else upper
            if later_low <= lower:
                continue
            remaining_upper = min(upper, later_low)
            status = "partially_filled" if remaining_upper < upper else "unfilled"
            gaps.append(DailyGap("up", current["session"], lower, upper, lower, remaining_upper, status))
        elif float(current["high"]) < float(previous["low"]):
            lower, upper = float(current["high"]), float(previous["low"])
            later_high = float(subsequent["high"].max()) if not subsequent.empty else lower
            if later_high >= upper:
                continue
            remaining_lower = max(lower, later_high)
            status = "partially_filled" if remaining_lower > lower else "unfilled"
            gaps.append(DailyGap("down", current["session"], lower, upper, remaining_lower, upper, status))
    return tuple(gaps)


def _evidence(kind: str, value: float, source_date: Optional[date], period: Optional[int], description: str) -> LevelEvidence:
    if not _is_finite_number(value):
        raise ValueError(f"non-finite technical-level evidence: {kind}")
    return LevelEvidence(kind, float(value), source_date, period, description)


def _build_evidences(
    frame: pd.DataFrame,
    mas: dict[int, Optional[float]],
    boll_upper: Optional[float], boll_middle: Optional[float], boll_lower: Optional[float],
    atr14: Optional[float], range_highs: dict[int, Optional[float]], range_lows: dict[int, Optional[float]],
    swing_high: Optional[SwingPoint], swing_low: Optional[SwingPoint], gaps: Sequence[DailyGap],
) -> list[LevelEvidence]:
    latest = frame.iloc[-1]
    session = latest["session"]
    values = [
        _evidence("previous_high", latest["high"], session, 1, "上一完整交易日最高价"),
        _evidence("previous_low", latest["low"], session, 1, "上一完整交易日最低价"),
    ]
    for period, value in mas.items():
        if value is not None:
            values.append(_evidence(f"ma{period}", value, session, period, f"{period}日收盘简单移动平均"))
    for kind, value, description in (
        ("boll20_upper", boll_upper, "BOLL20上轨（总体标准差×2）"),
        ("boll20_middle", boll_middle, "BOLL20中轨（20日收盘均值）"),
        ("boll20_lower", boll_lower, "BOLL20下轨（总体标准差×2）"),
    ):
        if value is not None:
            values.append(_evidence(kind, value, session, 20, description))
    for period in LOOKBACK_PERIODS:
        if range_highs[period] is not None:
            values.append(_evidence(f"high_{period}d", range_highs[period], session, period, f"近{period}日最高价"))
        if range_lows[period] is not None:
            values.append(_evidence(f"low_{period}d", range_lows[period], session, period, f"近{period}日最低价"))
    if swing_high:
        values.append(_evidence("confirmed_swing_high", swing_high.value, swing_high.date, 5, "2左2右确认局部高点"))
    if swing_low:
        values.append(_evidence("confirmed_swing_low", swing_low.value, swing_low.date, 5, "2左2右确认局部低点"))
    for gap in gaps:
        values.append(_evidence(f"{gap.direction}_gap_remaining_lower", gap.remaining_lower, gap.formed_on, None, "未完全回补日线缺口剩余下边界"))
        values.append(_evidence(f"{gap.direction}_gap_remaining_upper", gap.remaining_upper, gap.formed_on, None, "未完全回补日线缺口剩余上边界"))
    if atr14 is not None:
        close = float(latest["close"])
        projected_down = _finite_or_none(close - atr14)
        projected_up = _finite_or_none(close + atr14)
        if projected_down is not None:
            values.append(_evidence("atr14_sma_projection_down", projected_down, session, 14, "收盘减ATR14_SMA波动率投影"))
        if projected_up is not None:
            values.append(_evidence("atr14_sma_projection_up", projected_up, session, 14, "收盘加ATR14_SMA波动率投影"))
    return values


def _classify_evidences(
    evidences: Sequence[LevelEvidence], close: float,
) -> tuple[list[LevelEvidence], list[LevelEvidence], tuple[LevelEvidence, ...]]:
    neutral_ids: set[int] = set()
    gap_groups: dict[tuple[str, Optional[date]], list[LevelEvidence]] = {}
    for item in evidences:
        if "_gap_remaining_" in item.kind:
            gap_key = (item.kind.split("_gap_remaining_", 1)[0], item.source_date)
            gap_groups.setdefault(gap_key, []).append(item)
    for group in gap_groups.values():
        if len(group) == 2 and min(item.value for item in group) <= close <= max(item.value for item in group):
            neutral_ids.update(id(item) for item in group)

    support = [item for item in evidences if id(item) not in neutral_ids and item.value < close]
    resistance = [item for item in evidences if id(item) not in neutral_ids and item.value > close]
    neutral = tuple(item for item in evidences if id(item) in neutral_ids or item.value == close)
    return support, resistance, neutral


def _cluster_levels(
    evidences: Sequence[LevelEvidence], close: float, atr14: Optional[float], side: Literal["support", "resistance"],
) -> tuple[LevelZone, ...]:
    if not evidences:
        return ()
    atr = atr14 or 0.0
    merge_distance = max(close * 0.003, atr * 0.25)
    max_zone_width = max(close * 0.006, atr * 0.5)
    ordered = sorted(evidences, key=lambda item: (item.value, item.kind, item.source_date or date.min))
    clusters: list[list[LevelEvidence]] = []
    for item in ordered:
        if not clusters:
            clusters.append([item])
            continue
        current = clusters[-1]
        new_lower = min(current[0].value, item.value)
        new_upper = max(current[-1].value, item.value)
        if item.value - current[-1].value <= merge_distance and new_upper - new_lower <= max_zone_width:
            current.append(item)
        else:
            clusters.append([item])
    zones = [
        LevelZone(side, min(item.value for item in cluster), max(item.value for item in cluster),
                  float(median(sorted({item.value for item in cluster}))), tuple(cluster))
        for cluster in clusters
    ]
    zones.sort(key=lambda zone: min(abs(close - zone.lower), abs(close - zone.upper)))
    return tuple(zones[:3])


def _empty_result(
    code: str, name: str, expected_session: date, warnings: Sequence[str], missing: Sequence[str],
) -> IndexTechnicalLevels:
    empty_ranges = MappingProxyType({period: None for period in LOOKBACK_PERIODS})
    return IndexTechnicalLevels(
        code, name, expected_session, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        empty_ranges, MappingProxyType(dict(empty_ranges)), None, None, (), (), (), (),
        DataQualityStatus.INSUFFICIENT, tuple(dict.fromkeys(missing)), tuple(dict.fromkeys(warnings)),
    )
