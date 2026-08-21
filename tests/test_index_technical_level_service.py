# -*- coding: utf-8 -*-
"""Deterministic tests for the pure index technical-level calculator."""

from datetime import date, datetime, timedelta, timezone
from math import isclose, isfinite
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.services.index_technical_level_service import (
    DataQualityStatus,
    LevelEvidence,
    _cluster_levels,
    calculate_index_technical_levels,
)


EXPECTED = date(2026, 3, 31)


def _bars(count: int = 60, *, start: date = date(2026, 1, 1)) -> pd.DataFrame:
    dates = [start + timedelta(days=index) for index in range(count)]
    close = [100.0 + index for index in range(count)]
    return pd.DataFrame({
        "date": dates,
        "open": [value - 0.5 for value in close],
        "high": [value + 1.0 for value in close],
        "low": [value - 1.0 for value in close],
        "close": close,
        "volume": [1000.0] * count,
    })


def _calculate(frame: pd.DataFrame, expected: date | None = None):
    if expected is None:
        expected = pd.to_datetime(frame["date"]).max().date()
    return calculate_index_technical_levels("sh000001", "上证指数", frame, expected)


def _output_evidences(result):
    return tuple(
        item
        for zone in (*result.support_zones, *result.resistance_zones)
        for item in zone.evidences
    ) + result.neutral_evidences


def _output_evidence(result, kind: str) -> LevelEvidence:
    return next(item for item in _output_evidences(result) if item.kind == kind)


def test_exact_ma_boll_and_atr_values():
    frame = _bars(60)
    result = _calculate(frame)

    assert result.ma5 == pytest.approx(sum(range(155, 160)) / 5)
    assert result.ma10 == pytest.approx(sum(range(150, 160)) / 10)
    assert result.ma20 == pytest.approx(sum(range(140, 160)) / 20)
    assert result.ma60 == pytest.approx(sum(range(100, 160)) / 60)
    expected_std = pd.Series(range(140, 160), dtype=float).std(ddof=0)
    assert result.boll20_middle == pytest.approx(149.5)
    assert result.boll20_upper == pytest.approx(149.5 + 2 * expected_std)
    assert result.boll20_lower == pytest.approx(149.5 - 2 * expected_std)
    assert result.atr14_sma == pytest.approx(2.0)
    assert result.range_highs == {5: 160.0, 10: 160.0, 20: 160.0, 60: 160.0}
    assert result.range_lows == {5: 154.0, 10: 149.0, 20: 139.0, 60: 99.0}


def test_evidence_distinguishes_calculation_date_from_historical_source_date():
    frame = _bars(60)
    frame["open"] = 100.0
    frame["high"] = 101.0
    frame["low"] = 99.0
    frame["close"] = 100.0
    frame.loc[[56, 58], "high"] = 105.0
    frame.loc[[56, 58], "low"] = 95.0
    result = _calculate(frame)

    calculated_on = frame.iloc[-1]["date"]
    most_recent_extreme_date = frame.iloc[58]["date"]
    for period in (5, 10, 20, 60):
        assert _output_evidence(result, f"high_{period}d").source_date == most_recent_extreme_date
        assert _output_evidence(result, f"low_{period}d").source_date == most_recent_extreme_date

    for kind in (
        "ma5", "ma10", "ma20", "ma60",
        "boll20_upper", "boll20_middle", "boll20_lower",
        "atr14_sma_projection_down", "atr14_sma_projection_up",
    ):
        assert _output_evidence(result, kind).source_date is None

    assert _output_evidence(result, "previous_high").source_date == calculated_on
    assert _output_evidence(result, "previous_low").source_date == calculated_on
    assert all(item.calculated_on == calculated_on for item in _output_evidences(result))


def test_59_bars_do_not_fabricate_ma60():
    result = _calculate(_bars(59))
    assert result.ma60 is None
    assert "insufficient_history_for_ma60" in result.missing_reasons
    assert result.quality_status is DataQualityStatus.PARTIAL


def test_reverse_input_and_duplicate_session_keep_last_with_warning():
    frame = _bars(20)
    duplicate = frame.iloc[[-1]].copy()
    duplicate["close"] = 999.0
    duplicate["open"] = 998.0
    duplicate["high"] = 1000.0
    duplicate["low"] = 997.0
    result = _calculate(pd.concat([frame.iloc[::-1], duplicate], ignore_index=True))
    assert result.close == 999.0
    assert "duplicate_sessions_kept_last:1" in result.warnings


def test_input_dataframe_is_not_modified():
    frame = _bars(20)
    frame["close"] = frame["close"].astype(str)
    original = frame.copy(deep=True)
    _calculate(frame)
    assert_frame_equal(frame, original)


def test_mixed_date_string_timestamp_and_datetime_values_are_normalized():
    frame = _bars(5)
    frame["date"] = [
        date(2026, 3, 20),
        "2026-03-21",
        pd.Timestamp("2026-03-22 09:00"),
        datetime(2026, 3, 23, 15, 0),
        pd.Timestamp("2026-03-23 16:30", tz="UTC"),
    ]
    result = _calculate(frame, date(2026, 3, 24))
    assert result.latest_complete_session == date(2026, 3, 24)
    assert result.close == 104.0


def test_unparseable_date_is_removed_and_reported():
    frame = _bars(5)
    frame.loc[2, "date"] = "not-a-date"
    result = _calculate(frame, date(2026, 1, 5))
    assert "dropped_invalid_dates:1" in result.warnings
    assert "invalid_dates" in result.missing_reasons


def test_utc_timestamp_crossing_shanghai_midnight_is_filtered_by_shanghai_date():
    frame = _bars(2)
    frame["date"] = [
        date(2026, 3, 23),
        datetime(2026, 3, 23, 16, 30, tzinfo=timezone.utc),
    ]
    result = _calculate(frame, date(2026, 3, 23))
    assert result.latest_complete_session == date(2026, 3, 23)
    assert result.close == 100.0
    assert "future_bars_removed:1" in result.warnings


def test_naive_datetime_is_shanghai_wall_time_and_aware_values_convert_to_shanghai():
    frame = _bars(3)
    frame["date"] = [
        datetime(2026, 3, 24, 9, 0),
        datetime(2026, 3, 23, 16, 30, tzinfo=timezone.utc),
        pd.Timestamp("2026-03-24 13:00", tz="America/New_York"),
    ]
    result = _calculate(frame, date(2026, 3, 24))
    assert result.latest_complete_session == date(2026, 3, 24)
    assert result.close == 100.0
    assert "future_bars_removed:1" in result.warnings


def test_same_session_keeps_latest_shanghai_time_then_stable_input_order():
    frame = _bars(4)
    frame["date"] = [
        pd.Timestamp("2026-03-24 15:00", tz="Asia/Shanghai"),
        datetime(2026, 3, 24, 9, 0),
        date(2026, 3, 24),
        pd.Timestamp("2026-03-24 15:00", tz="Asia/Shanghai"),
    ]
    result = _calculate(frame, date(2026, 3, 24))
    assert result.close == 103.0
    assert "duplicate_sessions_kept_last:3" in result.warnings


def test_expected_session_rejects_datetime_even_though_it_is_a_date_subclass():
    with pytest.raises(ValueError, match="pure Python date"):
        calculate_index_technical_levels(
            "sh000001", "上证指数", _bars(5), datetime(2026, 3, 31),
        )


def test_future_bar_is_removed():
    frame = _bars(20)
    expected = pd.to_datetime(frame.iloc[-2]["date"]).date()
    result = _calculate(frame, expected)
    assert result.latest_complete_session == expected
    assert result.close == frame.iloc[-2]["close"]
    assert "future_bars_removed:1" in result.warnings


def test_stale_latest_bar_is_explicit():
    frame = _bars(20)
    expected = pd.to_datetime(frame.iloc[-1]["date"]).date() + timedelta(days=1)
    result = _calculate(frame, expected)
    assert result.quality_status is DataQualityStatus.STALE
    assert "latest_bar_before_expected_session" in result.missing_reasons


def test_invalid_ohlc_is_removed_and_reported():
    frame = _bars(20)
    frame.loc[5, "high"] = frame.loc[5, "low"] - 1
    result = _calculate(frame)
    assert "invalid_ohlc_rows_removed:1" in result.warnings
    assert "invalid_ohlc" in result.missing_reasons


def test_numeric_ohlc_strings_are_accepted():
    frame = _bars(5)
    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].astype(str)
    result = _calculate(frame)
    assert result.close == 104.0
    assert "invalid_ohlc" not in result.missing_reasons


@pytest.mark.parametrize("bad_value", ["bad", float("nan"), float("inf"), float("-inf")])
def test_non_numeric_or_non_finite_ohlc_row_is_removed(bad_value):
    frame = _bars(6)
    frame["high"] = frame["high"].astype(object)
    frame.loc[2, "high"] = bad_value
    result = _calculate(frame)
    assert "non_finite_ohlc" in result.missing_reasons
    assert "non_finite_ohlc_rows_removed:1" in result.warnings


@pytest.mark.parametrize(
    "column,bad_value,reason",
    [
        ("high", 0, "non_positive_ohlc"),
        ("low", -1, "non_positive_ohlc"),
        ("high", 50, "invalid_ohlc_relationship"),
        ("low", 500, "invalid_ohlc_relationship"),
    ],
)
def test_non_positive_and_invalid_ohlc_relationships_are_removed(column, bad_value, reason):
    frame = _bars(6)
    frame.loc[2, column] = bad_value
    result = _calculate(frame)
    assert reason in result.missing_reasons


def test_missing_and_negative_volume_are_quality_issues():
    no_volume = _bars(20).drop(columns="volume")
    assert "missing_volume" in _calculate(no_volume).missing_reasons
    negative = _bars(20)
    negative.loc[0, "volume"] = -1
    assert "negative_volume" in _calculate(negative).missing_reasons


@pytest.mark.parametrize("bad_value", ["bad", float("nan"), float("inf"), float("-inf")])
def test_invalid_or_non_finite_volume_is_explicitly_reported(bad_value):
    frame = _bars(20)
    frame["volume"] = frame["volume"].astype(object)
    frame.loc[0, "volume"] = bad_value
    result = _calculate(frame)
    assert "invalid_or_non_finite_volume" in result.missing_reasons
    assert "invalid_or_non_finite_volume_values:1" in result.warnings


def test_all_emitted_metrics_levels_and_zones_are_finite():
    result = _calculate(_bars(60))
    scalar_values = [
        result.open, result.high, result.low, result.close,
        result.ma5, result.ma10, result.ma20, result.ma60,
        result.boll20_upper, result.boll20_middle, result.boll20_lower,
        result.atr14_sma,
        *result.range_highs.values(), *result.range_lows.values(),
    ]
    assert all(value is None or isfinite(value) for value in scalar_values)
    for zone in (*result.support_zones, *result.resistance_zones):
        assert all(isfinite(value) for value in (zone.lower, zone.upper, zone.representative))
        assert all(isfinite(item.value) for item in zone.evidences)
    assert all(isfinite(item.value) for item in result.neutral_evidences)
    assert all(
        isfinite(value)
        for gap in result.unfilled_gaps
        for value in (
            gap.original_lower, gap.original_upper,
            gap.remaining_lower, gap.remaining_upper,
        )
    )
    assert result.recent_swing_high is None or isfinite(result.recent_swing_high.value)
    assert result.recent_swing_low is None or isfinite(result.recent_swing_low.value)


def test_range_mappings_are_read_only():
    result = _calculate(_bars(60))
    with pytest.raises(TypeError):
        result.range_highs[5] = 1.0
    with pytest.raises(TypeError):
        result.range_lows[5] = 1.0


def test_returns_latest_confirmed_two_left_two_right_swing_low():
    frame = _bars(10)
    frame["high"] = [11, 12, 20, 13, 12, 14, 15, 16, 30, 40]
    frame["low"] = [9, 8, 7, 8, 9, 8, 2, 8, 3, 4]
    frame["open"] = 10.0
    frame["close"] = 10.0
    result = _calculate(frame)
    assert result.recent_swing_high is not None
    assert result.recent_swing_high.date == frame.iloc[2]["date"]
    assert result.recent_swing_low is not None
    assert result.recent_swing_low.date == frame.iloc[6]["date"]
    assert result.recent_swing_high.value != 40
    assert result.recent_swing_low.value == 2
    assert _output_evidence(result, "confirmed_swing_high").source_date == frame.iloc[2]["date"]
    assert _output_evidence(result, "confirmed_swing_low").source_date == frame.iloc[6]["date"]
    assert _output_evidence(result, "confirmed_swing_low").calculated_on == frame.iloc[-1]["date"]


def test_lower_right_confirmation_bar_invalidates_earlier_swing_low_candidate():
    frame = _bars(10)
    frame["high"] = [11, 12, 20, 13, 12, 14, 15, 16, 30, 40]
    frame["low"] = [9, 8, 7, 8, 9, 8, 2, 8, 1, 0.5]
    frame["open"] = 10.0
    frame["close"] = 10.0
    result = _calculate(frame)

    assert result.recent_swing_low is not None
    assert result.recent_swing_low.date == frame.iloc[2]["date"]
    assert result.recent_swing_low.value == 7
    assert result.recent_swing_low.date != frame.iloc[6]["date"]
    assert result.recent_swing_low.value not in {1, 0.5}


def test_equal_highs_and_equal_lows_do_not_create_confirmed_swings():
    frame = _bars(7)
    frame["open"] = 10.0
    frame["close"] = 10.0
    frame["high"] = [11, 12, 20, 20, 20, 12, 11]
    frame["low"] = [9, 8, 2, 2, 2, 8, 9]
    result = _calculate(frame)
    assert result.recent_swing_high is None
    assert result.recent_swing_low is None


def test_atr_requires_14_rows_and_uses_gap_true_range():
    assert _calculate(_bars(13)).atr14_sma is None
    flat = _bars(14)
    flat[["open", "close"]] = 100.0
    flat["high"] = 101.0
    flat["low"] = 99.0
    assert _calculate(flat).atr14_sma == pytest.approx(2.0)
    flat.loc[13, ["open", "high", "low", "close"]] = [110.0, 111.0, 109.0, 110.0]
    assert _calculate(flat).atr14_sma == pytest.approx((13 * 2.0 + 11.0) / 14.0)


def _gap_frame(direction: str, later_value: float | None) -> pd.DataFrame:
    prices = {
        "up": {
            "open": [9.5, 12.5, 13.0, 13.0],
            "high": [10.0, 14.0, 14.0, 14.0],
            "low": [9.0, 12.0, 12.5, 12.5],
            "close": [9.5, 13.0, 13.0, 13.0],
        },
        "down": {
            "open": [20.5, 17.5, 17.0, 17.0],
            "high": [21.0, 18.0, 18.0, 18.0],
            "low": [20.0, 16.0, 16.0, 16.0],
            "close": [20.5, 17.0, 17.0, 17.0],
        },
    }[direction]
    frame = pd.DataFrame({
        "date": [date(2026, 3, 1) + timedelta(days=index) for index in range(4)],
        **prices,
        "volume": [1, 1, 1, 1],
    })
    if later_value is not None:
        column = "low" if direction == "up" else "high"
        frame.loc[2:, column] = later_value
        if direction == "up":
            frame.loc[2:, "open"] = frame.loc[2:, ["open", "low"]].max(axis=1)
            frame.loc[2:, "close"] = frame.loc[2:, ["close", "low"]].max(axis=1)
            frame.loc[2:, "high"] = frame.loc[2:, ["high", "open", "close"]].max(axis=1)
        else:
            frame.loc[2:, "open"] = frame.loc[2:, ["open", "high"]].min(axis=1)
            frame.loc[2:, "close"] = frame.loc[2:, ["close", "high"]].min(axis=1)
            frame.loc[2:, "low"] = frame.loc[2:, ["low", "open", "close"]].min(axis=1)
    return frame


@pytest.mark.parametrize("direction", ["up", "down"])
def test_unfilled_gap_is_retained(direction):
    result = _calculate(_gap_frame(direction, None))
    gap = next(gap for gap in result.unfilled_gaps if gap.direction == direction)
    assert gap.status == "unfilled"
    assert gap.remaining_lower == gap.original_lower
    assert gap.remaining_upper == gap.original_upper


@pytest.mark.parametrize("direction", ["up", "down"])
def test_gap_formation_bar_does_not_fill_its_own_gap(direction):
    frame = _gap_frame(direction, None).iloc[:2].copy()
    result = _calculate(frame)
    gap = next(gap for gap in result.unfilled_gaps if gap.direction == direction)
    assert gap.status == "unfilled"


@pytest.mark.parametrize("direction,later_value", [("up", 11.0), ("down", 19.0)])
def test_partially_filled_gap_keeps_only_remaining_interval(direction, later_value):
    result = _calculate(_gap_frame(direction, later_value))
    gap = next(gap for gap in result.unfilled_gaps if gap.direction == direction)
    assert gap.status == "partially_filled"
    assert gap.remaining_upper - gap.remaining_lower < gap.original_upper - gap.original_lower
    if direction == "up":
        assert (gap.remaining_lower, gap.remaining_upper) == (10.0, 11.0)
    else:
        assert (gap.remaining_lower, gap.remaining_upper) == (19.0, 20.0)


@pytest.mark.parametrize("direction,later_value", [("up", 10.0), ("down", 20.0)])
def test_fully_filled_gap_is_omitted(direction, later_value):
    result = _calculate(_gap_frame(direction, later_value))
    assert not any(gap.direction == direction for gap in result.unfilled_gaps)


def test_multiple_unfilled_gaps_coexist_independently():
    frame = pd.DataFrame({
        "date": [date(2026, 3, 1) + timedelta(days=index) for index in range(4)],
        "open": [9.5, 12.5, 15.5, 16.0],
        "high": [10.0, 13.0, 16.0, 17.0],
        "low": [9.0, 12.0, 15.0, 15.5],
        "close": [9.5, 12.5, 15.5, 16.0],
        "volume": [1, 1, 1, 1],
    })
    result = _calculate(frame)
    up_gaps = [gap for gap in result.unfilled_gaps if gap.direction == "up"]
    assert len(up_gaps) == 2
    assert [(gap.original_lower, gap.original_upper) for gap in up_gaps] == [(10.0, 12.0), (13.0, 15.0)]


def test_support_resistance_never_reverse_and_equal_is_neutral():
    frame = _bars(60)
    frame["open"] = 100.0
    frame["high"] = 101.0
    frame["low"] = 99.0
    frame["close"] = 100.0
    result = _calculate(frame)

    assert all(zone.upper < result.close for zone in result.support_zones)
    assert all(zone.lower > result.close for zone in result.resistance_zones)
    neutral_kinds = {item.kind for item in result.neutral_evidences}
    assert {"ma5", "ma10", "ma20", "ma60", "boll20_middle"}.issubset(neutral_kinds)
    assert all(item.value == result.close for item in result.neutral_evidences)
    directional_evidences = {
        item
        for zone in (*result.support_zones, *result.resistance_zones)
        for item in zone.evidences
    }
    assert all(item not in directional_evidences for item in result.neutral_evidences)


def test_gap_interval_crossing_close_is_neutral_as_a_whole():
    frame = _gap_frame("up", None)
    frame.loc[3, ["open", "high", "low", "close"]] = [10.3, 10.5, 10.2, 10.2]
    result = _calculate(frame)
    neutral_gap_kinds = {item.kind for item in result.neutral_evidences if "gap" in item.kind}
    assert neutral_gap_kinds == {"up_gap_remaining_lower", "up_gap_remaining_upper"}
    for kind in neutral_gap_kinds:
        evidence = _output_evidence(result, kind)
        assert evidence.source_date == frame.iloc[1]["date"]
        assert evidence.calculated_on == frame.iloc[-1]["date"]
    directional_kinds = {
        item.kind
        for zone in (*result.support_zones, *result.resistance_zones)
        for item in zone.evidences
    }
    assert neutral_gap_kinds.isdisjoint(directional_kinds)


def _level(value: float, number: int) -> LevelEvidence:
    return LevelEvidence(
        kind=f"test_{number}",
        value=value,
        calculated_on=EXPECTED,
        source_date=None,
        period=None,
        description="fixed test level",
    )


def test_cluster_threshold_and_maximum_width_are_deterministic():
    levels = [_level(value, index) for index, value in enumerate([98.0, 98.2, 98.4, 99.0])]
    zones = _cluster_levels(levels, 100.0, 1.0, "support")
    assert [(zone.lower, zone.upper) for zone in zones] == [(99.0, 99.0), (98.0, 98.4)]
    assert zones[1].representative == pytest.approx(98.2)


def test_duplicate_evidence_prices_count_as_one_median_node():
    levels = [
        _level(98.0, 0), _level(98.0, 1), _level(98.0, 2),
        _level(100.0, 3), _level(104.0, 4),
    ]
    zones = _cluster_levels(levels, 110.0, 20.0, "support")
    assert len(zones) == 1
    assert zones[0].representative == 100.0
    assert len(zones[0].evidences) == 5


def test_chain_merge_is_stopped_by_maximum_zone_width():
    levels = [_level(value, index) for index, value in enumerate([98.0, 98.29, 98.58, 98.87])]
    zones = _cluster_levels(levels, 100.0, 0.0, "support")
    assert len(zones) == 2
    assert all(zone.upper - zone.lower <= 0.6 for zone in zones)
    assert {(zone.lower, zone.upper) for zone in zones} == {(98.0, 98.58), (98.87, 98.87)}


def test_cluster_outputs_nearest_three_and_preserves_all_evidence():
    levels = [_level(value, index) for index, value in enumerate([90, 92, 94, 96, 98])]
    zones = _cluster_levels(levels, 100.0, 0.0, "support")
    assert len(zones) == 3
    assert [zone.representative for zone in zones] == [98, 96, 94]
    assert all(zone.evidences for zone in zones)
    assert all(isclose(zone.representative, zone.evidences[0].value) for zone in zones)


def test_resistance_outputs_nearest_three_in_distance_order():
    levels = [_level(value, index) for index, value in enumerate([105, 101, 104, 102, 103])]
    zones = _cluster_levels(levels, 100.0, 0.0, "resistance")
    assert len(zones) == 3
    assert [zone.representative for zone in zones] == [101, 102, 103]


def test_all_final_zones_retain_auditable_evidence():
    result = _calculate(_bars(60))
    for zone in (*result.support_zones, *result.resistance_zones):
        assert zone.evidences
        assert all(item.kind and item.description for item in zone.evidences)
        assert zone.lower == min(item.value for item in zone.evidences)
        assert zone.upper == max(item.value for item in zone.evidences)


def test_canonical_index_identity_is_enforced():
    with pytest.raises(ValueError, match="unsupported canonical index code"):
        calculate_index_technical_levels("000001", "上证指数", _bars(20), EXPECTED)
    with pytest.raises(ValueError, match="identity mismatch"):
        calculate_index_technical_levels("sh000300", "上证指数", _bars(20), EXPECTED)


@pytest.mark.parametrize(
    "count,expected_status",
    [
        (19, DataQualityStatus.PARTIAL),
        (13, DataQualityStatus.PARTIAL),
        (4, DataQualityStatus.INSUFFICIENT),
    ],
)
def test_short_history_quality_statuses(count, expected_status):
    assert _calculate(_bars(count)).quality_status is expected_status


def test_empty_frame_with_required_columns_is_insufficient():
    frame = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    result = calculate_index_technical_levels("sh000001", "上证指数", frame, EXPECTED)
    assert result.quality_status is DataQualityStatus.INSUFFICIENT
    assert result.latest_complete_session is None
    assert "no_valid_bars_on_or_before_expected_session" in result.missing_reasons
