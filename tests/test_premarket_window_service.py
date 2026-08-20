# -*- coding: utf-8 -*-
"""Unit tests for the A-share premarket window resolver."""

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.core import trading_calendar
from src.services.premarket_window_service import (
    PremarketWindowStatus,
    resolve_a_share_premarket_window,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


class _FakeCalendar:
    def __init__(self, sessions: list[date]):
        self._sessions = sorted(sessions)

    def is_session(self, check_date: date) -> bool:
        return check_date in self._sessions

    def date_to_session(self, check_date: date, direction: str = "previous") -> pd.Timestamp:
        if direction != "previous":
            raise ValueError(f"unsupported direction: {direction}")
        candidates = [session for session in self._sessions if session <= check_date]
        if not candidates:
            raise ValueError(f"no session for {check_date}")
        return pd.Timestamp(candidates[-1])

    def previous_session(self, session: pd.Timestamp) -> pd.Timestamp:
        index = self._sessions.index(session.date())
        if index == 0:
            raise ValueError("no previous session")
        return pd.Timestamp(self._sessions[index - 1])


def _resolve(current_time: datetime, sessions: list[date]):
    calendar = _FakeCalendar(sessions)
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", True), patch.object(
        trading_calendar,
        "xcals",
        SimpleNamespace(get_calendar=lambda _exchange: calendar),
        create=True,
    ):
        return resolve_a_share_premarket_window(current_time)


def test_regular_tuesday_at_0807_uses_monday_close():
    result = _resolve(
        datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI),
        [date(2026, 3, 23), date(2026, 3, 24)],
    )

    assert result.status is PremarketWindowStatus.READY
    assert result.previous_trading_date == date(2026, 3, 23)
    assert result.window_start == datetime(2026, 3, 23, 15, 0, tzinfo=SHANGHAI)
    assert result.window_end == datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI)


def test_monday_at_0807_uses_previous_friday_close():
    result = _resolve(
        datetime(2026, 3, 30, 8, 7, tzinfo=SHANGHAI),
        [date(2026, 3, 27), date(2026, 3, 30)],
    )

    assert result.window_start == datetime(2026, 3, 27, 15, 0, tzinfo=SHANGHAI)


def test_first_session_after_public_holiday_uses_last_actual_session():
    result = _resolve(
        datetime(2026, 10, 9, 8, 7, tzinfo=SHANGHAI),
        [date(2026, 9, 30), date(2026, 10, 9)],
    )

    assert result.status is PremarketWindowStatus.READY
    assert result.window_start == datetime(2026, 9, 30, 15, 0, tzinfo=SHANGHAI)


def test_regular_weekend_returns_explicit_skip_status():
    result = _resolve(
        datetime(2026, 3, 28, 8, 7, tzinfo=SHANGHAI),
        [date(2026, 3, 27), date(2026, 3, 30)],
    )

    assert result.status is PremarketWindowStatus.SKIPPED_NON_TRADING_DAY
    assert result.should_generate is False
    assert result.window_start is None
    assert result.window_end is None


def test_manual_run_on_exchange_holiday_returns_explicit_skip_status():
    result = _resolve(
        datetime(2026, 10, 2, 8, 7, tzinfo=SHANGHAI),
        [date(2026, 9, 30), date(2026, 10, 9)],
    )

    assert result.status is PremarketWindowStatus.SKIPPED_NON_TRADING_DAY
    assert "not an XSHG trading session" in (result.reason or "")


def test_delayed_github_actions_run_uses_actual_0815_end_time():
    result = _resolve(
        datetime(2026, 3, 24, 8, 15, tzinfo=SHANGHAI),
        [date(2026, 3, 23), date(2026, 3, 24)],
    )

    assert result.window_end == datetime(2026, 3, 24, 8, 15, tzinfo=SHANGHAI)


def test_cross_year_window_uses_previous_year_session():
    result = _resolve(
        datetime(2026, 1, 5, 8, 7, tzinfo=SHANGHAI),
        [date(2025, 12, 31), date(2026, 1, 5)],
    )

    assert result.window_start == datetime(2025, 12, 31, 15, 0, tzinfo=SHANGHAI)
    assert result.window_end == datetime(2026, 1, 5, 8, 7, tzinfo=SHANGHAI)


def test_naive_datetime_is_interpreted_as_shanghai_wall_time():
    result = _resolve(
        datetime(2026, 3, 24, 8, 7),
        [date(2026, 3, 23), date(2026, 3, 24)],
    )

    assert result.run_at == datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI)
    assert result.run_at.tzinfo == SHANGHAI


def test_window_boundaries_are_timezone_aware():
    result = _resolve(
        datetime(2026, 3, 24, 0, 7, tzinfo=ZoneInfo("UTC")),
        [date(2026, 3, 23), date(2026, 3, 24)],
    )

    assert result.window_start is not None
    assert result.window_end is not None
    assert result.window_start.utcoffset() is not None
    assert result.window_end.utcoffset() is not None
    assert result.window_end == datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI)


def test_previous_trading_date_requires_current_date_to_be_a_session():
    calendar = _FakeCalendar([date(2026, 3, 27), date(2026, 3, 30)])
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", True), patch.object(
        trading_calendar,
        "xcals",
        SimpleNamespace(get_calendar=lambda _exchange: calendar),
        create=True,
    ):
        assert trading_calendar.get_previous_trading_date("cn", date(2026, 3, 30)) == date(2026, 3, 27)
        assert trading_calendar.get_previous_trading_date("cn", date(2026, 3, 29)) is None


def test_calendar_unavailable_fails_closed_without_formal_window():
    current_time = datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI)
    with patch.object(trading_calendar, "_XCALS_AVAILABLE", False):
        result = resolve_a_share_premarket_window(current_time)

    assert result.status is PremarketWindowStatus.CALENDAR_UNAVAILABLE
    assert result.should_generate is False
    assert result.window_start is None
    assert result.window_end is None
