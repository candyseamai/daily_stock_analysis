# -*- coding: utf-8 -*-
"""Resolve the exact A-share premarket reporting window."""

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from src.core.trading_calendar import get_market_now, get_previous_trading_date, is_market_open


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
XSHG_MARKET = "cn"
PREVIOUS_SESSION_CLOSE = time(15, 0)


class PremarketWindowStatus(str, Enum):
    """Outcome of resolving a premarket window."""

    READY = "ready"
    SKIPPED_NON_TRADING_DAY = "skipped_non_trading_day"
    CALENDAR_UNAVAILABLE = "calendar_unavailable"


@dataclass(frozen=True)
class PremarketWindow:
    """A timezone-aware A-share premarket news window resolution result."""

    status: PremarketWindowStatus
    run_at: datetime
    session_date: date
    previous_trading_date: Optional[date] = None
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    reason: Optional[str] = None

    @property
    def should_generate(self) -> bool:
        return self.status is PremarketWindowStatus.READY


def resolve_a_share_premarket_window(current_time: Optional[datetime] = None) -> PremarketWindow:
    """Resolve the A-share premarket window for the actual run time.

    A naive ``current_time`` follows the repository-wide trading-calendar rule:
    it is interpreted as an Asia/Shanghai wall-clock time. An aware datetime is
    converted to Asia/Shanghai. A formal window is returned only when the local
    date is an XSHG trading session and its previous session can be confirmed.
    """
    run_at = get_market_now(XSHG_MARKET, current_time=current_time)
    session_date = run_at.date()

    if not is_market_open(XSHG_MARKET, session_date):
        return PremarketWindow(
            status=PremarketWindowStatus.SKIPPED_NON_TRADING_DAY,
            run_at=run_at,
            session_date=session_date,
            reason="current Asia/Shanghai date is not an XSHG trading session",
        )

    previous_trading_date = get_previous_trading_date(XSHG_MARKET, session_date)
    if previous_trading_date is None:
        return PremarketWindow(
            status=PremarketWindowStatus.CALENDAR_UNAVAILABLE,
            run_at=run_at,
            session_date=session_date,
            reason="unable to confirm the previous XSHG trading session",
        )

    window_start = datetime.combine(
        previous_trading_date,
        PREVIOUS_SESSION_CLOSE,
        tzinfo=SHANGHAI_TIMEZONE,
    )
    return PremarketWindow(
        status=PremarketWindowStatus.READY,
        run_at=run_at,
        session_date=session_date,
        previous_trading_date=previous_trading_date,
        window_start=window_start,
        window_end=run_at,
    )
