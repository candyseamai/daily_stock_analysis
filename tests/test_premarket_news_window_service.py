# -*- coding: utf-8 -*-
"""Offline tests for the strict premarket news window service."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.services.premarket_news_window_service import (
    PremarketNewsCandidate,
    filter_premarket_news_window,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
START = datetime(2026, 3, 23, 15, 0, tzinfo=SHANGHAI)
END = datetime(2026, 3, 24, 8, 7, tzinfo=SHANGHAI)
OBSERVED = END


def _candidate(
    published_at=END,
    *,
    title="市场新闻",
    snippet="摘要",
    url="https://example.com/news/1",
    source="示例媒体",
    observed_at=OBSERVED,
):
    return PremarketNewsCandidate(title, snippet, url, source, published_at, observed_at)


def _filter(candidates, *, start=START, end=END, naive_timezone=None):
    return filter_premarket_news_window(
        candidates,
        window_start=start,
        window_end=end,
        naive_timezone=naive_timezone,
    )


def test_closed_window_keeps_both_exact_boundaries_and_rejects_adjacent_seconds():
    candidates = [
        _candidate(START, title="起点", url="https://example.com/start"),
        _candidate(START - timedelta(seconds=1), title="起点前", url="https://example.com/before"),
        _candidate(END, title="终点", url="https://example.com/end"),
        _candidate(END + timedelta(seconds=1), title="终点后", url="https://example.com/after"),
    ]

    result = _filter(candidates)

    assert [item.title for item in result.verified_window_items] == ["终点", "起点"]
    assert result.rejected_before_window_count == 1
    assert result.rejected_after_window_count == 1
    assert {item.reason for item in result.rejected_items} == {"before_window", "after_window"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-24T00:00:00Z", datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI)),
        ("2026-03-24T09:00:00+09:00", datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI)),
        (datetime(2026, 3, 24, 0, 0, tzinfo=UTC), datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI)),
        ("Tue, 24 Mar 2026 00:00:00 GMT", datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI)),
    ],
)
def test_aware_formats_are_converted_to_shanghai(raw, expected):
    result = _filter([_candidate(raw)])
    assert result.verified_window_items[0].published_at == expected


def test_utc_crossing_shanghai_midnight_uses_local_instant():
    start = datetime(2026, 3, 24, 0, 0, tzinfo=SHANGHAI)
    end = datetime(2026, 3, 24, 1, 0, tzinfo=SHANGHAI)
    result = _filter([_candidate("2026-03-23T16:30:00Z")], start=start, end=end)
    assert result.verified_window_items[0].published_at == datetime(2026, 3, 24, 0, 30, tzinfo=SHANGHAI)


def test_monday_and_long_holiday_windows_use_supplied_boundaries_only():
    monday_start = datetime(2026, 3, 27, 15, 0, tzinfo=SHANGHAI)
    monday_end = datetime(2026, 3, 30, 8, 7, tzinfo=SHANGHAI)
    holiday_start = datetime(2026, 9, 30, 15, 0, tzinfo=SHANGHAI)
    holiday_end = datetime(2026, 10, 9, 8, 7, tzinfo=SHANGHAI)

    monday = _filter([_candidate(datetime(2026, 3, 28, 12, 0, tzinfo=SHANGHAI))], start=monday_start, end=monday_end)
    holiday = _filter([_candidate(datetime(2026, 10, 4, 12, 0, tzinfo=SHANGHAI))], start=holiday_start, end=holiday_end)

    assert len(monday.verified_window_items) == 1
    assert len(holiday.verified_window_items) == 1


@pytest.mark.parametrize("raw", ["2026-03-24", None, "", "不是时间", "今天", "昨天", "刚刚"])
def test_non_exact_times_are_time_unverified(raw):
    result = _filter([_candidate(raw)])
    assert result.verified_window_items == ()
    assert len(result.time_unverified_items) == 1
    assert result.time_unverified_items[0].published_at is None


def test_naive_datetime_is_unverified_unless_shanghai_is_explicit():
    raw = datetime(2026, 3, 24, 8, 0)
    default = _filter([_candidate(raw)])
    explicit = _filter([_candidate(raw)], naive_timezone=SHANGHAI)

    assert default.time_unverified_items[0].time_parse_method == "naive_datetime"
    assert explicit.verified_window_items[0].published_at == raw.replace(tzinfo=SHANGHAI)
    assert explicit.verified_window_items[0].time_parse_method == "naive_asia_shanghai"


def test_naive_iso_string_is_unverified_and_date_only_remains_unverified_with_policy():
    naive = _filter([_candidate("2026-03-24T08:00:00")])
    date_only = _filter([_candidate("2026-03-24")], naive_timezone=SHANGHAI)
    rfc_date_only = _filter([_candidate("Tue, 24 Mar 2026")], naive_timezone=SHANGHAI)
    assert naive.time_unverified_items[0].time_parse_method == "naive_datetime"
    assert date_only.time_unverified_items[0].time_parse_method == "date_only"
    assert rfc_date_only.time_unverified_items[0].time_parse_method == "date_only"
    assert rfc_date_only.time_unverified_items[0].published_at is None
    assert rfc_date_only.verified_window_items == ()


@pytest.mark.parametrize(
    "raw",
    [
        "Wed, 24 Mar 2026",
        "Tue, 32 Mar 2026",
        "Foo, 24 Mar 2026",
        "Tue, 24 Xxx 2026",
    ],
)
def test_invalid_rfc_date_only_values_remain_unparseable_with_naive_policy(raw):
    result = _filter([_candidate(raw)], naive_timezone=SHANGHAI)

    assert result.verified_window_items == ()
    assert result.time_unverified_items[0].published_at is None
    assert result.time_unverified_items[0].time_parse_method == "unparseable"


@pytest.mark.parametrize(
    ("raw", "expected", "method"),
    [
        ("7分钟前", datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI), "relative_to_observed_at"),
        ("1 hour ago", datetime(2026, 3, 24, 7, 7, tzinfo=SHANGHAI), "relative_to_observed_at"),
        ("1天前", datetime(2026, 3, 23, 8, 7, tzinfo=SHANGHAI), "relative_to_observed_at"),
    ],
)
def test_numeric_relative_times_are_anchored_to_observed_at(raw, expected, method):
    result = _filter(
        [_candidate(raw)],
        start=datetime(2026, 3, 22, 15, 0, tzinfo=SHANGHAI),
    )
    item = result.verified_window_items[0]
    assert item.published_at == expected
    assert item.time_parse_method == method


def test_unix_seconds_and_milliseconds_are_supported():
    instant = datetime(2026, 3, 24, 0, 0, tzinfo=UTC)
    seconds = int(instant.timestamp())
    result = _filter([
        _candidate(seconds, title="秒", url="https://example.com/seconds"),
        _candidate(seconds * 1000, title="毫秒", url="https://example.com/milliseconds"),
    ])
    assert [item.published_at for item in result.verified_window_items] == [
        datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI),
        datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI),
    ]


def test_observed_at_and_window_boundaries_must_be_aware():
    with pytest.raises(ValueError, match="observed_at"):
        _candidate(observed_at=datetime(2026, 3, 24, 8, 7))
    with pytest.raises(ValueError, match="window_start"):
        _filter([], start=datetime(2026, 3, 23, 15, 0))
    with pytest.raises(ValueError, match="naive_timezone"):
        _filter([], naive_timezone=UTC)


def test_url_tracking_parameters_are_removed_and_duplicates_counted():
    first = _candidate(url="HTTPS://Example.COM:443/news/?id=7&utm_source=x#part", title="甲")
    second = _candidate(url="https://example.com/news?id=7&gclid=y", title="乙")
    result = _filter([first, second])
    assert result.duplicate_count == 1
    assert result.verified_window_items[0].url == "https://example.com/news?id=7"


def test_business_query_parameters_are_preserved_sorted():
    result = _filter([_candidate(url="https://example.com/n?b=2&id=7&a=1&spm=x")])
    assert result.verified_window_items[0].url == "https://example.com/n?a=1&b=2&id=7"


def test_normalized_equal_titles_deduplicate_but_distinct_titles_remain():
    candidates = [
        _candidate(title="重大消息（转载）", url="https://a.example/1", source="媒体A"),
        _candidate(title="重大消息", url="https://b.example/2", source="媒体B"),
        _candidate(title="重大消息的后续细节", url="https://c.example/3", source="媒体C"),
    ]
    result = _filter(candidates)
    assert result.duplicate_count == 1
    assert {item.title for item in result.verified_window_items} == {"重大消息（转载）", "重大消息的后续细节"}


def test_official_source_wins_duplicate_before_verified_and_input_order():
    candidates = [
        _candidate(title="政策发布", url="https://media.example/1", source="转载媒体"),
        _candidate("2026-03-24", title="政策发布", url="https://www.csrc.gov.cn/notice", source="证监会"),
    ]
    result = _filter(candidates)
    assert result.duplicate_count == 1
    assert result.verified_window_items == ()
    assert result.time_unverified_items[0].source == "证监会"


def test_same_priority_duplicate_keeps_earlier_input():
    candidates = [
        _candidate(title="同题", url="https://a.example/1", source="先到媒体"),
        _candidate(title="同题", url="https://b.example/2", source="后到媒体"),
    ]
    assert _filter(candidates).verified_window_items[0].source == "先到媒体"


def test_verified_time_wins_unverified_duplicate_for_non_official_sources():
    candidates = [
        _candidate("2026-03-24", title="同一新闻", url="https://a.example/1", source="媒体A"),
        _candidate(END, title="同一新闻", url="https://b.example/2", source="媒体B"),
    ]
    result = _filter(candidates)
    assert result.verified_window_items[0].source == "媒体B"
    assert result.time_unverified_items == ()


def test_invalid_title_and_url_are_rejected_with_fixed_reasons():
    result = _filter([
        _candidate(title="  ", url="https://example.com/1"),
        _candidate(title="有效标题", url="javascript:alert(1)"),
    ])
    assert [item.reason for item in result.rejected_items] == ["invalid_title", "invalid_url"]


def test_missing_source_is_derived_from_domain_and_snippet_may_be_empty():
    result = _filter([_candidate(source="", snippet="", url="https://News.Example.com/item")])
    item = result.verified_window_items[0]
    assert item.source == "news.example.com"
    assert item.snippet == ""
    assert result.warnings == ("source_derived_from_url:1",)


def test_input_is_not_modified_and_outputs_are_frozen_tuples():
    candidate = _candidate()
    candidates = [candidate]
    before = list(candidates)
    result = _filter(candidates)

    assert candidates == before
    assert isinstance(result.verified_window_items, tuple)
    assert isinstance(result.rejected_items, tuple)
    with pytest.raises(FrozenInstanceError):
        result.duplicate_count = 99
    with pytest.raises(FrozenInstanceError):
        candidate.title = "changed"


def test_item_id_is_stable_across_repeated_runs():
    candidate = _candidate(url="https://example.com/news?utm_source=a&id=1")
    first = _filter([candidate]).verified_window_items[0].item_id
    second = _filter([candidate]).verified_window_items[0].item_id
    assert first == second
    assert len(first) == 24


def test_every_published_at_is_timezone_aware_and_verified_sorted_newest_first():
    candidates = [
        _candidate(datetime(2026, 3, 24, 7, 0, tzinfo=SHANGHAI), title="早", url="https://example.com/early"),
        _candidate(datetime(2026, 3, 24, 8, 0, tzinfo=SHANGHAI), title="晚", url="https://example.com/late"),
    ]
    result = _filter(candidates)
    assert [item.title for item in result.verified_window_items] == ["晚", "早"]
    assert all(item.published_at is not None and item.published_at.utcoffset() is not None for item in result.verified_window_items)


def test_unverified_items_keep_input_order():
    result = _filter([
        _candidate(None, title="一", url="https://example.com/1"),
        _candidate("今天", title="二", url="https://example.com/2"),
    ])
    assert [item.title for item in result.time_unverified_items] == ["一", "二"]


def test_service_has_no_external_service_imports():
    source = Path("src/services/premarket_news_window_service.py").read_text(encoding="utf-8")
    forbidden = ("search_service", "SearchResult", "LLM", "NotificationService", "requests", "httpx")
    assert not any(name in source for name in forbidden)
