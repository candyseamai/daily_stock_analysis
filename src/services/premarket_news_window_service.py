# -*- coding: utf-8 -*-
"""Pure, auditable filtering for the A-share overnight news window."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
_TRACKING_PARAMETERS = {"gclid", "fbclid", "spm"}
_FUZZY_RELATIVE_TIMES = {
    "今天",
    "今日",
    "昨天",
    "前天",
    "刚刚",
    "today",
    "yesterday",
    "just now",
    "now",
}
_REPOST_SUFFIXES = (
    "-转载",
    "—转载",
    "_转载",
    "（转载）",
    "(转载)",
    "【转载】",
)
_RFC_DATE_ONLY_PATTERN = re.compile(
    r"^(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(?P<year>\d{4})$",
    flags=re.IGNORECASE,
)
_RFC_WEEKDAYS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
_RFC_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass(frozen=True)
class PremarketNewsCandidate:
    """Lightweight provider result observed at a known instant."""

    title: str
    snippet: str
    url: str
    source: str
    raw_published_date: Any
    observed_at: datetime

    def __post_init__(self) -> None:
        if not _is_aware(self.observed_at):
            raise ValueError("observed_at must be a timezone-aware datetime")


@dataclass(frozen=True)
class PremarketNewsItem:
    """A validated news fact without market interpretation."""

    item_id: str
    title: str
    snippet: str
    url: str
    source: str
    raw_published_date: Any
    observed_at: datetime
    published_at: Optional[datetime]
    time_parse_method: str


@dataclass(frozen=True)
class RejectedNewsItem:
    """A rejected candidate represented by a fixed, non-sensitive reason."""

    item_id: str
    title: str
    url: str
    reason: str


@dataclass(frozen=True)
class PremarketNewsWindowResult:
    """Strict partition of candidates for one exact premarket window."""

    window_start: datetime
    window_end: datetime
    verified_window_items: tuple[PremarketNewsItem, ...]
    time_unverified_items: tuple[PremarketNewsItem, ...]
    rejected_items: tuple[RejectedNewsItem, ...]
    duplicate_count: int
    rejected_before_window_count: int
    rejected_after_window_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedCandidate:
    index: int
    item: PremarketNewsItem
    canonical_url: str
    normalized_title: str
    official_source: bool


def _is_aware(value: Any) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title or "").strip()
    normalized = normalized.translate(
        str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "！": "!", "？": "?"})
    )
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    for suffix in _REPOST_SUFFIXES:
        if normalized.endswith(suffix.lower()):
            normalized = normalized[: -len(suffix)].rstrip()
            break
    return normalized


def _canonicalize_url(url: str) -> Optional[str]:
    try:
        parsed = urlsplit((url or "").strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        scheme = parsed.scheme.lower()
        host = parsed.hostname.lower()
        try:
            port = parsed.port
        except ValueError:
            return None
        if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        query = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered = key.lower()
            if lowered.startswith("utm_") or lowered in _TRACKING_PARAMETERS:
                continue
            query.append((key, value))
        query.sort(key=lambda pair: (pair[0], pair[1]))
        return urlunsplit((scheme, host, path, urlencode(query, doseq=True), ""))
    except (TypeError, ValueError):
        return None


def _stable_item_id(*parts: Any) -> str:
    material = "\x1f".join(unicodedata.normalize("NFKC", str(part or "")).strip() for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _source_from_url(canonical_url: str) -> str:
    return urlsplit(canonical_url).hostname or ""


def _is_official_source(source: str, canonical_url: str) -> bool:
    haystack = f"{source} {urlsplit(canonical_url).hostname or ''}".lower()
    official_markers = (
        ".gov.cn",
        "gov.cn",
        "pbc.gov.cn",
        "csrc.gov.cn",
        "sse.com.cn",
        "szse.cn",
        "bse.cn",
        "cninfo.com.cn",
        "国务院",
        "证监会",
        "人民银行",
        "央行",
        "交易所",
        "公司公告",
        "官方公告",
    )
    return any(marker in haystack for marker in official_markers)


def _parse_exact_relative(text: str, observed_at: datetime) -> Optional[datetime]:
    zh = re.fullmatch(r"\s*(\d+)\s*(分钟|小时|天)\s*前\s*", text)
    if zh:
        amount = int(zh.group(1))
        unit = zh.group(2)
        delta = {
            "分钟": timedelta(minutes=amount),
            "小时": timedelta(hours=amount),
            "天": timedelta(days=amount),
        }[unit]
        return observed_at - delta

    en = re.fullmatch(
        r"\s*(\d+)\s*(minute|minutes|min|mins|hour|hours|day|days)\s+ago\s*",
        text.lower(),
    )
    if not en:
        return None
    amount = int(en.group(1))
    unit = en.group(2)
    if unit in {"minute", "minutes", "min", "mins"}:
        return observed_at - timedelta(minutes=amount)
    if unit in {"hour", "hours"}:
        return observed_at - timedelta(hours=amount)
    return observed_at - timedelta(days=amount)


def _parse_published_at(
    value: Any,
    observed_at: datetime,
    naive_timezone: Optional[tzinfo],
) -> tuple[Optional[datetime], str]:
    if value is None:
        return None, "missing"
    if isinstance(value, datetime):
        if not _is_aware(value):
            if naive_timezone is None:
                return None, "naive_datetime"
            value = value.replace(tzinfo=naive_timezone)
            return value.astimezone(SHANGHAI_TIMEZONE), "naive_asia_shanghai"
        return value.astimezone(SHANGHAI_TIMEZONE), "aware_datetime"
    if isinstance(value, date):
        return None, "date_only"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None, "unparseable"
        seconds = numeric / 1000 if abs(numeric) >= 100_000_000_000 else numeric
        try:
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None, "unparseable"
        return parsed.astimezone(SHANGHAI_TIMEZONE), "unix_timestamp"

    text = str(value).strip()
    if not text:
        return None, "missing"
    lower = text.lower()
    if lower in _FUZZY_RELATIVE_TIMES:
        return None, "fuzzy_relative"

    relative = _parse_exact_relative(text, observed_at)
    if relative is not None:
        return relative.astimezone(SHANGHAI_TIMEZONE), "relative_to_observed_at"

    if re.fullmatch(r"\d{10}|\d{13}", text):
        seconds = int(text) / (1000 if len(text) == 13 else 1)
        try:
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None, "unparseable"
        return parsed.astimezone(SHANGHAI_TIMEZONE), "unix_timestamp"

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        try:
            date.fromisoformat(text)
        except ValueError:
            return None, "unparseable"
        return None, "date_only"

    rfc_date_only = _RFC_DATE_ONLY_PATTERN.fullmatch(text)
    if rfc_date_only is not None:
        try:
            parsed_date = date(
                int(rfc_date_only.group("year")),
                _RFC_MONTHS[rfc_date_only.group("month").lower()],
                int(rfc_date_only.group("day")),
            )
        except (KeyError, ValueError):
            return None, "unparseable"
        expected_weekday = _RFC_WEEKDAYS[rfc_date_only.group("weekday").lower()]
        if parsed_date.weekday() != expected_weekday:
            return None, "unparseable"
        return None, "date_only"

    iso_candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed_iso = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed_iso = None
    if parsed_iso is not None:
        if not _is_aware(parsed_iso):
            if naive_timezone is None:
                return None, "naive_datetime"
            parsed_iso = parsed_iso.replace(tzinfo=naive_timezone)
            return parsed_iso.astimezone(SHANGHAI_TIMEZONE), "naive_asia_shanghai"
        return parsed_iso.astimezone(SHANGHAI_TIMEZONE), "iso8601"

    try:
        parsed_rfc = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed_rfc = None
    if parsed_rfc is not None:
        if not re.search(r"\d{1,2}:\d{2}", text):
            return None, "date_only"
        if not _is_aware(parsed_rfc):
            if naive_timezone is None:
                return None, "naive_datetime"
            parsed_rfc = parsed_rfc.replace(tzinfo=naive_timezone)
            return parsed_rfc.astimezone(SHANGHAI_TIMEZONE), "naive_asia_shanghai"
        return parsed_rfc.astimezone(SHANGHAI_TIMEZONE), "rfc_datetime"
    return None, "unparseable"


def _validate_naive_timezone(naive_timezone: Optional[tzinfo]) -> None:
    if naive_timezone is None:
        return
    if naive_timezone != SHANGHAI_TIMEZONE:
        raise ValueError("naive_timezone must be Asia/Shanghai when provided")


def filter_premarket_news_window(
    candidates: Iterable[PremarketNewsCandidate],
    *,
    window_start: datetime,
    window_end: datetime,
    naive_timezone: Optional[tzinfo] = None,
) -> PremarketNewsWindowResult:
    """Validate, de-duplicate and partition facts by one exact closed window."""
    if not _is_aware(window_start) or not _is_aware(window_end):
        raise ValueError("window_start and window_end must be timezone-aware datetimes")
    _validate_naive_timezone(naive_timezone)
    start = window_start.astimezone(SHANGHAI_TIMEZONE)
    end = window_end.astimezone(SHANGHAI_TIMEZONE)
    if end < start:
        raise ValueError("window_end must not be earlier than window_start")

    prepared: list[_PreparedCandidate] = []
    rejected: list[RejectedNewsItem] = []
    derived_source_count = 0

    for index, candidate in enumerate(tuple(candidates)):
        if not isinstance(candidate, PremarketNewsCandidate):
            raise TypeError("candidates must contain PremarketNewsCandidate instances")
        title = (candidate.title or "").strip()
        normalized_title = _normalize_title(title)
        raw_url = (candidate.url or "").strip()
        candidate_id = _stable_item_id(normalized_title, raw_url, candidate.source, candidate.raw_published_date)
        if not normalized_title:
            rejected.append(RejectedNewsItem(candidate_id, title, raw_url, "invalid_title"))
            continue
        canonical_url = _canonicalize_url(raw_url)
        if canonical_url is None:
            rejected.append(RejectedNewsItem(candidate_id, title, raw_url, "invalid_url"))
            continue

        source = (candidate.source or "").strip()
        if not source:
            source = _source_from_url(canonical_url)
            derived_source_count += 1
        published_at, parse_method = _parse_published_at(
            candidate.raw_published_date,
            candidate.observed_at,
            naive_timezone,
        )
        if published_at is not None and not _is_aware(published_at):
            raise AssertionError("published_at normalization produced a naive datetime")
        item_id = _stable_item_id(normalized_title, canonical_url)
        item = PremarketNewsItem(
            item_id=item_id,
            title=title,
            snippet=candidate.snippet or "",
            url=canonical_url,
            source=source,
            raw_published_date=candidate.raw_published_date,
            observed_at=candidate.observed_at,
            published_at=published_at,
            time_parse_method=parse_method,
        )
        prepared.append(
            _PreparedCandidate(
                index=index,
                item=item,
                canonical_url=canonical_url,
                normalized_title=normalized_title,
                official_source=_is_official_source(source, canonical_url),
            )
        )

    groups: list[list[_PreparedCandidate]] = []
    for candidate in prepared:
        matching = [
            group
            for group in groups
            if any(
                member.canonical_url == candidate.canonical_url
                or member.normalized_title == candidate.normalized_title
                for member in group
            )
        ]
        if not matching:
            groups.append([candidate])
            continue
        primary = matching[0]
        primary.append(candidate)
        for extra in matching[1:]:
            primary.extend(extra)
            groups.remove(extra)

    representatives: list[_PreparedCandidate] = []
    duplicate_count = 0
    for group in groups:
        duplicate_count += len(group) - 1
        representatives.append(
            min(
                group,
                key=lambda entry: (
                    not entry.official_source,
                    entry.item.published_at is None,
                    entry.index,
                ),
            )
        )

    verified: list[_PreparedCandidate] = []
    unverified: list[_PreparedCandidate] = []
    before_count = 0
    after_count = 0
    for entry in representatives:
        published_at = entry.item.published_at
        if published_at is None:
            unverified.append(entry)
        elif published_at < start:
            before_count += 1
            rejected.append(
                RejectedNewsItem(entry.item.item_id, entry.item.title, entry.item.url, "before_window")
            )
        elif published_at > end:
            after_count += 1
            rejected.append(
                RejectedNewsItem(entry.item.item_id, entry.item.title, entry.item.url, "after_window")
            )
        else:
            verified.append(entry)

    verified.sort(
        key=lambda entry: (
            -(entry.item.published_at.timestamp() if entry.item.published_at is not None else 0.0),
            entry.index,
        )
    )
    unverified.sort(key=lambda entry: entry.index)

    warnings: list[str] = []
    if derived_source_count:
        warnings.append(f"source_derived_from_url:{derived_source_count}")
    if unverified:
        warnings.append(f"time_unverified:{len(unverified)}")
    if duplicate_count:
        warnings.append(f"duplicates_removed:{duplicate_count}")

    return PremarketNewsWindowResult(
        window_start=start,
        window_end=end,
        verified_window_items=tuple(entry.item for entry in verified),
        time_unverified_items=tuple(entry.item for entry in unverified),
        rejected_items=tuple(rejected),
        duplicate_count=duplicate_count,
        rejected_before_window_count=before_count,
        rejected_after_window_count=after_count,
        warnings=tuple(warnings),
    )


__all__ = [
    "PremarketNewsCandidate",
    "PremarketNewsItem",
    "PremarketNewsWindowResult",
    "RejectedNewsItem",
    "SHANGHAI_TIMEZONE",
    "filter_premarket_news_window",
]
