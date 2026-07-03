"""Raw GDELT 2.0 DOC API article fetcher with aggressive retry/backoff and
per-window on-disk caching (so a long, interrupted pull can resume cheaply).

This module only fetches raw article metadata (title, url, domain, seendate).
FinBERT scoring happens later in src/data/sentiment_gdelt_finbert.py (M2).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def query_gdelt(
    keyword: str,
    start: datetime,
    end: datetime,
    max_records: int = 250,
    retry_max: int = 8,
    retry_backoff_sec: float = 20.0,
    retry_backoff_max_sec: float = 300.0,
    timeout: float = 30.0,
) -> pd.DataFrame | None:
    """Query the GDELT 2.0 DOC API for one keyword/date-range window.

    Returns an empty DataFrame for a genuinely-empty result (API said "no
    articles"), and None if the window ultimately FAILS after retry_max
    attempts — callers must not cache a None as an empty window.
    """
    params = {
        "query": f'"{keyword}" sourcelang:english',
        "mode": "artlist",
        "format": "json",
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    backoff = retry_backoff_sec
    for attempt in range(1, retry_max + 1):
        try:
            resp = requests.get(GDELT_DOC_URL, params=params, timeout=timeout)
        except requests.RequestException as exc:
            logger.warning("GDELT request error (attempt %d/%d): %s", attempt, retry_max, exc)
            time.sleep(min(backoff, retry_backoff_max_sec))
            backoff *= 1.5
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                # GDELT sometimes returns 200 with a plain-text error body
                # (e.g. rate-limit text or malformed JSON) — treat as retryable.
                logger.warning(
                    "GDELT non-JSON 200 for %r %s-%s (attempt %d/%d): %r",
                    keyword, start.date(), end.date(), attempt, retry_max, resp.text[:120],
                )
                time.sleep(min(backoff, retry_backoff_max_sec))
                backoff *= 1.5
                continue
            articles = data.get("articles", [])
            if not articles:
                return pd.DataFrame(columns=["url", "title", "seendate", "domain", "language"])
            df = pd.DataFrame(articles)
            keep = [c for c in ["url", "title", "seendate", "domain", "language", "sourcecountry"] if c in df.columns]
            return df[keep]

        if resp.status_code == 429:
            logger.info(
                "GDELT 429 for %r %s-%s (attempt %d/%d), backing off %.0fs",
                keyword, start.date(), end.date(), attempt, retry_max, backoff,
            )
            time.sleep(min(backoff, retry_backoff_max_sec))
            backoff *= 1.5
            continue

        logger.warning("GDELT HTTP %d for %r %s-%s: %s", resp.status_code, keyword, start, end, resp.text[:200])
        time.sleep(min(backoff, retry_backoff_max_sec))
        backoff *= 1.5

    logger.error("GDELT: exhausted retries for %r %s-%s, window FAILED", keyword, start.date(), end.date())
    return None


def _daily_windows(start: datetime, end: datetime, chunk_days: int = 3) -> list[tuple[datetime, datetime]]:
    """Case-study windows. NOTE: daily feature resolution does NOT require
    1-day query windows — aggregation uses each article's own `seendate`.
    Multi-day chunks preserve daily resolution while cutting query count
    (observed volume ~20-40 articles/day per keyword << the 250-record cap
    for a 3-day chunk), which matters under GDELT's aggressive rate limit."""
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _monthly_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    windows = []
    cur = start.replace(day=1)
    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        win_start = max(cur, start)
        win_end = min(nxt, end)
        if win_start < win_end:
            windows.append((win_start, win_end))
        cur = nxt
    return windows


def download_gdelt_news_for_ticker(
    ticker: str,
    keywords: list[str],
    start: str,
    end: str | None,
    case_study_window: tuple[str, str],
    raw_dir: Path,
    max_records: int = 250,
    rate_limit_sleep_sec: float = 12.0,
    retry_max: int = 8,
    retry_backoff_sec: float = 20.0,
    retry_backoff_max_sec: float = 300.0,
    min_start: str = "2017-01-01",
    phase: str = "all",           # "all" | "case" (daily windows only) | "monthly"
) -> pd.DataFrame:
    """Fetch raw GDELT articles for one ticker across the full span, using
    daily-resolution windows inside the case-study window and monthly-resolution
    windows elsewhere. Each window is cached separately so reruns are cheap and
    an interrupted job can resume without re-querying completed windows.
    """
    out_dir = raw_dir / "gdelt" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.today() if end is None else pd.to_datetime(end).to_pydatetime()
    # GDELT DOC 2.0 rejects queries before its index start (~2017-01-01).
    start_dt = max(pd.to_datetime(start).to_pydatetime(), pd.to_datetime(min_start).to_pydatetime())
    cs_start = pd.to_datetime(case_study_window[0]).to_pydatetime()
    cs_end = pd.to_datetime(case_study_window[1]).to_pydatetime()

    windows: list[tuple[datetime, datetime, str]] = []
    if start_dt < cs_start:
        for w in _monthly_windows(start_dt, cs_start):
            windows.append((*w, "monthly"))
    for w in _daily_windows(max(start_dt, cs_start), min(end_dt, cs_end)):
        windows.append((*w, "daily"))
    if cs_end < end_dt:
        for w in _monthly_windows(cs_end, end_dt):
            windows.append((*w, "monthly"))

    if phase == "case":
        windows = [w for w in windows if w[2] == "daily"]
    elif phase == "monthly":
        windows = [w for w in windows if w[2] == "monthly"]

    all_frames = []
    for w_start, w_end, granularity in windows:
        cache_name = f"{w_start.strftime('%Y%m%d')}_{w_end.strftime('%Y%m%d')}_{granularity}.csv"
        cache_path = out_dir / cache_name
        if cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            per_keyword = []
            window_failed = False
            for kw in keywords:
                sub = query_gdelt(
                    kw, w_start, w_end,
                    max_records=max_records,
                    retry_max=retry_max,
                    retry_backoff_sec=retry_backoff_sec,
                    retry_backoff_max_sec=retry_backoff_max_sec,
                )
                if sub is None:
                    window_failed = True
                elif not sub.empty:
                    sub["keyword"] = kw
                    per_keyword.append(sub)
                time.sleep(rate_limit_sleep_sec)
            df = pd.concat(per_keyword, ignore_index=True) if per_keyword else pd.DataFrame(
                columns=["url", "title", "seendate", "domain", "language", "keyword"]
            )
            if window_failed:
                # Do NOT cache: a rerun must retry this window rather than
                # mistaking a transient failure for a genuinely news-free window.
                logger.warning(
                    "GDELT %s: window %s-%s (%s) had failed queries, NOT cached",
                    ticker, w_start.date(), w_end.date(), granularity,
                )
            else:
                df.to_csv(cache_path, index=False)
                logger.info(
                    "GDELT %s: window %s-%s (%s) -> %d articles cached",
                    ticker, w_start.date(), w_end.date(), granularity, len(df),
                )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not combined.empty and "url" in combined.columns:
        combined = combined.drop_duplicates(subset=["url"])
    combined_path = raw_dir / "gdelt" / f"{ticker}_combined.csv"
    combined.to_csv(combined_path, index=False)
    return combined
