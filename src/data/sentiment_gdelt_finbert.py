"""FinBERT sentiment scoring over cached GDELT headlines + daily aggregation.

Pipeline:
1. Read cached per-window GDELT article CSVs for a ticker (from gdelt_raw).
2. Score each unique headline with FinBERT (ProsusAI/finbert) -> P_pos, P_neu,
   P_neg, signed = P_pos - P_neg. Scores are cached to disk keyed by URL so
   reruns never re-score.
3. Aggregate per (ticker, trading day):
     sent_mean    - mean signed score of that day's articles
     sent_polarity- (n_pos - n_neg) / n_articles  (article-level majority sign)
     news_volume  - log1p(article count)
     sent_decay   - 3-day exponentially decay-weighted mean signed score
     has_news     - 1 if any article that day, else 0
   Days with no articles get neutral 0 + has_news=0 (documented missingness).

Temporal resolution note: articles fetched with monthly-granularity queries
(outside the case-study window) carry their own per-article `seendate`, so
aggregation is still done on true article dates. Coverage per day is simply
sparser outside the case-study window (max 250 articles per keyword per month).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()


def load_raw_articles(raw_dir: Path, ticker: str) -> pd.DataFrame:
    """Load and deduplicate all cached GDELT windows for a ticker."""
    win_dir = raw_dir / "gdelt" / ticker
    frames = []
    if win_dir.exists():
        for p in sorted(win_dir.glob("*.csv")):
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["url", "title", "seendate"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["url", "title", "seendate"]).drop_duplicates(subset=["url"])
    return out


def score_headlines_finbert(
    articles: pd.DataFrame,
    cache_path: Path,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    max_length: int = 64,
) -> pd.DataFrame:
    """Score article titles with FinBERT; incremental on-disk cache by URL hash.

    Returns frame with columns [url, p_pos, p_neu, p_neg, signed].
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame(columns=["url_key", "p_pos", "p_neu", "p_neg", "signed"])

    articles = articles.copy()
    articles["url_key"] = articles["url"].map(_url_key)
    todo = articles[~articles["url_key"].isin(set(cached["url_key"]))]

    if not todo.empty:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
        id2label = {i: l.lower() for i, l in model.config.id2label.items()}

        rows = []
        titles = todo["title"].astype(str).tolist()
        keys = todo["url_key"].tolist()
        with torch.no_grad():
            for i in range(0, len(titles), batch_size):
                batch = titles[i : i + batch_size]
                enc = tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
                for j, p in enumerate(probs):
                    by_label = {id2label[k]: float(p[k]) for k in range(len(p))}
                    rows.append({
                        "url_key": keys[i + j],
                        "p_pos": by_label["positive"],
                        "p_neu": by_label["neutral"],
                        "p_neg": by_label["negative"],
                        "signed": by_label["positive"] - by_label["negative"],
                    })
                if (i // batch_size) % 20 == 0:
                    logger.info("FinBERT: scored %d/%d headlines", min(i + batch_size, len(titles)), len(titles))
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache_path, index=False)
        logger.info("FinBERT: cache now holds %d scored headlines -> %s", len(cached), cache_path)

    merged = articles.merge(cached, on="url_key", how="left")
    return merged[["url", "title", "seendate", "p_pos", "p_neu", "p_neg", "signed"]]


def aggregate_daily_sentiment(
    scored: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    decay_days: int = 3,
) -> pd.DataFrame:
    """Aggregate scored articles to the trading-day grid (see module docstring)."""
    cols = ["sent_mean", "sent_polarity", "news_volume", "sent_decay", "has_news"]
    if scored.empty:
        out = pd.DataFrame(0.0, index=trading_days, columns=cols)
        out["has_news"] = 0.0
        return out

    df = scored.copy()
    df["date"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "signed"])

    daily = df.groupby("date").agg(
        sent_mean=("signed", "mean"),
        n_articles=("signed", "size"),
        n_pos=("signed", lambda s: int((s > 0.05).sum())),
        n_neg=("signed", lambda s: int((s < -0.05).sum())),
    )
    daily["sent_polarity"] = (daily["n_pos"] - daily["n_neg"]) / daily["n_articles"]
    daily["news_volume"] = np.log1p(daily["n_articles"])

    # Map article calendar dates onto the trading grid: news on non-trading days
    # rolls forward to the NEXT trading day (it can only influence the next session).
    grid = pd.DataFrame(index=trading_days)
    pos = trading_days.searchsorted(daily.index, side="left")
    valid = pos < len(trading_days)
    daily = daily[valid]
    daily["grid_day"] = trading_days[pos[valid]]
    on_grid = daily.groupby("grid_day").agg(
        sent_mean=("sent_mean", "mean"),
        sent_polarity=("sent_polarity", "mean"),
        n_articles=("n_articles", "sum"),
    )
    grid = grid.join(on_grid)
    grid["news_volume"] = np.log1p(grid["n_articles"].fillna(0.0))
    grid["has_news"] = (~grid["sent_mean"].isna()).astype(float)
    grid[["sent_mean", "sent_polarity"]] = grid[["sent_mean", "sent_polarity"]].fillna(0.0)

    # 3-day decay-weighted score over trading days (uses only past/current days)
    alpha = 1.0 - np.exp(-1.0 / decay_days)
    grid["sent_decay"] = grid["sent_mean"].ewm(alpha=alpha, adjust=False).mean()

    return grid[cols].astype(float)


def build_sentiment_features(
    raw_dir: Path,
    processed_dir: Path,
    tickers: list[str],
    trading_days: pd.DatetimeIndex,
    model_name: str = "ProsusAI/finbert",
    decay_days: int = 3,
) -> pd.DataFrame:
    """Full sentiment feature matrix: per-ticker daily features, columns
    prefixed `{ticker}_`. Missing tickers/days -> neutral + has_news=0."""
    frames = []
    for ticker in tickers:
        articles = load_raw_articles(raw_dir, ticker)
        logger.info("sentiment %s: %d unique articles", ticker, len(articles))
        scored = score_headlines_finbert(
            articles, processed_dir / "finbert_cache" / f"{ticker}.csv", model_name
        )
        daily = aggregate_daily_sentiment(scored, trading_days, decay_days)
        daily.columns = [f"{ticker}_{c}" for c in daily.columns]
        frames.append(daily)
    return pd.concat(frames, axis=1)
