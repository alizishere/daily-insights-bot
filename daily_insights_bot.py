# daily_insights_bot.py
"""
Automated Daily Leadership Insights Bot
--------------------------------------
Fetches the five most important leadership & strategy articles each day,
creates English summaries + Persian (Farsi) translations, and delivers the
message to a Telegram chat at 09:30 Asia/Tehran.

Deployment
~~~~~~~~~~
1. **Create a free service** on Render.com or Railway.app (recommended: cron job).
2. **Add environment variables** in the dashboard:
   - `OPENAI_API_KEY`  : Your OpenAI key.
   - `TELEGRAM_BOT_TOKEN` : The provided bot token.
   - `TELEGRAM_CHAT_ID`   : The provided chat ID.
3. **Schedule the script** to run daily at `06:00 UTC`, which corresponds to
   `09:30 Asia/Tehran` year-round (Tehran is UTC+3:30 and no longer uses DST).
   - **Render**:  Settings ▸ Cron > “0 6 * * *”.
   - **Railway**:  New ▸ Cron Job > schedule "0 6 * * *".
4. **Add a `requirements.txt`** (see bottom of this file) and enable automatic
   builds.

Local quick test:
```
python daily_insights_bot.py --run-once
```
"""
from __future__ import annotations

import os
import html
import logging
import textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import feedparser
import openai
import pytz
import requests

# Optional but greatly improves full-text extraction; comment out on slim images
try:
    from newspaper import Article  # type: ignore
except ImportError:  # newspaper3k not installed
    Article = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Feeds chosen for consistent availability of free leadership/strategy content
RSS_FEEDS = [
    ("McKinsey Insights", "https://www.mckinsey.com/featured-insights/rss"),
    ("Harvard Business Review", "https://feeds.harvardbusiness.org/harvardbusiness"),
    ("Fortune Leadership", "https://fortune.com/category/leadership/feed"),
    ("World Economic Forum", "https://www.weforum.org/agenda/feed"),
    ("KPMG", "https://home.kpmg/us/en/blogs/home.rss"),
]

MAX_ITEMS = 5     # deliver exactly five insights
MAX_ARTICLE_AGE_HOURS = 36  # consider items published within last 36 h

# Models – adapt if you have different quota
MODEL_SUMMARY = "gpt-4o"
MODEL_TRANSLATE = "gpt-4o-mini"  # cheaper model for translation

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_recent_entries() -> List[Tuple[str, str, str, datetime]]:
    """Return list of (title, link, summary_html, published_dt) sorted newest-first."""
    cutoff = datetime.utcnow() - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
    entries: List[Tuple[str, str, str, datetime]] = []
    for source, url in RSS_FEEDS:
        d = feedparser.parse(url)
        for e in d.entries:
            # Try to parse publication date; skip if unavailable
            if "published_parsed" not in e:
                continue
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if published < cutoff:
                continue
            title = e.get("title", "(no title)")
            link = e.get("link", "")
            summary = e.get("summary", "")
            entries.append((title, link, summary, published))
    # Sort by recency
    entries.sort(key=lambda x: x[3], reverse=True)
    return entries[: MAX_ITEMS * 2]  # take extra in case some fail later


def extract_text(url: str, fallback_html: str) -> str:
    """Return clean article text either via newspaper3k or by stripping HTML."""
    if Article:
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text:
                return article.text
        except Exception as exc:  # pragma: no cover
            logging.warning("newspaper3k failed: %s", exc)
    # Fallback: strip tags from feed summary
    return html.unescape(
        textwrap.shorten(
            " ".join(html.unescape(fallback_html).split()), width=2000, placeholder="…"
        )
    )


def chat_completion(model: str, system: str, user: str) -> str:
    """Wrapper to call OpenAI with sensible defaults."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=300,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def summarize(article_text: str) -> str:
    system = "You are a strategy editor writing concise executive summaries."
    user = (
        "Summarize the following text in 2–3 formal sentences for a senior business leader "
        "(clarity, no buzzwords):\n\n" + article_text
    )
    return chat_completion(MODEL_SUMMARY, system, user)


def translate_persian(summary: str) -> str:
    system = "You are a professional translator proficient in Persian (Farsi)."
    user = (
        "Translate the following English executive summary into formal Persian (Farsi) while keeping the tone professional and concise:\n\n"
        + summary
    )
    return chat_completion(MODEL_TRANSLATE, system, user)


def build_message(items: List[Tuple[str, str, str]]) -> str:
    """Assemble final Telegram message string within 4096-char limit."""
    lines = []
    for idx, (title, en, fa) in enumerate(items, 1):
        lines.append(f"📌 Insight #{idx}:")
        lines.append(f"🗞 Title: {html.escape(title)}")
        lines.append(f"✍️ English Summary (Formal): {html.escape(en)}")
        lines.append(f"🈯 Persian Translation (Formal): {html.escape(fa)}")
        lines.append("\n")  # blank line
    return "\n".join(lines)[:4000]  # Telegram limit safety


def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram credentials missing.")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text})
    if not resp.ok:
        raise RuntimeError(f"Telegram error: {resp.text}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_once() -> None:
    logging.info("Fetching recent entries…")
    entries = fetch_recent_entries()
    processed: List[Tuple[str, str, str]] = []
    for title, link, summary_html, _ in entries:
        if len(processed) >= MAX_ITEMS:
            break
        logging.info("Processing: %s", title)
        full_text = extract_text(link, summary_html)
        try:
            en_summary = summarize(full_text)
            fa_summary = translate_persian(en_summary)
            processed.append((title, en_summary, fa_summary))
        except Exception as exc:
            logging.error("OpenAI failed for '%s': %s", title, exc)
    if not processed:
        logging.warning("No items processed – aborting send.")
        return
    message = build_message(processed)
    send_telegram(message)
    logging.info("Message sent to Telegram (length %s)", len(message))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily Leadership Insights Bot")
    parser.add_argument("--run-once", action="store_true", help="Execute immediately instead of schedule")
    args = parser.parse_args()

    if args.run_once:
        run_once()
    else:
        # Run now and then sleep until next 09:30 Asia/Tehran
        tz_tehran = pytz.timezone("Asia/Tehran")
        while True:
            now_utc = datetime.now(timezone.utc)
            now_teh = now_utc.astimezone(tz_tehran)
            target = now_teh.replace(hour=9, minute=30, second=0, microsecond=0)
            if now_teh >= target:
                target += timedelta(days=1)
            sleep_seconds = (target - now_teh).total_seconds()
            logging.info("Sleeping %.1f h until next run at %s Tehran", sleep_seconds / 3600, target)
            import time

            time.sleep(sleep_seconds)
            run_once()

# ---------------------------------------------------------------------------
# requirements.txt (place in separate file when deploying)
# ---------------------------------------------------------------------------
# feedparser
# openai>=1.13.3
# requests
# python-dateutil
# pytz
# python-telegram-bot==20.8  # optional, we use raw requests so not strictly needed
# newspaper3k==0.2.8  # optional but recommended
# schedule==1.2.1  # only if you prefer schedule library instead of manual sleep
