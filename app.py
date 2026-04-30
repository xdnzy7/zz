from __future__ import annotations

import contextlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
import io
from io import StringIO
import logging
import random
import re
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

APP_VERSION = "pre-move-momentum-scanner-polished-ui-2026-04-30"
YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
DEFAULT_OPTIONAL_WATCHLIST = ""

FALLBACK_TICKERS = [
    "AKAN",
    "RDAC",
    "BIYA",
    "SBLX",
    "ATER",
    "MULN",
    "HOLO",
    "GNS",
    "LUCY",
    "WISA",
    "FFIE",
    "TIVC",
    "SOUN",
    "BBAI",
    "KULR",
    "OPEN",
    "PLUG",
]

SKIP_KEYS = [
    "missing_data",
    "price_filter",
    "no_momentum",
    "stale_data",
]

SECURITY_NAME_CACHE: dict[str, str] = {}
ACTIVE_QUOTE_CACHE: dict[str, dict[str, object]] = {}


@dataclass
class ScannerSettings:
    cloud_fast_mode: bool = True
    scan_interval_seconds: int = 60
    scan_timeout_seconds: int = 20
    max_tickers: int = 100
    max_workers: int = 8
    min_price: float = 0.5
    max_price: float = 200.0
    min_volume: int = 0
    min_relative_volume: float = 0.8
    near_high_threshold_pct: float = 3.0
    output_limit: int = 100
    request_timeout_seconds: int = 3
    cache_seconds: int = 20
    optional_watchlist: str = DEFAULT_OPTIONAL_WATCHLIST


@dataclass
class AppState:
    version: str = APP_VERSION
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    scanner_thread: Optional[threading.Thread] = None
    settings: ScannerSettings = field(default_factory=ScannerSettings)
    candidates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    watched_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_debug_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skip_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in SKIP_KEYS})
    cycle_count: int = 0
    scanned_count: int = 0
    candidate_count: int = 0
    last_scan_started: Optional[str] = None
    last_scan_finished: Optional[str] = None
    last_scan_seconds: float = 0.0
    timed_out: bool = False
    status_text: str = "Starting scanner..."
    source_message: Optional[str] = None
    error: Optional[str] = None
    download_cache: dict[str, tuple[float, pd.DataFrame]] = field(default_factory=dict)
    universe_cache: tuple[float, list[str]] | None = None


@st.cache_resource(show_spinner=False)
def get_state() -> AppState:
    return AppState()


def is_valid_ticker(ticker: str) -> bool:
    return bool(VALID_TICKER_RE.fullmatch(str(ticker).upper().strip()))


def parse_optional_watchlist(value: str) -> list[str]:
    tickers = [part.upper().strip() for part in re.split(r"[\s,;]+", value or "")]
    return [ticker for ticker in dict.fromkeys(tickers) if is_valid_ticker(ticker)]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def fmt_price(value: object) -> str:
    value = safe_float(value, np.nan)
    return "N/A" if np.isnan(value) else f"{value:.2f}"


def fmt_num(value: object) -> str:
    return f"{int(safe_float(value, 0)):,}"


def fmt_currency(value: object) -> str:
    value = safe_float(value, np.nan)
    return "N/A" if np.isnan(value) else f"${value:.2f}"


def fmt_pct(value: object) -> str:
    value = safe_float(value, np.nan)
    return "N/A" if np.isnan(value) else f"{value:.2f}%"


def fmt_rvol(value: object) -> str:
    value = safe_float(value, np.nan)
    return "N/A" if np.isnan(value) else f"{value:.2f}"


def fmt_score(value: object) -> str:
    value = safe_float(value, 0)
    return f"{value:.0f}/100"


def round_cent(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def status_tone(status: str) -> str:
    if status == "VALID TRADE":
        return "green"
    if status in {"❌ NO LONG", "INVALID DATA"}:
        return "red"
    return "orange"


def status_badge(status: str) -> str:
    tone = status_tone(status)
    return f":{tone}[{status}]"


def safe_request_json(url: str, params: dict[str, object], timeout: int = 4) -> dict[str, object]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def add_quote_from_yahoo(symbol: str, quote: dict[str, object]) -> None:
    ACTIVE_QUOTE_CACHE[symbol] = {
        "price": quote.get("regularMarketPrice") or quote.get("preMarketPrice") or quote.get("postMarketPrice"),
        "previous_close": quote.get("regularMarketPreviousClose"),
        "change_pct": quote.get("regularMarketChangePercent") or quote.get("preMarketChangePercent"),
        "volume": quote.get("regularMarketVolume") or quote.get("preMarketVolume"),
    }


def get_yahoo_screen_tickers(screen: str, max_count: int) -> list[str]:
    try:
        data = safe_request_json(YAHOO_SCREENER_URL, {"scrIds": screen, "count": min(max_count, 250)})
        quotes = data["finance"]["result"][0]["quotes"]  # type: ignore[index]
    except Exception:
        return []

    tickers: list[str] = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper().strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        market = str(quote.get("market", "")).lower()
        if quote_type != "EQUITY" or market not in {"us_market", ""} or not is_valid_ticker(symbol):
            continue
        tickers.append(symbol)
        add_quote_from_yahoo(symbol, quote)
    return list(dict.fromkeys(tickers))


def get_trending_tickers(max_count: int) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(YAHOO_TRENDING_URL, headers=headers, timeout=4)
        response.raise_for_status()
        quotes = response.json()["finance"]["result"][0]["quotes"]
    except Exception:
        return []

    tickers: list[str] = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper().strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        if quote_type in {"EQUITY", ""} and is_valid_ticker(symbol):
            tickers.append(symbol)
            add_quote_from_yahoo(symbol, quote)
    return list(dict.fromkeys(tickers))[:max_count]


def get_active_movers(max_count: int) -> tuple[list[str], list[dict[str, object]], dict[str, str], bool]:
    global ACTIVE_QUOTE_CACHE
    sources = [
        ("day_gainers", lambda: get_yahoo_screen_tickers("day_gainers", max_count)),
        ("most_actives", lambda: get_yahoo_screen_tickers("most_actives", max_count)),
        ("premarket_movers", lambda: get_yahoo_screen_tickers("pre_market_gainers", max_count) + get_yahoo_screen_tickers("pre_market_most_actives", max_count)),
        ("trending", lambda: get_trending_tickers(max_count)),
    ]
    tickers: list[str] = []
    source_by_ticker: dict[str, str] = {}
    debug_rows: list[dict[str, object]] = []

    for source_name, fetcher in sources:
        source_tickers = list(dict.fromkeys(fetcher()))
        debug_rows.append({"Source": source_name, "Fetched": len(source_tickers), "Scanned": 0, "Passed Filters": 0})
        for ticker in source_tickers:
            tickers.append(ticker)
            source_by_ticker.setdefault(ticker, source_name)

    tickers = list(dict.fromkeys(tickers))[:max_count]
    unavailable = len(tickers) == 0
    return tickers, debug_rows, source_by_ticker, unavailable


def fetch_security_names(state: AppState) -> dict[str, str]:
    global SECURITY_NAME_CACHE
    now = time.monotonic()
    with state.lock:
        if state.universe_cache and now - state.universe_cache[0] <= 900 and SECURITY_NAME_CACHE:
            return SECURITY_NAME_CACHE

    frames: list[pd.DataFrame] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        nasdaq_text = requests.get(NASDAQ_LISTED_URL, headers=headers, timeout=4).text
        nasdaq = pd.read_csv(StringIO(nasdaq_text), sep="|")
        nasdaq = nasdaq[nasdaq["Symbol"].notna()]
        nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"]
        nasdaq = nasdaq.rename(columns={"Symbol": "Ticker", "Security Name": "Name"})
        frames.append(nasdaq[["Ticker", "Name"]])

    except Exception:
        return SECURITY_NAME_CACHE

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["Name"] = df["Name"].fillna("").astype(str).str.lower()
        SECURITY_NAME_CACHE = dict(zip(df["Ticker"], df["Name"]))
        with state.lock:
            state.universe_cache = (now, df["Ticker"].drop_duplicates().tolist())
    return SECURITY_NAME_CACHE


def build_scan_universe(state: AppState, settings: ScannerSettings) -> tuple[list[str], list[dict[str, object]], dict[str, str], Optional[str]]:
    optional_watchlist = parse_optional_watchlist(settings.optional_watchlist)
    active, debug_rows, source_by_ticker, unavailable = get_active_movers(max(settings.max_tickers, 150))
    message = None

    if unavailable:
        message = "Active movers source unavailable"
        names = fetch_security_names(state)
        fallback = list(names)[: settings.max_tickers] if names else FALLBACK_TICKERS
        active = list(dict.fromkeys(fallback))[: settings.max_tickers]
        source_by_ticker = {ticker: "fallback" for ticker in active}
        debug_rows.append({"Source": "fallback", "Fetched": len(active), "Scanned": 0, "Passed Filters": 0})

    if settings.cloud_fast_mode:
        tickers = list(dict.fromkeys(active + optional_watchlist))[: settings.max_tickers]
    else:
        names = fetch_security_names(state)
        all_tickers = list(names) or FALLBACK_TICKERS
        remaining = [ticker for ticker in all_tickers if ticker not in set(active + optional_watchlist)]
        explore_size = min(len(remaining), max(0, settings.max_tickers - len(active + optional_watchlist)))
        explore = random.sample(remaining, explore_size) if explore_size else []
        tickers = list(dict.fromkeys(active + optional_watchlist + explore))[: settings.max_tickers]
        for ticker in explore:
            source_by_ticker.setdefault(ticker, "fallback")

    if optional_watchlist:
        debug_rows.append({"Source": "optional_watchlist", "Fetched": len(optional_watchlist), "Scanned": 0, "Passed Filters": 0})
        for ticker in optional_watchlist:
            source_by_ticker.setdefault(ticker, "optional_watchlist")

    return tickers, debug_rows, source_by_ticker, message


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def cached_intraday(ticker: str, state: AppState, settings: ScannerSettings) -> pd.DataFrame:
    now = time.monotonic()
    with state.lock:
        cached = state.download_cache.get(ticker)
        if cached and now - cached[0] <= settings.cache_seconds:
            return cached[1].copy()

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                ticker,
                period="1d",
                interval="5m",
                prepost=True,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=settings.request_timeout_seconds,
            )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty or "Close" not in df:
        df = pd.DataFrame()
    else:
        df = normalize_columns(df).dropna(subset=["Close"])

    with state.lock:
        state.download_cache[ticker] = (now, df.copy())
        if len(state.download_cache) > 500:
            state.download_cache = dict(list(state.download_cache.items())[-250:])
    return df


def session_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    eastern = working.index.tz_convert("America/New_York")
    working["session_date"] = eastern.date
    today = working[working["session_date"] == working["session_date"].max()].copy()
    regular = today.between_time("09:30", "16:00")
    return today, regular


def estimate_previous_close(ticker: str, current_price: float, open_price: float) -> tuple[float, str]:
    quote = ACTIVE_QUOTE_CACHE.get(ticker, {})
    previous_close = safe_float(quote.get("previous_close"), 0)
    if previous_close > 0:
        return previous_close, "confirmed previous close"

    quote_price = safe_float(quote.get("price"), 0)
    quote_change_pct = safe_float(quote.get("change_pct"), np.nan)
    if quote_price > 0 and not np.isnan(quote_change_pct) and quote_change_pct > -99:
        return quote_price / (1 + quote_change_pct / 100), "estimated from active-mover quote"

    return open_price if open_price > 0 else current_price, "estimated previous close"


def calc_vwap(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df or float(df["Volume"].sum()) <= 0:
        return np.nan
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((typical * df["Volume"]).sum() / df["Volume"].sum())


def volume_acceleration(df: pd.DataFrame) -> tuple[float, bool]:
    if df.empty or len(df) < 12 or "Volume" not in df:
        return 0.0, False
    last_two = float(df["Volume"].tail(2).mean())
    previous = float(df["Volume"].iloc[:-2].tail(10).mean())
    if previous <= 0:
        return 0.0, False
    ratio = last_two / previous
    return round(ratio, 2), ratio >= 1.5


def relative_volume(volume: int, df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df:
        return 0.0
    bars_seen = max(1, min(len(df), 78))
    recent_bar_avg = float(df["Volume"].tail(20).mean())
    baseline = recent_bar_avg * 78
    if baseline <= 0:
        return 0.0
    projected = volume * 78 / bars_seen
    return round(projected / baseline, 2)


def tight_consolidation(df: pd.DataFrame, current_price: float, day_high: float) -> tuple[float, bool]:
    if df.empty or len(df) < 6 or current_price <= 0:
        return 100.0, False
    recent = df.tail(8)
    range_pct = ((float(recent["High"].max()) - float(recent["Low"].min())) / current_price) * 100
    near_high_pct = ((day_high - current_price) / current_price) * 100
    return round(range_pct, 2), range_pct <= 4.0 and near_high_pct <= 3.0


def has_higher_lows(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < 6:
        return False
    lows = df["Low"].tail(6).reset_index(drop=True)
    return float(lows.iloc[3:].min()) > float(lows.iloc[:3].min()) and float(lows.iloc[-1]) >= float(lows.iloc[-3])


def halt_spike_potential(df: pd.DataFrame, current_price: float) -> bool:
    if df.empty or len(df) < 4 or current_price <= 0:
        return False
    recent = df.tail(4)
    recent_low = float(recent["Low"].min())
    recent_high = max(float(recent["High"].max()), current_price)
    return recent_low > 0 and ((recent_high - recent_low) / recent_low) * 100 >= 12


def keyword_boost(ticker: str) -> bool:
    name = SECURITY_NAME_CACHE.get(ticker.upper(), "")
    terms = ("acquisition", "capital", "holdings", "biotech", "therapeutics", "pharma", "micro")
    return any(term in name for term in terms)


def pre_move_score(
    rvol: float,
    accel: float,
    near_high_pct: float,
    compression_pct: float,
    is_tight: bool,
    higher_lows: bool,
    vwap_reclaim: bool,
    gap_pct: float,
    premarket_strength: bool,
    halt_potential: bool,
    low_float_boost: bool,
) -> float:
    score = 0.0
    score += min(rvol, 8.0) * 9
    score += min(accel, 6.0) * 8
    score += max(0.0, 3.0 - near_high_pct) * 9
    score += max(0.0, 5.0 - compression_pct) * 5
    score += 14 if is_tight else 0
    score += 12 if higher_lows else 0
    score += 12 if vwap_reclaim else 0
    score += min(max(gap_pct, 0), 20) * 1.2
    score += 12 if premarket_strength else 0
    score += 10 if halt_potential else 0
    score += 8 if low_float_boost else 0
    return round(min(score, 100), 1)


def analyze_ticker(ticker: str, state: AppState, settings: ScannerSettings) -> tuple[Optional[dict[str, object]], Optional[dict[str, object]], str]:
    if not is_valid_ticker(ticker):
        return None, None, "missing_data"

    df = cached_intraday(ticker, state, settings)
    if df.empty:
        return None, None, "missing_data"

    today, regular = session_rows(df)
    analysis = regular if not regular.empty else today
    if today.empty or analysis.empty:
        return None, None, "missing_data"

    current_price = safe_float(today["Close"].iloc[-1])
    open_price = safe_float(regular["Open"].iloc[0] if not regular.empty else today["Open"].iloc[0])
    previous_close, data_quality = estimate_previous_close(ticker, current_price, open_price)
    if current_price <= 0 or previous_close <= 0:
        return None, None, "missing_data"

    day_high = safe_float(today["High"].max(), current_price)
    day_low = safe_float(today["Low"].min(), current_price)
    volume = int(safe_float(today["Volume"].sum(), 0))
    gap_pct = ((current_price - previous_close) / previous_close) * 100
    gain_pct = gap_pct
    near_high_pct = ((day_high - current_price) / current_price) * 100 if current_price > 0 else 100.0
    rvol = relative_volume(volume, analysis)
    accel, ignition = volume_acceleration(analysis)
    compression_pct, is_tight = tight_consolidation(analysis, current_price, day_high)
    higher_lows = has_higher_lows(analysis)
    vwap = calc_vwap(analysis)
    above_vwap = not np.isnan(vwap) and current_price >= vwap
    vwap_reclaim = above_vwap and day_low <= vwap
    premarket_strength = gap_pct >= 3 or current_price >= open_price
    halt_potential = halt_spike_potential(analysis, current_price)
    low_float_boost = keyword_boost(ticker)

    watched = {
        "Ticker": ticker,
        "Current Price": round(current_price, 4),
        "Gap %": round(gap_pct, 2),
        "RVOL": rvol,
        "Volume": volume,
        "Near High %": round(near_high_pct, 2),
        "Volume Accel": accel,
        "Data Quality": data_quality,
        "Last Updated": datetime.now().isoformat(timespec="seconds"),
    }

    if current_price < settings.min_price or current_price > settings.max_price:
        return None, watched, "price_filter"

    score = pre_move_score(
        rvol,
        accel,
        near_high_pct,
        compression_pct,
        is_tight,
        higher_lows,
        vwap_reclaim,
        gap_pct,
        premarket_strength,
        halt_potential,
        low_float_boost,
    )

    reason_parts: list[str] = []
    if ignition:
        reason_parts.append("volume ignition")
    if rvol >= 1.3:
        reason_parts.append(f"RVOL {rvol:.2f}x")
    if near_high_pct <= settings.near_high_threshold_pct:
        reason_parts.append("near day high")
    if is_tight:
        reason_parts.append("tight consolidation")
    if higher_lows:
        reason_parts.append("higher lows")
    if vwap_reclaim:
        reason_parts.append("VWAP reclaim")
    if gap_pct >= 3:
        reason_parts.append("gap up")
    if halt_potential:
        reason_parts.append("halt/spike potential")
    if low_float_boost:
        reason_parts.append("low-float keyword boost")

    has_pressure = score >= 35 or near_high_pct <= 3 or ignition or is_tight or vwap_reclaim
    if not has_pressure:
        return None, watched, "no_momentum"

    support_low = round_cent(max(day_low, current_price * 0.965))
    trigger = round_cent(max(day_high + 0.02, current_price * 1.006))
    stop = round_cent(max(0.01, min(support_low, current_price * 0.97)))
    risk = max(trigger - stop, 0)
    target_1 = round_cent(trigger + risk * 2) if risk > 0 else round_cent(current_price * 1.08)
    target_2 = round_cent(trigger + risk * 3) if risk > 0 else round_cent(current_price * 1.14)
    rr = ((target_1 - trigger) / risk) if risk > 0 else 0.0
    trigger_distance_pct = abs(trigger - current_price) / current_price * 100 if current_price > 0 else 999.0
    extended_pct = ((current_price - support_low) / support_low) * 100 if support_low > 0 else 999.0
    below_open_or_vwap = current_price < open_price or (not np.isnan(vwap) and current_price < vwap)
    data_stale = trigger_distance_pct > 25 or data_quality.startswith("estimated") and abs(gap_pct) > 80

    if data_stale:
        status = "INVALID DATA"
        setup_type = "INVALID DATA"
    elif gain_pct < -2 or below_open_or_vwap:
        status = "WAIT FOR REVERSAL"
        setup_type = "WAIT FOR REVERSAL"
    elif extended_pct > 12 and gain_pct > 12:
        status = "WAIT FOR PULLBACK"
        setup_type = "WAIT FOR PULLBACK"
    elif ignition and score >= 55:
        status = "BREAKOUT IMMINENT"
        setup_type = "⚡ VOLUME IGNITION"
    elif score >= 45 and near_high_pct <= 3:
        status = "PRE-MOVE WATCH"
        setup_type = "BREAKOUT IMMINENT"
    elif score >= 35:
        status = "WAIT FOR TRIGGER"
        setup_type = "PRE-MOVE WATCH"
    else:
        status = "❌ NO LONG"
        setup_type = "❌ NO LONG"

    if status in {"BREAKOUT IMMINENT", "PRE-MOVE WATCH"} and trigger_distance_pct <= 4 and rvol >= 1.3 and rr >= 2:
        status = "VALID TRADE"
    elif status == "VALID TRADE":
        status = "WAIT FOR TRIGGER"

    if status in {"❌ NO LONG", "INVALID DATA"}:
        entry = stop_out = target_1_out = target_2_out = None
    else:
        entry = trigger
        stop_out = stop
        target_1_out = target_1
        target_2_out = target_2

    confirmation = f"Trigger only if price breaks {trigger:.2f} with RVOL >= 1.3 and holds above VWAP."
    invalidation = f"Invalid below {stop:.2f} or if price loses VWAP/open and volume fades."
    why = ", ".join(reason_parts) if reason_parts else "early pressure detected before a confirmed breakout"
    avoid = (
        "Avoid if already extended more than 12% from support, price rejects the trigger, "
        "or RVOL drops below 1.0 before entry."
    )

    row = {
        "Ticker": ticker,
        "Current Price": round(current_price, 4),
        "Pre-Move Score": score,
        "Status": status,
        "Setup Type": setup_type,
        "Trigger Entry": entry,
        "Stop": stop_out,
        "Target 1": target_1_out,
        "Target 2": target_2_out,
        "Risk/Reward": "N/A" if rr <= 0 else f"1:{rr:.2f}",
        "RVOL": rvol,
        "Volume Accel": accel,
        "Near High %": round(near_high_pct, 2),
        "Gap %": round(gap_pct, 2),
        "Volume": volume,
        "Data Quality": data_quality,
        "Confirmation": confirmation,
        "Invalidation": invalidation,
        "Why This May Run": why,
        "Avoid Reason": avoid,
        "VWAP": round(vwap, 4) if not np.isnan(vwap) else np.nan,
        "Last Updated": datetime.now().isoformat(timespec="seconds"),
    }
    watched["Status"] = status
    watched["Pre-Move Score"] = score
    return row, watched, ""


def rank_candidates(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"], keep="last")
    status_rank = {
        "VALID TRADE": 0,
        "BREAKOUT IMMINENT": 1,
        "PRE-MOVE WATCH": 2,
        "WAIT FOR TRIGGER": 3,
        "WAIT FOR PULLBACK": 4,
        "WAIT FOR REVERSAL": 5,
        "❌ NO LONG": 6,
        "INVALID DATA": 7,
    }
    df["_status_rank"] = df["Status"].map(status_rank).fillna(9)
    return df.sort_values(["_status_rank", "Pre-Move Score", "RVOL"], ascending=[True, False, False]).drop(columns=["_status_rank"])


def rank_watched(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"], keep="last")
    sort_cols = [col for col in ["Pre-Move Score", "Gap %", "RVOL", "Volume"] if col in df.columns]
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(150) if sort_cols else df.head(150)


def update_source_debug(
    source_debug_rows: list[dict[str, object]],
    scanned_by_source: dict[str, int],
    passed_by_source: dict[str, int],
) -> list[dict[str, object]]:
    rows = [dict(row) for row in source_debug_rows]
    known_sources = {str(row.get("Source")) for row in rows}
    for source in set(scanned_by_source) | set(passed_by_source):
        if source not in known_sources:
            rows.append({"Source": source, "Fetched": 0, "Scanned": 0, "Passed Filters": 0})
    for row in rows:
        source = str(row.get("Source"))
        row["Scanned"] = int(scanned_by_source.get(source, 0))
        row["Passed Filters"] = int(passed_by_source.get(source, 0))
    return rows


def publish(
    state: AppState,
    candidate_rows: list[dict[str, object]],
    watched_rows: list[dict[str, object]],
    skip_counts: dict[str, int],
    source_debug_rows: list[dict[str, object]],
    scanned: int,
    started: float,
    status_text: str,
    timed_out: bool = False,
    source_message: Optional[str] = None,
) -> None:
    candidates = rank_candidates(candidate_rows).head(state.settings.output_limit)
    watched = rank_watched(watched_rows)
    with state.lock:
        state.candidates_df = candidates
        state.watched_df = watched
        state.source_debug_df = pd.DataFrame(source_debug_rows)
        state.skip_counts = dict(skip_counts)
        state.scanned_count = scanned
        state.candidate_count = len(candidates)
        state.last_scan_seconds = time.monotonic() - started
        state.timed_out = timed_out
        state.status_text = status_text
        state.source_message = source_message


def scan_once(state: AppState, settings: ScannerSettings) -> None:
    started = time.monotonic()
    deadline = started + settings.scan_timeout_seconds
    tickers, source_debug_rows, source_by_ticker, source_message = build_scan_universe(state, settings)
    candidate_rows: list[dict[str, object]] = []
    watched_rows: list[dict[str, object]] = []
    skip_counts = {key: 0 for key in SKIP_KEYS}
    scanned_by_source: dict[str, int] = {}
    passed_by_source: dict[str, int] = {}
    scanned = 0

    with state.lock:
        state.last_scan_started = datetime.now().isoformat(timespec="seconds")
        state.status_text = f"Scanning {len(tickers)} active symbols..."
        state.source_message = source_message
        state.error = None
        state.timed_out = False

    executor = ThreadPoolExecutor(max_workers=max(1, min(settings.max_workers, 12)))
    try:
        pending = {executor.submit(analyze_ticker, ticker, state, settings): ticker for ticker in tickers}
        while pending and not state.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending_set = wait(set(pending), timeout=min(0.5, remaining), return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                ticker = pending.pop(future, "")
                scanned += 1
                source_name = source_by_ticker.get(ticker, "fallback")
                scanned_by_source[source_name] = scanned_by_source.get(source_name, 0) + 1
                try:
                    candidate, watched, skip_reason = future.result(timeout=0)
                except Exception:
                    candidate, watched, skip_reason = None, None, "missing_data"
                if watched is not None:
                    watched["Source"] = source_name
                    watched_rows.append(watched)
                if candidate is not None:
                    candidate["Source"] = source_name
                    candidate_rows.append(candidate)
                    passed_by_source[source_name] = passed_by_source.get(source_name, 0) + 1
                elif skip_reason in skip_counts:
                    skip_counts[skip_reason] += 1
                status = f"Scanning active movers: {scanned}/{len(tickers)} complete"
                current_debug = update_source_debug(source_debug_rows, scanned_by_source, passed_by_source)
                publish(state, candidate_rows, watched_rows, skip_counts, current_debug, scanned, started, status, source_message=source_message)
            pending = {future: pending[future] for future in pending_set}

        timed_out = bool(pending)
        for future in pending:
            future.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    status_text = "Scan timeout reached; showing partial results." if timed_out else "Scan complete."
    final_debug = update_source_debug(source_debug_rows, scanned_by_source, passed_by_source)
    publish(state, candidate_rows, watched_rows, skip_counts, final_debug, scanned, started, status_text, timed_out, source_message)
    with state.lock:
        state.last_scan_finished = datetime.now().isoformat(timespec="seconds")
        state.cycle_count += 1


def scanner_loop(state: AppState) -> None:
    while not state.stop_event.is_set():
        with state.lock:
            settings = state.settings
        try:
            scan_once(state, settings)
        except Exception as exc:
            with state.lock:
                state.error = f"{type(exc).__name__}: {exc}"
                state.status_text = "Scanner error. Check details below."

        sleep_until = time.monotonic() + max(5, settings.scan_interval_seconds)
        while time.monotonic() < sleep_until and not state.stop_event.is_set():
            time.sleep(0.25)


def restart_scanner(state: AppState, settings: ScannerSettings, clear_cache: bool = False) -> None:
    with state.lock:
        old_thread = state.scanner_thread
        state.stop_event.set()
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=1.5)

    with state.lock:
        state.version = APP_VERSION
        state.stop_event = threading.Event()
        state.settings = settings
        state.candidates_df = pd.DataFrame()
        state.watched_df = pd.DataFrame()
        state.source_debug_df = pd.DataFrame()
        state.skip_counts = {key: 0 for key in SKIP_KEYS}
        state.scanned_count = 0
        state.candidate_count = 0
        state.last_scan_seconds = 0.0
        state.timed_out = False
        state.error = None
        state.source_message = None
        state.status_text = "Restarting scanner..."
        if clear_cache:
            state.download_cache = {}
            state.universe_cache = None
        thread = threading.Thread(target=scanner_loop, args=(state,), daemon=True, name="pre-move-scanner")
        state.scanner_thread = thread
        thread.start()


def ensure_scanner(state: AppState) -> None:
    with state.lock:
        stale_version = state.version != APP_VERSION
        alive = bool(state.scanner_thread and state.scanner_thread.is_alive())
        settings = state.settings
    if stale_version:
        restart_scanner(state, ScannerSettings(), clear_cache=True)
    elif not alive:
        restart_scanner(state, settings)


def snapshot(state: AppState) -> dict[str, object]:
    with state.lock:
        return {
            "settings": state.settings,
            "candidates": state.candidates_df.copy(),
            "watched": state.watched_df.copy(),
            "source_debug": state.source_debug_df.copy(),
            "skip_counts": dict(state.skip_counts),
            "cycle": state.cycle_count,
            "scanned": state.scanned_count,
            "candidate_count": state.candidate_count,
            "last_scan_seconds": state.last_scan_seconds,
            "last_scan_started": state.last_scan_started,
            "last_scan_finished": state.last_scan_finished,
            "timed_out": state.timed_out,
            "status_text": state.status_text,
            "source_message": state.source_message,
            "error": state.error,
            "thread_alive": bool(state.scanner_thread and state.scanner_thread.is_alive()),
        }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --panel: #0d1422;
                --panel-2: #111a2b;
                --line: #223047;
                --muted: #94a3b8;
                --text: #e5e7eb;
                --green: #22c55e;
                --orange: #f59e0b;
                --red: #ef4444;
            }
            .stApp {
                background:
                    radial-gradient(circle at 20% 0%, rgba(34, 197, 94, 0.08), transparent 28rem),
                    linear-gradient(180deg, #060a11 0%, #0a101b 100%);
                color: var(--text);
            }
            [data-testid="stSidebar"] {
                background: #09111f;
                border-right: 1px solid #1f2937;
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
                max-width: 1680px;
            }
            h1, h2, h3 {
                color: #f8fafc;
                letter-spacing: 0;
            }
            h1 { font-size: clamp(1.7rem, 3vw, 2.6rem); }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
                min-height: 96px;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
            }
            .stButton button {
                border-radius: 16px;
                border: 1px solid #334155;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 16px;
                border-color: var(--line);
                background: linear-gradient(180deg, rgba(17, 26, 43, 0.96), rgba(11, 18, 32, 0.96));
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.25);
            }
            .pm-card-title {
                font-size: clamp(1.6rem, 4vw, 2.2rem);
                line-height: 1;
                font-weight: 800;
                color: #f8fafc;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .pm-card-subtle {
                color: var(--muted);
                font-size: 0.86rem;
                margin-top: -0.25rem;
            }
            .pm-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin-top: 0.45rem;
            }
            .pm-chip {
                display: inline-flex;
                align-items: center;
                border: 1px solid #334155;
                border-radius: 999px;
                padding: 0.2rem 0.55rem;
                color: #dbeafe;
                background: rgba(30, 41, 59, 0.7);
                font-size: 0.78rem;
                white-space: nowrap;
            }
            .pm-mobile-cards { display: none; }
            [data-testid="stDataFrame"] div,
            [data-testid="stDataFrame"] span {
                white-space: nowrap;
            }
            @media (max-width: 760px) {
                .block-container {
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                }
                div[data-testid="stMetric"] {
                    min-height: 82px;
                    padding: 0.8rem;
                }
                .pm-desktop-table,
                div[data-testid="stDataFrame"] {
                    display: none;
                }
                .pm-mobile-cards {
                    display: block;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_priority_card(row: pd.Series) -> None:
    with st.container(border=True):
        ticker = str(row.get("Ticker", "N/A"))
        status = str(row.get("Status", "WAIT FOR TRIGGER"))
        score = safe_float(row.get("Pre-Move Score"), 0)
        reason_text = str(row.get("Why This May Run", ""))
        chips = []
        for label, needle in [
            ("near high", "near day high"),
            ("gap up", "gap up"),
            ("volume spike", "volume ignition"),
            ("VWAP reclaim", "VWAP reclaim"),
            ("higher lows", "higher lows"),
        ]:
            if needle.lower() in reason_text.lower():
                chips.append(label)
        chips = chips[:3] or ["watching pressure"]

        head = st.columns([0.52, 0.48])
        head[0].markdown(f"<div class='pm-card-title'>{ticker}</div>", unsafe_allow_html=True)
        head[1].markdown(status_badge(status))
        st.caption(str(row.get("Setup Type", "PRE-MOVE WATCH")))

        st.progress(min(max(score / 100, 0), 1), text=f"Pre-Move Score {fmt_score(score)}")

        grid_top = st.columns(2)
        grid_top[0].metric("Price", fmt_currency(row.get("Current Price")))
        grid_top[1].metric("Trigger", fmt_currency(row.get("Trigger Entry")))
        grid_bottom = st.columns(2)
        grid_bottom[0].metric("Stop", fmt_currency(row.get("Stop")))
        grid_bottom[1].metric("Target", fmt_currency(row.get("Target 1")))

        chip_html = "".join(f"<span class='pm-chip'>{chip}</span>" for chip in chips)
        st.markdown(f"<div class='pm-chip-row'>{chip_html}</div>", unsafe_allow_html=True)
        st.caption(reason_text)


def compact_reason(value: object) -> str:
    text = str(value or "")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ", ".join(parts[:3]) if parts else "pressure building"


def make_compact_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        rows.append(
            {
                "Ticker": str(row.get("Ticker", "")),
                "Status": str(row.get("Status", "")),
                "Score": fmt_score(row.get("Pre-Move Score")),
                "Price": fmt_currency(row.get("Current Price")),
                "Trigger": fmt_currency(row.get("Trigger Entry")),
                "Stop": fmt_currency(row.get("Stop")),
                "Target 1": fmt_currency(row.get("Target 1")),
                "RVOL": fmt_rvol(row.get("RVOL")),
                "Gap %": fmt_pct(row.get("Gap %")),
                "Reason": compact_reason(row.get("Why This May Run")),
            }
        )
    return pd.DataFrame(rows)


def render_dashboard(snap: dict[str, object]) -> None:
    candidates: pd.DataFrame = snap["candidates"]  # type: ignore[assignment]
    watched: pd.DataFrame = snap["watched"]  # type: ignore[assignment]
    source_debug: pd.DataFrame = snap["source_debug"]  # type: ignore[assignment]
    skip_counts: dict[str, int] = snap["skip_counts"]  # type: ignore[assignment]

    metrics = st.columns(5)
    metrics[0].metric("Scanner Status", "RUNNING" if snap["thread_alive"] else "STOPPED")
    metrics[1].metric("Cycle", f"{int(snap['cycle']):,}")
    metrics[2].metric("Scanned", f"{int(snap['scanned']):,}")
    metrics[3].metric("Candidates", f"{len(candidates):,}")
    metrics[4].metric("Last Scan", f"{safe_float(snap['last_scan_seconds'], 0):.1f}s")

    st.caption(
        f"{snap['status_text']} | Last updated: {datetime.now().isoformat(timespec='seconds')} | "
        f"Last finished: {snap['last_scan_finished'] or 'Waiting...'}"
    )
    if snap["timed_out"]:
        st.warning("20 second scan timeout reached. Partial results are displayed.")
    if snap["source_message"]:
        st.warning(str(snap["source_message"]))
    if snap["error"]:
        st.error(str(snap["error"]))

    st.subheader("High Priority Pre-Move")
    top_priority = candidates.head(3)
    if top_priority.empty:
        st.info("No filtered pre-move candidates yet. Showing watched movers below.")
    else:
        cols = st.columns(min(3, len(top_priority)), gap="large")
        for index, (_, row) in enumerate(top_priority.iterrows()):
            with cols[index % len(cols)]:
                render_priority_card(row)

    st.subheader("Pre-Move Scanner Table")
    if not candidates.empty:
        table = make_compact_table(candidates)
        st.markdown("<div class='pm-desktop-table'>", unsafe_allow_html=True)
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Trigger": st.column_config.TextColumn("Trigger", width="small"),
                "Stop": st.column_config.TextColumn("Stop", width="small"),
                "Target 1": st.column_config.TextColumn("Target 1", width="small"),
                "RVOL": st.column_config.TextColumn("RVOL", width="small"),
                "Gap %": st.column_config.TextColumn("Gap %", width="small"),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Extra trade-plan details", expanded=False):
            detail_cols = [
                "Ticker",
                "Setup Type",
                "Target 2",
                "Risk/Reward",
                "Volume Accel",
                "Near High %",
                "Confirmation",
                "Invalidation",
                "Why This May Run",
                "Avoid Reason",
            ]
            details = candidates[[col for col in detail_cols if col in candidates.columns]].copy()
            if not details.empty:
                st.dataframe(details, use_container_width=True, hide_index=True)
    elif not watched.empty:
        st.dataframe(watched, use_container_width=True, hide_index=True)
    else:
        top_skip = max(SKIP_KEYS, key=lambda key: skip_counts.get(key, 0))
        st.warning(f"No symbols returned usable data yet. Largest skip bucket: {top_skip} ({skip_counts.get(top_skip, 0):,}).")

    with st.expander("Watched movers and scanner diagnostics", expanded=False):
        if not source_debug.empty:
            st.dataframe(source_debug, use_container_width=True, hide_index=True)
        if not watched.empty:
            st.dataframe(watched, use_container_width=True, hide_index=True)
        st.dataframe(
            pd.DataFrame([{"Reason": key.replace("_", " "), "Count": skip_counts.get(key, 0)} for key in SKIP_KEYS]),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Pre-Move Momentum Scanner", layout="wide")
    apply_theme()
    state = get_state()
    ensure_scanner(state)
    snap = snapshot(state)
    settings: ScannerSettings = snap["settings"]  # type: ignore[assignment]

    st.title("Pre-Move Momentum Scanner")
    st.caption("Scanner discovers tickers automatically. Optional watchlist is only used to force-check names.")

    with st.sidebar:
        st.header("Scanner Controls")
        cloud_fast_mode = st.toggle("Cloud Fast Mode", value=settings.cloud_fast_mode)
        local_full_mode = st.toggle("Local Full Mode", value=not settings.cloud_fast_mode)
        if local_full_mode:
            cloud_fast_mode = False

        max_tickers_max = 150 if cloud_fast_mode else 1000
        max_tickers = st.slider("Max tickers", 50, max_tickers_max, min(settings.max_tickers, max_tickers_max), step=10)
        workers = st.slider("Workers", 1, 12, min(settings.max_workers, 12), step=1)
        scan_interval = st.slider("Scan interval seconds", 15, 300, settings.scan_interval_seconds, step=5)
        scan_timeout = st.slider("Scan timeout seconds", 5, 30, settings.scan_timeout_seconds, step=1)
        min_volume = st.number_input("Minimum volume", min_value=0, value=settings.min_volume, step=25_000)
        max_price = st.slider("Maximum price", 1.0, 200.0, float(settings.max_price), step=1.0)
        min_rvol = st.slider("Minimum RVOL pressure", 0.0, 3.0, float(settings.min_relative_volume), step=0.1)
        with st.expander("Advanced", expanded=False):
            optional_watchlist = st.text_area("Optional advanced watchlist", value=settings.optional_watchlist, height=80)
            st.caption("Use this only to force-check names. The scanner discovers tickers automatically.")

        new_settings = ScannerSettings(
            cloud_fast_mode=cloud_fast_mode,
            scan_interval_seconds=int(scan_interval),
            scan_timeout_seconds=int(scan_timeout),
            max_tickers=int(max_tickers),
            max_workers=int(workers),
            max_price=float(max_price),
            min_volume=int(min_volume),
            min_relative_volume=float(min_rvol),
            optional_watchlist=optional_watchlist,
        )

        if st.button("Run scan now", type="primary", use_container_width=True):
            restart_scanner(state, new_settings)
            st.rerun()
        if st.button("Clear cache / Reset scanner", use_container_width=True):
            restart_scanner(state, ScannerSettings(), clear_cache=True)
            st.rerun()

        with state.lock:
            state.settings = new_settings

    render_dashboard(snapshot(state))


if __name__ == "__main__":
    main()
