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

APP_VERSION = "small-cap-explosive-runner-scanner-2026-04-30"
YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

SMALL_CAP_RUNNER_FALLBACK = [
    "HCAI",
    "MRAM",
    "AKAN",
    "MXL",
    "RDAC",
    "BIYA",
    "SBLX",
    "ATER",
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

SKIP_KEYS = ["missing_data", "price_filter", "no_setup", "invalid_data"]
SOURCE_NAMES = ["day_gainers", "most_actives", "trending", "premarket_movers", "nasdaq_fallback", "runner_fallback", "optional_watchlist"]

ACTIVE_QUOTE_CACHE: dict[str, dict[str, object]] = {}
SECURITY_NAME_CACHE: dict[str, str] = {}


@dataclass
class ScannerSettings:
    cloud_fast_mode: bool = True
    scan_interval_seconds: int = 60
    scan_timeout_seconds: int = 20
    max_tickers: int = 150
    max_workers: int = 8
    min_price: float = 0.2
    max_price: float = 200.0
    min_volume: int = 0
    min_rvol_for_valid: float = 1.3
    optional_watchlist: str = ""
    request_timeout_seconds: int = 3
    cache_seconds: int = 20


@dataclass
class AppState:
    version: str = APP_VERSION
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    scanner_thread: Optional[threading.Thread] = None
    settings: ScannerSettings = field(default_factory=ScannerSettings)
    explosive_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    premove_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    watched_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_debug_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skip_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in SKIP_KEYS})
    cycle_count: int = 0
    scanned_count: int = 0
    last_scan_started: Optional[str] = None
    last_scan_finished: Optional[str] = None
    last_scan_seconds: float = 0.0
    timed_out: bool = False
    status_text: str = "Starting scanner..."
    source_message: Optional[str] = None
    error: Optional[str] = None
    intraday_cache: dict[str, tuple[float, pd.DataFrame]] = field(default_factory=dict)
    daily_cache: dict[str, tuple[float, pd.DataFrame]] = field(default_factory=dict)
    universe_cache: tuple[float, list[str]] | None = None


@st.cache_resource(show_spinner=False)
def get_state() -> AppState:
    return AppState()


def is_valid_ticker(ticker: str) -> bool:
    return bool(VALID_TICKER_RE.fullmatch(str(ticker).upper().strip()))


def parse_tickers(value: str) -> list[str]:
    tickers = [part.upper().strip() for part in re.split(r"[\s,;]+", value or "")]
    return [ticker for ticker in dict.fromkeys(tickers) if is_valid_ticker(ticker)]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number


def fmt_currency(value: object) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"${number:.2f}"


def fmt_pct(value: object) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.2f}%"


def fmt_rvol(value: object) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.2f}"


def fmt_score(value: object) -> str:
    return f"{safe_float(value, 0):.0f}/100"


def fmt_num(value: object) -> str:
    return f"{int(safe_float(value, 0)):,}"


def round_cent(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def status_badge_text(status: str) -> str:
    if status == "VALID TRADE":
        return f":green[{status}]"
    if status in {"INVALID DATA", "NO LONG"}:
        return f":red[{status}]"
    if status.startswith("HOT RUNNER"):
        return f":red[{status}]"
    if status == "BREAKOUT IMMINENT":
        return f":violet[{status}]"
    return f":orange[{status}]"


def request_json(url: str, params: Optional[dict[str, object]] = None, timeout: int = 4) -> dict[str, object]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def cache_quote(symbol: str, quote: dict[str, object]) -> None:
    ACTIVE_QUOTE_CACHE[symbol] = {
        "price": quote.get("regularMarketPrice") or quote.get("preMarketPrice") or quote.get("postMarketPrice"),
        "previous_close": quote.get("regularMarketPreviousClose"),
        "change_pct": quote.get("regularMarketChangePercent") or quote.get("preMarketChangePercent"),
        "volume": quote.get("regularMarketVolume") or quote.get("preMarketVolume"),
    }


def yahoo_screen(source_name: str, screen_id: str, max_count: int) -> tuple[str, list[str]]:
    try:
        data = request_json(YAHOO_SCREENER_URL, {"scrIds": screen_id, "count": min(max_count, 250)})
        quotes = data["finance"]["result"][0]["quotes"]  # type: ignore[index]
    except Exception:
        return source_name, []

    tickers: list[str] = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper().strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        market = str(quote.get("market", "")).lower()
        if quote_type == "EQUITY" and market in {"us_market", ""} and is_valid_ticker(symbol):
            tickers.append(symbol)
            cache_quote(symbol, quote)
    return source_name, list(dict.fromkeys(tickers))


def yahoo_trending(max_count: int) -> tuple[str, list[str]]:
    try:
        data = request_json(YAHOO_TRENDING_URL)
        quotes = data["finance"]["result"][0]["quotes"]  # type: ignore[index]
    except Exception:
        return "trending", []

    tickers: list[str] = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper().strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        if quote_type in {"", "EQUITY"} and is_valid_ticker(symbol):
            tickers.append(symbol)
            cache_quote(symbol, quote)
    return "trending", list(dict.fromkeys(tickers))[:max_count]


def nasdaq_fallback(state: AppState, max_count: int) -> list[str]:
    global SECURITY_NAME_CACHE
    now = time.monotonic()
    with state.lock:
        if state.universe_cache and now - state.universe_cache[0] <= 900:
            return state.universe_cache[1][:max_count]

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        text = requests.get(NASDAQ_LISTED_URL, headers=headers, timeout=4).text
        df = pd.read_csv(StringIO(text), sep="|")
        df = df[df["Symbol"].notna()]
        df = df[df["Symbol"] != "File Creation Time"]
        df = df.rename(columns={"Symbol": "Ticker", "Security Name": "Name"})
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["Name"] = df["Name"].fillna("").astype(str).str.lower()
        df = df[[is_valid_ticker(ticker) for ticker in df["Ticker"]]]
        if "ETF" in df:
            df = df[df["ETF"].astype(str).str.upper().ne("Y")]
        if "Test Issue" in df:
            df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]
        SECURITY_NAME_CACHE = dict(zip(df["Ticker"], df["Name"]))
        tickers = df["Ticker"].drop_duplicates().tolist()
    except Exception:
        tickers = SMALL_CAP_RUNNER_FALLBACK

    with state.lock:
        state.universe_cache = (now, tickers)
    return tickers[:max_count]


def build_universe(state: AppState, settings: ScannerSettings) -> tuple[list[str], list[dict[str, object]], dict[str, str], Optional[str]]:
    source_results = [
        yahoo_screen("day_gainers", "day_gainers", settings.max_tickers),
        yahoo_screen("most_actives", "most_actives", settings.max_tickers),
        yahoo_trending(settings.max_tickers),
        yahoo_screen("premarket_movers", "pre_market_gainers", settings.max_tickers),
        yahoo_screen("premarket_movers", "pre_market_most_actives", settings.max_tickers),
    ]

    tickers: list[str] = []
    source_by_ticker: dict[str, str] = {}
    debug: dict[str, dict[str, object]] = {
        source: {"Source": source, "Fetched": 0, "Scanned": 0, "Passed Filters": 0} for source in SOURCE_NAMES
    }

    for source, source_tickers in source_results:
        unique_source_tickers = list(dict.fromkeys(source_tickers))
        debug[source]["Fetched"] = int(debug[source]["Fetched"]) + len(unique_source_tickers)
        for ticker in unique_source_tickers:
            tickers.append(ticker)
            source_by_ticker.setdefault(ticker, source)

    optional = parse_tickers(settings.optional_watchlist)
    if optional:
        debug["optional_watchlist"]["Fetched"] = len(optional)
        for ticker in optional:
            tickers.append(ticker)
            source_by_ticker.setdefault(ticker, "optional_watchlist")

    source_message = None
    if not tickers:
        source_message = "Active movers source unavailable"
        fallback = nasdaq_fallback(state, settings.max_tickers)
        debug["nasdaq_fallback"]["Fetched"] = len(fallback)
        for ticker in fallback:
            tickers.append(ticker)
            source_by_ticker.setdefault(ticker, "nasdaq_fallback")

    for ticker in SMALL_CAP_RUNNER_FALLBACK:
        tickers.append(ticker)
        source_by_ticker.setdefault(ticker, "runner_fallback")
    debug["runner_fallback"]["Fetched"] = len(SMALL_CAP_RUNNER_FALLBACK)

    if not settings.cloud_fast_mode:
        fallback = nasdaq_fallback(state, settings.max_tickers)
        for ticker in fallback:
            tickers.append(ticker)
            source_by_ticker.setdefault(ticker, "nasdaq_fallback")
        debug["nasdaq_fallback"]["Fetched"] = max(int(debug["nasdaq_fallback"]["Fetched"]), len(fallback))

    final_tickers = list(dict.fromkeys(tickers))[: settings.max_tickers]
    debug_rows = [row for row in debug.values() if int(row["Fetched"]) > 0 or row["Source"] in {"day_gainers", "most_actives", "trending", "premarket_movers"}]
    return final_tickers, debug_rows, source_by_ticker, source_message


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def cached_download(
    ticker: str,
    state: AppState,
    settings: ScannerSettings,
    period: str,
    interval: str,
    prepost: bool,
) -> pd.DataFrame:
    key = f"{ticker}:{period}:{interval}:{prepost}"
    now = time.monotonic()
    cache = state.daily_cache if interval == "1d" else state.intraday_cache
    ttl = 900 if interval == "1d" else settings.cache_seconds

    with state.lock:
        cached = cache.get(key)
        if cached and now - cached[0] <= ttl:
            return cached[1].copy()

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                prepost=prepost,
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
        cache[key] = (now, df.copy())
        if len(cache) > 600:
            cache.clear()
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
    change_pct = safe_float(quote.get("change_pct"), np.nan)
    if quote_price > 0 and not np.isnan(change_pct) and change_pct > -99:
        return quote_price / (1 + change_pct / 100), "estimated from active source"
    return open_price if open_price > 0 else current_price, "estimated previous close"


def calc_vwap(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df or float(df["Volume"].sum()) <= 0:
        return np.nan
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((typical * df["Volume"]).sum() / df["Volume"].sum())


def calc_relative_volume(volume: int, df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df:
        return 0.0
    bars_seen = max(1, min(len(df), 78))
    recent_bar_avg = float(df["Volume"].tail(20).mean())
    baseline = recent_bar_avg * 78
    if baseline <= 0:
        return 0.0
    projected = volume * 78 / bars_seen
    return round(projected / baseline, 2)


def calc_volume_acceleration(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 12 or "Volume" not in df:
        return 0.0
    last_two = float(df["Volume"].tail(2).mean())
    previous_ten = float(df["Volume"].iloc[:-2].tail(10).mean())
    return round(last_two / previous_ten, 2) if previous_ten > 0 else 0.0


def last_two_candle_move(df: pd.DataFrame, current_price: float) -> float:
    if df.empty or len(df) < 3 or current_price <= 0:
        return 0.0
    base = safe_float(df["Close"].iloc[-3], 0)
    return ((current_price - base) / base) * 100 if base > 0 else 0.0


def tight_consolidation(df: pd.DataFrame, current_price: float, day_high: float) -> tuple[float, bool]:
    if df.empty or len(df) < 6 or current_price <= 0:
        return 100.0, False
    recent = df.tail(8)
    range_pct = ((safe_float(recent["High"].max()) - safe_float(recent["Low"].min())) / current_price) * 100
    near_high_pct = ((day_high - current_price) / current_price) * 100
    return round(range_pct, 2), range_pct <= 4 and near_high_pct <= 3


def higher_lows(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < 6:
        return False
    lows = df["Low"].tail(6).reset_index(drop=True)
    return safe_float(lows.iloc[3:].min()) > safe_float(lows.iloc[:3].min()) and safe_float(lows.iloc[-1]) >= safe_float(lows.iloc[-3])


def breakout_20d(ticker: str, current_price: float, state: AppState, settings: ScannerSettings) -> bool:
    daily = cached_download(ticker, state, settings, "1mo", "1d", False)
    if daily.empty or len(daily) < 10:
        return False
    prior = daily.iloc[:-1].tail(20) if len(daily) > 1 else daily.tail(20)
    if prior.empty or "High" not in prior:
        return False
    return current_price >= safe_float(prior["High"].max(), current_price * 2)


def explosion_score(gain_pct: float, rvol: float, accel: float, near_high_pct: float, volume: int) -> float:
    score = 0.0
    score += min(max(gain_pct, 0), 40) * 1.25
    score += min(max(rvol, 0), 8) * 8
    score += min(max(accel, 0), 8) * 7
    score += max(0, 4 - near_high_pct) * 8
    score += min(volume / 1_000_000, 10) * 2.5
    return round(min(score, 100), 1)


def premove_score(
    is_tight: bool,
    has_higher_lows: bool,
    near_high_pct: float,
    vwap_reclaim: bool,
    volume_ignition: bool,
    gap_pct: float,
    breakout_20: bool,
) -> float:
    score = 0.0
    score += 22 if is_tight else 0
    score += 18 if has_higher_lows else 0
    score += max(0, 3 - near_high_pct) * 10
    score += 18 if vwap_reclaim else 0
    score += 16 if volume_ignition else 0
    score += min(max(gap_pct, 0), 12) * 1.2
    score += 12 if breakout_20 else 0
    return round(min(score, 100), 1)


def analyze_ticker(ticker: str, state: AppState, settings: ScannerSettings) -> tuple[Optional[dict[str, object]], Optional[dict[str, object]], str]:
    intraday = cached_download(ticker, state, settings, "1d", "5m", True)
    if intraday.empty:
        return None, None, "missing_data"

    today, regular = session_rows(intraday)
    analysis = regular if not regular.empty else today
    if today.empty or analysis.empty:
        return None, None, "missing_data"

    current_price = safe_float(today["Close"].iloc[-1])
    open_price = safe_float(regular["Open"].iloc[0] if not regular.empty else today["Open"].iloc[0])
    previous_close, data_quality = estimate_previous_close(ticker, current_price, open_price)
    if current_price <= 0 or previous_close <= 0:
        return None, None, "missing_data"
    if current_price < settings.min_price or current_price > settings.max_price:
        return None, None, "price_filter"

    day_high = safe_float(today["High"].max(), current_price)
    day_low = safe_float(today["Low"].min(), current_price)
    volume = int(safe_float(today["Volume"].sum(), 0))
    gap_pct = ((current_price - previous_close) / previous_close) * 100
    intraday_gain_pct = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0.0
    gain_pct = max(gap_pct, intraday_gain_pct)
    last_2_move_pct = last_two_candle_move(analysis, current_price)
    accel = calc_volume_acceleration(analysis)
    rvol = calc_relative_volume(volume, analysis)
    near_high_pct = ((day_high - current_price) / current_price) * 100 if current_price > 0 else 100.0
    vwap = calc_vwap(analysis)
    above_vwap = not np.isnan(vwap) and current_price >= vwap
    vwap_reclaim = above_vwap and day_low <= vwap
    compression_pct, is_tight = tight_consolidation(analysis, current_price, day_high)
    has_higher_lows = higher_lows(analysis)
    volume_ignition = accel >= 1.5

    explosive_runner = intraday_gain_pct >= 10 or gap_pct >= 10 or last_2_move_pct >= 5 or accel >= 3
    pre_move_candidate = near_high_pct <= 3 and (is_tight or has_higher_lows or vwap_reclaim or accel >= 1.5)
    if not explosive_runner and not pre_move_candidate:
        watched = {
            "Ticker": ticker,
            "Price": current_price,
            "Gain %": gain_pct,
            "Volume": volume,
            "RVOL": rvol,
            "Near High %": near_high_pct,
            "Reason": "watched active mover",
        }
        return None, watched, "no_setup"

    breakout_20 = breakout_20d(ticker, current_price, state, settings) if (explosive_runner or pre_move_candidate) else False
    exp_score = explosion_score(gain_pct, rvol, accel, near_high_pct, volume)
    pre_score = premove_score(is_tight, has_higher_lows, near_high_pct, vwap_reclaim, volume_ignition, gap_pct, breakout_20)
    mode = "EXPLOSIVE" if explosive_runner else "PREMOVE"
    score = exp_score if explosive_runner else pre_score

    support = round_cent(max(day_low, current_price * 0.965))
    trigger = round_cent(max(day_high + 0.02, current_price * 1.006))
    pullback_low = round_cent(max(support, current_price * 0.92 if explosive_runner else current_price * 0.965))
    pullback_high = round_cent(max(pullback_low + 0.01, current_price * 0.97 if explosive_runner else current_price * 0.985))
    stop = round_cent(max(0.01, min(support, pullback_low * 0.985)))
    risk = max(trigger - stop, 0)
    target_1 = round_cent(trigger + risk * 2) if risk > 0 else round_cent(current_price * 1.08)
    target_2 = round_cent(trigger + risk * 3) if risk > 0 else round_cent(current_price * 1.14)
    rr = (target_1 - trigger) / risk if risk > 0 else 0.0
    trigger_distance_pct = abs(trigger - current_price) / current_price * 100 if current_price > 0 else 999
    data_mismatch_pct = abs(current_price - safe_float(ACTIVE_QUOTE_CACHE.get(ticker, {}).get("price"), current_price)) / current_price * 100
    below_open_and_vwap = current_price < open_price and (not np.isnan(vwap) and current_price < vwap)
    extended_from_pullback = ((current_price - pullback_high) / pullback_high) * 100 if pullback_high > 0 else 999

    reasons: list[str] = []
    if explosive_runner:
        if gain_pct >= 10:
            reasons.append("10%+ momentum")
        if last_2_move_pct >= 5:
            reasons.append("last 2 candles spike")
        if accel >= 3:
            reasons.append("volume acceleration >= 3x")
    if pre_move_candidate:
        if is_tight:
            reasons.append("tight consolidation")
        if has_higher_lows:
            reasons.append("higher lows")
        if vwap_reclaim:
            reasons.append("VWAP reclaim")
        if volume_ignition:
            reasons.append("volume ignition starting")
    if near_high_pct <= 3:
        reasons.append("near day high")
    if gap_pct > 0:
        reasons.append("gap up")
    if breakout_20:
        reasons.append("20-day breakout")

    if data_mismatch_pct > 10:
        status = "INVALID DATA"
    elif below_open_and_vwap:
        status = "WAIT FOR REVERSAL"
    elif explosive_runner and (extended_from_pullback > 10 or gain_pct >= 18):
        status = "HOT RUNNER — WAIT FOR PULLBACK"
    elif rvol < settings.min_rvol_for_valid:
        status = "BREAKOUT IMMINENT" if explosive_runner else "PRE-MOVE WATCH"
    elif trigger_distance_pct <= 3 and rr >= 2 and above_vwap and volume > settings.min_volume:
        status = "VALID TRADE"
    else:
        status = "BREAKOUT IMMINENT" if explosive_runner else "PRE-MOVE WATCH"

    if rvol < settings.min_rvol_for_valid and status == "VALID TRADE":
        status = "BREAKOUT IMMINENT" if explosive_runner else "PRE-MOVE WATCH"

    setup_type = "Explosive Runner Now" if explosive_runner else "Pre-Move Candidate"
    avoid_reason = (
        "Already extended; wait for pullback zone or clean VWAP reclaim."
        if explosive_runner
        else "Avoid if price loses VWAP/open, breaks higher-low structure, or volume fades."
    )
    if status == "INVALID DATA":
        avoid_reason = "Data mismatch above 10%; wait for a fresh scan before planning a trade."
    elif status == "WAIT FOR REVERSAL":
        avoid_reason = "Price is below VWAP and open; wait for reclaim before any long setup."

    row = {
        "Ticker": ticker,
        "Mode": mode,
        "Current Price": round(current_price, 4),
        "Gain %": round(gain_pct, 2),
        "Gap %": round(gap_pct, 2),
        "Intraday Gain %": round(intraday_gain_pct, 2),
        "Last 2 Candle Move %": round(last_2_move_pct, 2),
        "Volume": volume,
        "RVOL": rvol,
        "Volume Acceleration": accel,
        "Near High %": round(near_high_pct, 2),
        "VWAP Reclaim": vwap_reclaim,
        "Tight Consolidation": is_tight,
        "Higher Lows": has_higher_lows,
        "20D Breakout": breakout_20,
        "Score": score,
        "Explosion Score": exp_score,
        "Pre-Move Score": pre_score,
        "Status": status,
        "Setup Type": setup_type,
        "Trigger Entry": trigger if status != "INVALID DATA" else None,
        "Pullback Zone": f"{pullback_low:.2f}-{pullback_high:.2f}" if status != "INVALID DATA" else "N/A",
        "Stop": stop if status != "INVALID DATA" else None,
        "Target 1": target_1 if status != "INVALID DATA" else None,
        "Target 2": target_2 if status != "INVALID DATA" else None,
        "Risk/Reward": "N/A" if rr <= 0 or status == "INVALID DATA" else f"1:{rr:.2f}",
        "Reason": ", ".join(reasons) if reasons else "active pressure detected",
        "Avoid Reason": avoid_reason,
        "Data Quality": data_quality,
        "Last Updated": datetime.now().isoformat(timespec="seconds"),
    }
    return row, row.copy(), ""


def split_and_rank(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"], keep="last")
    status_rank = {
        "VALID TRADE": 0,
        "BREAKOUT IMMINENT": 1,
        "PRE-MOVE WATCH": 2,
        "HOT RUNNER — WAIT FOR PULLBACK": 3,
        "WAIT FOR REVERSAL": 4,
        "INVALID DATA": 5,
    }
    df["_status_rank"] = df["Status"].map(status_rank).fillna(9)
    explosive = df[df["Mode"].eq("EXPLOSIVE")].sort_values(["_status_rank", "Explosion Score", "RVOL"], ascending=[True, False, False])
    premove = df[df["Mode"].eq("PREMOVE")].sort_values(["_status_rank", "Pre-Move Score", "RVOL"], ascending=[True, False, False])
    return explosive.drop(columns=["_status_rank"]), premove.drop(columns=["_status_rank"])


def rank_watched(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"], keep="last")
    sort_cols = [col for col in ["Gain %", "RVOL", "Volume"] if col in df.columns]
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(150) if sort_cols else df.head(150)


def update_source_debug(debug_rows: list[dict[str, object]], scanned: dict[str, int], passed: dict[str, int]) -> list[dict[str, object]]:
    rows = [dict(row) for row in debug_rows]
    known = {str(row["Source"]) for row in rows}
    for source in set(scanned) | set(passed):
        if source not in known:
            rows.append({"Source": source, "Fetched": 0, "Scanned": 0, "Passed Filters": 0})
    for row in rows:
        source = str(row["Source"])
        row["Scanned"] = int(scanned.get(source, 0))
        row["Passed Filters"] = int(passed.get(source, 0))
    return rows


def publish(
    state: AppState,
    candidate_rows: list[dict[str, object]],
    watched_rows: list[dict[str, object]],
    skip_counts: dict[str, int],
    source_debug: list[dict[str, object]],
    scanned_count: int,
    started: float,
    status_text: str,
    timed_out: bool = False,
    source_message: Optional[str] = None,
) -> None:
    explosive, premove = split_and_rank(candidate_rows)
    watched = rank_watched(watched_rows)
    with state.lock:
        state.explosive_df = explosive
        state.premove_df = premove
        state.watched_df = watched
        state.source_debug_df = pd.DataFrame(source_debug)
        state.skip_counts = dict(skip_counts)
        state.scanned_count = scanned_count
        state.last_scan_seconds = time.monotonic() - started
        state.status_text = status_text
        state.timed_out = timed_out
        state.source_message = source_message


def scan_once(state: AppState, settings: ScannerSettings) -> None:
    started = time.monotonic()
    deadline = started + settings.scan_timeout_seconds
    tickers, source_debug, source_by_ticker, source_message = build_universe(state, settings)
    candidate_rows: list[dict[str, object]] = []
    watched_rows: list[dict[str, object]] = []
    skip_counts = {key: 0 for key in SKIP_KEYS}
    scanned_by_source: dict[str, int] = {}
    passed_by_source: dict[str, int] = {}
    scanned = 0

    with state.lock:
        state.last_scan_started = datetime.now().isoformat(timespec="seconds")
        state.status_text = f"Scanning {len(tickers)} active small-cap candidates..."
        state.error = None
        state.timed_out = False
        state.source_message = source_message

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
                source = source_by_ticker.get(ticker, "runner_fallback")
                scanned += 1
                scanned_by_source[source] = scanned_by_source.get(source, 0) + 1
                try:
                    candidate, watched, skip_reason = future.result(timeout=0)
                except Exception:
                    candidate, watched, skip_reason = None, None, "missing_data"
                if watched is not None:
                    watched["Source"] = source
                    watched_rows.append(watched)
                if candidate is not None:
                    candidate["Source"] = source
                    candidate_rows.append(candidate)
                    passed_by_source[source] = passed_by_source.get(source, 0) + 1
                elif skip_reason in skip_counts:
                    skip_counts[skip_reason] += 1
                debug = update_source_debug(source_debug, scanned_by_source, passed_by_source)
                publish(state, candidate_rows, watched_rows, skip_counts, debug, scanned, started, f"Scanning: {scanned}/{len(tickers)} complete", source_message=source_message)
            pending = {future: pending[future] for future in pending_set}
        timed_out = bool(pending)
        for future in pending:
            future.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    debug = update_source_debug(source_debug, scanned_by_source, passed_by_source)
    status = "Scan timeout reached; showing partial results." if timed_out else "Scan complete."
    publish(state, candidate_rows, watched_rows, skip_counts, debug, scanned, started, status, timed_out, source_message)
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
                state.status_text = "Scanner error."
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
        state.explosive_df = pd.DataFrame()
        state.premove_df = pd.DataFrame()
        state.watched_df = pd.DataFrame()
        state.source_debug_df = pd.DataFrame()
        state.skip_counts = {key: 0 for key in SKIP_KEYS}
        state.scanned_count = 0
        state.last_scan_seconds = 0.0
        state.timed_out = False
        state.source_message = None
        state.error = None
        state.status_text = "Restarting scanner..."
        if clear_cache:
            state.intraday_cache = {}
            state.daily_cache = {}
            state.universe_cache = None
        thread = threading.Thread(target=scanner_loop, args=(state,), daemon=True, name="explosive-runner-scanner")
        state.scanner_thread = thread
        thread.start()


def ensure_scanner(state: AppState) -> None:
    with state.lock:
        stale = state.version != APP_VERSION
        alive = bool(state.scanner_thread and state.scanner_thread.is_alive())
        settings = state.settings
    if stale:
        restart_scanner(state, ScannerSettings(), clear_cache=True)
    elif not alive:
        restart_scanner(state, settings)


def snapshot(state: AppState) -> dict[str, object]:
    with state.lock:
        return {
            "settings": state.settings,
            "explosive": state.explosive_df.copy(),
            "premove": state.premove_df.copy(),
            "watched": state.watched_df.copy(),
            "source_debug": state.source_debug_df.copy(),
            "skip_counts": dict(state.skip_counts),
            "cycle": state.cycle_count,
            "scanned": state.scanned_count,
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
            }
            .stApp {
                background:
                    radial-gradient(circle at 20% 0%, rgba(239, 68, 68, 0.08), transparent 28rem),
                    linear-gradient(180deg, #060a11 0%, #0a101b 100%);
                color: var(--text);
            }
            [data-testid="stSidebar"] {
                background: #09111f;
                border-right: 1px solid #1f2937;
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1760px;
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
                padding: 0.85rem;
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
                min-height: 82px;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 16px;
                border-color: var(--line);
                background: linear-gradient(180deg, rgba(17, 26, 43, 0.96), rgba(11, 18, 32, 0.96));
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.25);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                min-height: 340px;
            }
            .stButton button {
                border-radius: 16px;
                border: 1px solid #334155;
            }
            [data-testid="stDataFrame"] div,
            [data-testid="stDataFrame"] span {
                white-space: nowrap;
            }
            @media (max-width: 760px) {
                .block-container {
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                }
                div[data-testid="stVerticalBlockBorderWrapper"] > div {
                    min-height: auto;
                }
                div[data-testid="stDataFrame"] {
                    display: none;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_runner_card(row: pd.Series) -> None:
    with st.container(border=True):
        header = st.columns([0.52, 0.48], vertical_alignment="center")
        header[0].markdown(f"### {row.get('Ticker', 'N/A')}")
        header[1].markdown(status_badge_text(str(row.get("Status", "WATCH"))))
        score = row.get("Explosion Score") if row.get("Mode") == "EXPLOSIVE" else row.get("Pre-Move Score")
        st.progress(min(max(safe_float(score, 0) / 100, 0), 1), text=f"Score {fmt_score(score)}")
        grid_top = st.columns(2)
        grid_top[0].metric("Price", fmt_currency(row.get("Current Price")))
        grid_top[1].metric("Trigger", fmt_currency(row.get("Trigger Entry")))
        grid_bottom = st.columns(2)
        grid_bottom[0].metric("Pullback", str(row.get("Pullback Zone", "N/A")))
        grid_bottom[1].metric("Target", fmt_currency(row.get("Target 1")))
        st.caption(str(row.get("Reason", "")))
        st.caption(f"Avoid: {row.get('Avoid Reason', '')}")


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Ticker": row.get("Ticker", ""),
                "Status": row.get("Status", ""),
                "Score": fmt_score(row.get("Score")),
                "Price": fmt_currency(row.get("Current Price")),
                "Gain": fmt_pct(row.get("Gain %")),
                "RVOL": fmt_rvol(row.get("RVOL")),
                "Trigger": fmt_currency(row.get("Trigger Entry")),
                "Pullback": row.get("Pullback Zone", "N/A"),
                "Stop": fmt_currency(row.get("Stop")),
                "Target 1": fmt_currency(row.get("Target 1")),
                "Reason": row.get("Reason", ""),
            }
        )
    return pd.DataFrame(rows)


def render_section(title: str, df: pd.DataFrame, empty_text: str) -> None:
    st.subheader(title)
    if df.empty:
        st.info(empty_text)
        return
    cards = df.head(3)
    cols = st.columns(min(3, len(cards)), gap="large")
    for index, (_, row) in enumerate(cards.iterrows()):
        with cols[index % len(cols)]:
            render_runner_card(row)
    st.dataframe(compact_table(df), use_container_width=True, hide_index=True)


def render_dashboard(snap: dict[str, object]) -> None:
    explosive: pd.DataFrame = snap["explosive"]  # type: ignore[assignment]
    premove: pd.DataFrame = snap["premove"]  # type: ignore[assignment]
    watched: pd.DataFrame = snap["watched"]  # type: ignore[assignment]
    source_debug: pd.DataFrame = snap["source_debug"]  # type: ignore[assignment]
    skip_counts: dict[str, int] = snap["skip_counts"]  # type: ignore[assignment]

    total_candidates = len(explosive) + len(premove)
    metrics = st.columns(5)
    metrics[0].metric("Scanner", "RUNNING" if snap["thread_alive"] else "STOPPED")
    metrics[1].metric("Cycle", f"{int(snap['cycle']):,}")
    metrics[2].metric("Scanned", f"{int(snap['scanned']):,}")
    metrics[3].metric("Setups", f"{total_candidates:,}")
    metrics[4].metric("Last Scan", f"{safe_float(snap['last_scan_seconds'], 0):.1f}s")

    st.caption(
        f"{snap['status_text']} | Last updated: {datetime.now().isoformat(timespec='seconds')} | "
        f"Last finished: {snap['last_scan_finished'] or 'Waiting...'}"
    )
    if snap["timed_out"]:
        st.warning("20 second timeout reached. Partial results are shown.")
    if snap["source_message"]:
        st.warning(str(snap["source_message"]))
    if snap["error"]:
        st.error(str(snap["error"]))

    render_section("Explosive Runners Now", explosive, "No explosive runners detected yet.")
    render_section("Pre-Move Candidates", premove, "No high-quality pre-move candidates detected yet.")

    st.subheader("Watched Movers")
    if watched.empty:
        top_skip = max(SKIP_KEYS, key=lambda key: skip_counts.get(key, 0))
        st.warning(f"No watched movers yet. Largest skip bucket: {top_skip} ({skip_counts.get(top_skip, 0):,}).")
    else:
        watched_cols = [col for col in ["Ticker", "Price", "Gain %", "Volume", "RVOL", "Near High %", "Reason", "Source"] if col in watched.columns]
        st.dataframe(watched[watched_cols], use_container_width=True, hide_index=True)

    with st.expander("Extra details and diagnostics", expanded=False):
        details = pd.concat([explosive, premove], ignore_index=True) if not explosive.empty or not premove.empty else pd.DataFrame()
        if not details.empty:
            extra_cols = [
                "Ticker",
                "Setup Type",
                "Gap %",
                "Intraday Gain %",
                "Last 2 Candle Move %",
                "Volume Acceleration",
                "Near High %",
                "VWAP Reclaim",
                "20D Breakout",
                "Risk/Reward",
                "Avoid Reason",
                "Data Quality",
                "Source",
            ]
            st.dataframe(details[[col for col in extra_cols if col in details.columns]], use_container_width=True, hide_index=True)
        if not source_debug.empty:
            st.dataframe(source_debug, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([{"Reason": key, "Count": skip_counts.get(key, 0)} for key in SKIP_KEYS]), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Small-Cap Explosive Runner Scanner", layout="wide")
    apply_theme()
    state = get_state()
    ensure_scanner(state)
    snap = snapshot(state)
    settings: ScannerSettings = snap["settings"]  # type: ignore[assignment]

    st.title("Small-Cap Explosive Runner Scanner")
    st.caption("Ranks explosive runners and pre-breakout pressure without pretending every signal is a trade.")

    with st.sidebar:
        st.header("Scanner Controls")
        cloud_fast = st.toggle("Cloud Fast Mode", value=settings.cloud_fast_mode)
        local_full = st.toggle("Local Full Mode", value=not settings.cloud_fast_mode)
        if local_full:
            cloud_fast = False
        max_limit = 150 if cloud_fast else 1000
        max_tickers = st.slider("Max tickers", 50, max_limit, min(settings.max_tickers, max_limit), step=10)
        workers = st.slider("Workers", 1, 12, min(settings.max_workers, 12), step=1)
        scan_interval = st.slider("Scan interval seconds", 15, 300, settings.scan_interval_seconds, step=5)
        scan_timeout = st.slider("Scan timeout seconds", 5, 30, settings.scan_timeout_seconds, step=1)
        max_price = st.slider("Maximum price", 1.0, 200.0, float(settings.max_price), step=1.0)
        min_rvol = st.slider("Minimum RVOL for VALID TRADE", 0.5, 3.0, float(settings.min_rvol_for_valid), step=0.1)
        with st.expander("Advanced", expanded=False):
            watchlist = st.text_area("Optional advanced watchlist", value=settings.optional_watchlist, height=80)
            st.caption("Automatic discovery is primary. Use this only to force-check names.")

        new_settings = ScannerSettings(
            cloud_fast_mode=cloud_fast,
            scan_interval_seconds=int(scan_interval),
            scan_timeout_seconds=int(scan_timeout),
            max_tickers=int(max_tickers),
            max_workers=int(workers),
            max_price=float(max_price),
            min_rvol_for_valid=float(min_rvol),
            optional_watchlist=watchlist,
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
