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

APP_VERSION = "pre-move-momentum-scanner-2026-04-30"
YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
DEFAULT_PRIORITY_TICKERS = "AKAN, RDAC, BIYA, SBLX, ATER"

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
    priority_tickers: str = DEFAULT_PRIORITY_TICKERS


@dataclass
class AppState:
    version: str = APP_VERSION
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    scanner_thread: Optional[threading.Thread] = None
    settings: ScannerSettings = field(default_factory=ScannerSettings)
    candidates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    watched_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skip_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in SKIP_KEYS})
    cycle_count: int = 0
    scanned_count: int = 0
    candidate_count: int = 0
    last_scan_started: Optional[str] = None
    last_scan_finished: Optional[str] = None
    last_scan_seconds: float = 0.0
    timed_out: bool = False
    status_text: str = "Starting scanner..."
    error: Optional[str] = None
    download_cache: dict[str, tuple[float, pd.DataFrame]] = field(default_factory=dict)
    universe_cache: tuple[float, list[str]] | None = None


@st.cache_resource(show_spinner=False)
def get_state() -> AppState:
    return AppState()


def is_valid_ticker(ticker: str) -> bool:
    return bool(VALID_TICKER_RE.fullmatch(str(ticker).upper().strip()))


def parse_priority_tickers(value: str) -> list[str]:
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


def get_active_movers(max_count: int) -> list[str]:
    global ACTIVE_QUOTE_CACHE
    tickers: list[str] = []
    screens = [
        "day_gainers",
        "most_actives",
        "pre_market_gainers",
        "pre_market_most_actives",
        "undervalued_growth_stocks",
    ]

    for screen in screens:
        try:
            data = safe_request_json(YAHOO_SCREENER_URL, {"scrIds": screen, "count": min(max_count, 250)})
            quotes = data["finance"]["result"][0]["quotes"]  # type: ignore[index]
        except Exception:
            continue

        for quote in quotes:
            symbol = str(quote.get("symbol", "")).upper().strip()
            quote_type = str(quote.get("quoteType", "")).upper()
            market = str(quote.get("market", "")).lower()
            if quote_type != "EQUITY" or market not in {"us_market", ""} or not is_valid_ticker(symbol):
                continue
            tickers.append(symbol)
            ACTIVE_QUOTE_CACHE[symbol] = {
                "price": quote.get("regularMarketPrice") or quote.get("preMarketPrice") or quote.get("postMarketPrice"),
                "previous_close": quote.get("regularMarketPreviousClose"),
                "change_pct": quote.get("regularMarketChangePercent") or quote.get("preMarketChangePercent"),
                "volume": quote.get("regularMarketVolume") or quote.get("preMarketVolume"),
            }

    return list(dict.fromkeys(tickers))[:max_count]


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

        other_text = requests.get(OTHER_LISTED_URL, headers=headers, timeout=4).text
        other = pd.read_csv(StringIO(other_text), sep="|")
        other = other[other["ACT Symbol"].notna()]
        other = other[other["ACT Symbol"] != "File Creation Time"]
        other = other.rename(columns={"ACT Symbol": "Ticker", "Security Name": "Name"})
        frames.append(other[["Ticker", "Name"]])
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


def build_scan_universe(state: AppState, settings: ScannerSettings) -> list[str]:
    priority = parse_priority_tickers(settings.priority_tickers)
    active = get_active_movers(max(settings.max_tickers, 150))

    if settings.cloud_fast_mode:
        return list(dict.fromkeys(priority + active))[: settings.max_tickers]

    names = fetch_security_names(state)
    all_tickers = list(names) or FALLBACK_TICKERS
    remaining = [ticker for ticker in all_tickers if ticker not in set(priority + active)]
    explore_size = min(len(remaining), max(0, settings.max_tickers - len(priority + active)))
    explore = random.sample(remaining, explore_size) if explore_size else []
    return list(dict.fromkeys(priority + active + explore))[: settings.max_tickers]


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


def publish(
    state: AppState,
    candidate_rows: list[dict[str, object]],
    watched_rows: list[dict[str, object]],
    skip_counts: dict[str, int],
    scanned: int,
    started: float,
    status_text: str,
    timed_out: bool = False,
) -> None:
    candidates = rank_candidates(candidate_rows).head(state.settings.output_limit)
    watched = rank_watched(watched_rows)
    with state.lock:
        state.candidates_df = candidates
        state.watched_df = watched
        state.skip_counts = dict(skip_counts)
        state.scanned_count = scanned
        state.candidate_count = len(candidates)
        state.last_scan_seconds = time.monotonic() - started
        state.timed_out = timed_out
        state.status_text = status_text


def scan_once(state: AppState, settings: ScannerSettings) -> None:
    started = time.monotonic()
    deadline = started + settings.scan_timeout_seconds
    tickers = build_scan_universe(state, settings)
    candidate_rows: list[dict[str, object]] = []
    watched_rows: list[dict[str, object]] = []
    skip_counts = {key: 0 for key in SKIP_KEYS}
    scanned = 0

    with state.lock:
        state.last_scan_started = datetime.now().isoformat(timespec="seconds")
        state.status_text = f"Scanning {len(tickers)} active symbols..."
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
                try:
                    candidate, watched, skip_reason = future.result(timeout=0)
                except Exception:
                    candidate, watched, skip_reason = None, None, "missing_data"
                if watched is not None:
                    watched_rows.append(watched)
                if candidate is not None:
                    candidate_rows.append(candidate)
                elif skip_reason in skip_counts:
                    skip_counts[skip_reason] += 1
                status = f"Scanning active movers: {scanned}/{len(tickers)} complete"
                publish(state, candidate_rows, watched_rows, skip_counts, scanned, started, status)
            pending = {future: pending[future] for future in pending_set}

        timed_out = bool(pending)
        for future in pending:
            future.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    status_text = "Scan timeout reached; showing partial results." if timed_out else "Scan complete."
    publish(state, candidate_rows, watched_rows, skip_counts, scanned, started, status_text, timed_out)
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
        state.skip_counts = {key: 0 for key in SKIP_KEYS}
        state.scanned_count = 0
        state.candidate_count = 0
        state.last_scan_seconds = 0.0
        state.timed_out = False
        state.error = None
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
            "skip_counts": dict(state.skip_counts),
            "cycle": state.cycle_count,
            "scanned": state.scanned_count,
            "candidate_count": state.candidate_count,
            "last_scan_seconds": state.last_scan_seconds,
            "last_scan_started": state.last_scan_started,
            "last_scan_finished": state.last_scan_finished,
            "timed_out": state.timed_out,
            "status_text": state.status_text,
            "error": state.error,
            "thread_alive": bool(state.scanner_thread and state.scanner_thread.is_alive()),
        }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #070b12; color: #e5e7eb; }
            [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1f2937; }
            .block-container { padding-top: 1.2rem; max-width: 1440px; }
            h1, h2, h3 { color: #f8fafc; letter-spacing: 0; }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
                border: 1px solid #243244;
                border-radius: 8px;
                padding: 0.75rem;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #1f2937;
                border-radius: 8px;
            }
            .stButton button {
                border-radius: 8px;
                border: 1px solid #334155;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_priority_card(row: pd.Series) -> None:
    with st.container(border=True):
        head = st.columns([0.26, 0.34, 0.4])
        head[0].markdown(f"**{row.get('Ticker', 'N/A')}**")
        head[1].markdown(status_badge(str(row.get("Status", "WAIT FOR TRIGGER"))))
        head[2].caption(str(row.get("Setup Type", "PRE-MOVE WATCH")))

        metrics = st.columns(4)
        metrics[0].metric("Price", f"${fmt_price(row.get('Current Price'))}")
        metrics[1].metric("Score", f"{safe_float(row.get('Pre-Move Score'), 0):.1f}")
        metrics[2].metric("RVOL", f"{safe_float(row.get('RVOL'), 0):.2f}x")
        metrics[3].metric("Near High", f"{safe_float(row.get('Near High %'), 0):.2f}%")

        trade = st.columns(4)
        trade[0].caption(f"Entry {fmt_price(row.get('Trigger Entry'))}")
        trade[1].caption(f"Stop {fmt_price(row.get('Stop'))}")
        trade[2].caption(f"T1 {fmt_price(row.get('Target 1'))}")
        trade[3].caption(str(row.get("Risk/Reward", "N/A")))
        st.caption(str(row.get("Why This May Run", "")))


def render_dashboard(snap: dict[str, object]) -> None:
    candidates: pd.DataFrame = snap["candidates"]  # type: ignore[assignment]
    watched: pd.DataFrame = snap["watched"]  # type: ignore[assignment]
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
    if snap["error"]:
        st.error(str(snap["error"]))

    st.subheader("High Priority Pre-Move")
    top5 = candidates.head(5)
    if top5.empty:
        st.info("No filtered pre-move candidates yet. Showing watched movers below.")
    else:
        cols = st.columns(min(5, len(top5)))
        for index, (_, row) in enumerate(top5.iterrows()):
            with cols[index % len(cols)]:
                render_priority_card(row)

    st.subheader("Pre-Move Scanner Table")
    display_cols = [
        "Ticker",
        "Current Price",
        "Pre-Move Score",
        "Status",
        "Setup Type",
        "Trigger Entry",
        "Stop",
        "Target 1",
        "Target 2",
        "Risk/Reward",
        "RVOL",
        "Volume Accel",
        "Near High %",
        "Gap %",
        "Confirmation",
        "Invalidation",
        "Why This May Run",
        "Avoid Reason",
    ]

    if not candidates.empty:
        table = candidates[[col for col in display_cols if col in candidates.columns]].copy()
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pre-Move Score": st.column_config.ProgressColumn("Pre-Move Score", min_value=0, max_value=100),
                "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "Trigger Entry": st.column_config.NumberColumn("Trigger Entry", format="$%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                "Target 1": st.column_config.NumberColumn("Target 1", format="$%.2f"),
                "Target 2": st.column_config.NumberColumn("Target 2", format="$%.2f"),
                "RVOL": st.column_config.NumberColumn("RVOL", format="%.2fx"),
                "Volume Accel": st.column_config.NumberColumn("Volume Accel", format="%.2fx"),
                "Near High %": st.column_config.NumberColumn("Near High %", format="%.2f%%"),
                "Gap %": st.column_config.NumberColumn("Gap %", format="%.2f%%"),
            },
        )
    elif not watched.empty:
        st.dataframe(watched, use_container_width=True, hide_index=True)
    else:
        top_skip = max(SKIP_KEYS, key=lambda key: skip_counts.get(key, 0))
        st.warning(f"No symbols returned usable data yet. Largest skip bucket: {top_skip} ({skip_counts.get(top_skip, 0):,}).")

    with st.expander("Watched movers and scanner diagnostics", expanded=False):
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
    st.caption("Speed-first active mover scanner for pressure before breakout, not late chase entries.")

    with st.sidebar:
        st.header("Scanner Controls")
        cloud_fast_mode = st.toggle("Cloud Fast Mode", value=settings.cloud_fast_mode)
        local_full_mode = st.toggle("Local Full Mode", value=not settings.cloud_fast_mode)
        if local_full_mode:
            cloud_fast_mode = False

        priority_tickers = st.text_area("Priority tickers", value=settings.priority_tickers, height=80)
        max_tickers_max = 150 if cloud_fast_mode else 1000
        max_tickers = st.slider("Max tickers", 50, max_tickers_max, min(settings.max_tickers, max_tickers_max), step=10)
        workers = st.slider("Workers", 1, 12, min(settings.max_workers, 12), step=1)
        scan_interval = st.slider("Scan interval seconds", 15, 300, settings.scan_interval_seconds, step=5)
        scan_timeout = st.slider("Scan timeout seconds", 5, 30, settings.scan_timeout_seconds, step=1)
        min_volume = st.number_input("Minimum volume", min_value=0, value=settings.min_volume, step=25_000)
        max_price = st.slider("Maximum price", 1.0, 200.0, float(settings.max_price), step=1.0)
        min_rvol = st.slider("Minimum RVOL pressure", 0.0, 3.0, float(settings.min_relative_volume), step=0.1)

        new_settings = ScannerSettings(
            cloud_fast_mode=cloud_fast_mode,
            scan_interval_seconds=int(scan_interval),
            scan_timeout_seconds=int(scan_timeout),
            max_tickers=int(max_tickers),
            max_workers=int(workers),
            max_price=float(max_price),
            min_volume=int(min_volume),
            min_relative_volume=float(min_rvol),
            priority_tickers=priority_tickers,
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
