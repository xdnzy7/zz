from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
YAHOO_GAINERS_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

FALLBACK_TICKERS = [
    "BIYA",
    "SBLX",
    "ATER",
    "AKAN",
    "ARET",
    "MULN",
    "HOLO",
    "GNS",
    "LUCY",
    "WISA",
    "FFIE",
    "TIVC",
    "BFRG",
    "MLGO",
    "SOUN",
    "BBAI",
    "KULR",
    "WULF",
    "OPEN",
    "IONQ",
    "PLUG",
]

SECURITY_NAME_CACHE: dict[str, str] = {}


@dataclass
class ScannerSettings:
    fast_mode: bool = True
    scan_interval_seconds: int = 7
    min_price: float = 0.50
    max_price: float = 100.00
    min_volume: int = 100_000
    min_gain_pct: float = 0.0
    min_relative_volume: float = 1.3
    near_high_threshold_pct: float = 3.0
    volume_acceleration_threshold: float = 1.5
    halt_spike_threshold_pct: float = 15.0
    max_tickers: int = 1000
    active_movers_count: int = 500
    explore_tickers_count: int = 500
    batch_size: int = 40
    max_workers: int = 20
    output_limit: int = 100
    cache_seconds: int = 20
    request_timeout_seconds: int = 4


@dataclass
class AppState:
    lock: threading.RLock
    stop_event: threading.Event
    settings: ScannerSettings
    scanner_thread: Optional[threading.Thread]
    candidates_df: pd.DataFrame
    plans_df: pd.DataFrame
    last_scan_started: Optional[str]
    last_scan_finished: Optional[str]
    last_scan_seconds: float
    scanned_count: int
    skipped_count: int
    cycle_count: int
    error: Optional[str]
    download_cache: dict[tuple[str, str, str, bool], tuple[float, pd.DataFrame]]
    previous_close_cache: dict[str, tuple[float, Optional[float]]]
    average_volume_cache: dict[str, tuple[float, Optional[float]]]
    universe_cache: tuple[float, list[str]] | None


def make_empty_state() -> AppState:
    return AppState(
        lock=threading.RLock(),
        stop_event=threading.Event(),
        settings=ScannerSettings(),
        scanner_thread=None,
        candidates_df=pd.DataFrame(),
        plans_df=pd.DataFrame(),
        last_scan_started=None,
        last_scan_finished=None,
        last_scan_seconds=0.0,
        scanned_count=0,
        skipped_count=0,
        cycle_count=0,
        error=None,
        download_cache={},
        previous_close_cache={},
        average_volume_cache={},
        universe_cache=None,
    )


@st.cache_resource(show_spinner=False)
def get_state() -> AppState:
    return make_empty_state()


def round_cent(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def fmt_price(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(value: object) -> str:
    if value is None or pd.isna(value):
        return "0"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def status_color(status: str) -> str:
    if status == "VALID TRADE":
        return "#16a34a"
    if status in {"WAIT", "WAIT FOR PULLBACK", "WAIT FOR REVERSAL"}:
        return "#d6a100"
    return "#dc2626"


def status_icon(status: str) -> str:
    if status == "VALID TRADE":
        return ":green[VALID TRADE]"
    if status in {"WAIT", "WAIT FOR PULLBACK", "WAIT FOR REVERSAL"}:
        return f":orange[{status}]"
    return f":red[{status}]"


def is_valid_ticker(ticker: str) -> bool:
    return bool(VALID_TICKER_RE.fullmatch(str(ticker).upper().strip()))


def safe_request_text(url: str, timeout: int = 5) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def get_us_tickers(state: AppState) -> list[str]:
    global SECURITY_NAME_CACHE
    now = time.monotonic()
    with state.lock:
        if state.universe_cache and now - state.universe_cache[0] <= 900:
            return list(state.universe_cache[1])

    frames: list[pd.DataFrame] = []
    try:
        nasdaq = pd.read_csv(StringIO(safe_request_text(NASDAQ_LISTED_URL)), sep="|")
        nasdaq = nasdaq[nasdaq["Symbol"].notna()]
        nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"]
        nasdaq = nasdaq.rename(columns={"Symbol": "Ticker", "Security Name": "Name"})
        nasdaq["Exchange"] = "NASDAQ"
        frames.append(nasdaq[["Ticker", "Name", "Exchange", "ETF", "Test Issue"]])

        other = pd.read_csv(StringIO(safe_request_text(OTHER_LISTED_URL)), sep="|")
        other = other[other["ACT Symbol"].notna()]
        other = other[other["ACT Symbol"] != "File Creation Time"]
        other = other.rename(columns={"ACT Symbol": "Ticker", "Security Name": "Name"})
        other["Exchange"] = other["Exchange"].replace({"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe"})
        frames.append(other[["Ticker", "Name", "Exchange", "ETF", "Test Issue"]])
    except Exception:
        return FALLBACK_TICKERS

    if not frames:
        return FALLBACK_TICKERS

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].fillna("").astype(str).str.lower()
    SECURITY_NAME_CACHE = dict(zip(df["Ticker"], df["Name"]))
    df = df[[is_valid_ticker(ticker) for ticker in df["Ticker"]]]
    df = df[df["ETF"].astype(str).str.upper().ne("Y")]
    df = df[df["Test Issue"].astype(str).str.upper().ne("Y")]

    excluded_terms = [
        "warrant",
        "warrants",
        "unit",
        "units",
        "right",
        "rights",
        "preferred",
        "preference",
        "depositary",
        "depository",
        "note",
        "bond",
        "etf",
        "fund",
        "trust",
        "closed end",
        "income",
        "dividend",
        "treasury",
        "municipal",
    ]
    for term in excluded_terms:
        df = df[~df["Name"].str.contains(term, na=False)]

    df = df[df["Exchange"].isin({"NASDAQ", "NYSE", "NYSE American", "NYSE Arca"})]
    tickers = df["Ticker"].drop_duplicates().tolist() or FALLBACK_TICKERS
    with state.lock:
        state.universe_cache = (now, tickers)
    return tickers


def get_active_movers(max_count: int) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers: list[str] = []
    for screen_id in ("day_gainers", "most_actives"):
        params = {"scrIds": screen_id, "count": min(max_count, 250)}
        try:
            response = requests.get(YAHOO_GAINERS_URL, params=params, headers=headers, timeout=4)
            response.raise_for_status()
            quotes = response.json()["finance"]["result"][0]["quotes"]
        except Exception:
            continue

        for quote in quotes:
            symbol = str(quote.get("symbol", "")).upper().strip()
            quote_type = str(quote.get("quoteType", "")).upper()
            market = str(quote.get("market", "")).lower()
            if quote_type == "EQUITY" and market in {"us_market", ""} and is_valid_ticker(symbol):
                tickers.append(symbol)
    return list(dict.fromkeys(tickers))[:max_count]


def build_scan_universe(state: AppState, settings: ScannerSettings) -> list[str]:
    all_tickers = get_us_tickers(state)
    all_set = set(all_tickers)
    active = [ticker for ticker in get_active_movers(settings.max_tickers) if ticker in all_set]
    active = active[: min(len(active), settings.max_tickers)]

    explore_pool = [ticker for ticker in all_tickers if ticker not in set(active)]
    explore_size = min(len(explore_pool), settings.explore_tickers_count, max(0, settings.max_tickers - len(active)))
    explore = random.sample(explore_pool, explore_size) if explore_size else []

    tickers = list(dict.fromkeys(active + explore))
    if len(tickers) < min(settings.max_tickers, len(all_tickers)):
        remaining = [ticker for ticker in all_tickers if ticker not in set(tickers)]
        fill_size = min(len(remaining), settings.max_tickers - len(tickers))
        if fill_size > 0:
            tickers.extend(random.sample(remaining, fill_size))
    return tickers[: min(settings.max_tickers, len(all_tickers))]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def cached_download(ticker: str, state: AppState, settings: ScannerSettings) -> pd.DataFrame:
    key = (ticker, "1d", "5m", True)
    now = time.monotonic()
    with state.lock:
        cached = state.download_cache.get(key)
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

    if df is None or df.empty:
        df = pd.DataFrame()
    else:
        df = normalize_columns(df)
        df = df.dropna(subset=["Close"]) if "Close" in df else pd.DataFrame()

    with state.lock:
        state.download_cache[key] = (now, df.copy())
        if len(state.download_cache) > 1500:
            state.download_cache = dict(list(state.download_cache.items())[-800:])
    return df


def valid_price(value: object) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(price) or price <= 0:
        return None
    return price


def get_previous_close(ticker: str, state: AppState, settings: ScannerSettings) -> Optional[float]:
    now = time.monotonic()
    with state.lock:
        cached = state.previous_close_cache.get(ticker)
        if cached and now - cached[0] <= 900:
            return cached[1]

    previous_close: Optional[float] = None

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fast_info = yf.Ticker(ticker).fast_info
        for key in ("previous_close", "regular_market_previous_close", "last_close"):
            try:
                previous_close = valid_price(fast_info.get(key))  # type: ignore[attr-defined]
            except AttributeError:
                previous_close = valid_price(getattr(fast_info, key, None))
            if previous_close is not None:
                break
    except Exception:
        previous_close = None

    if previous_close is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                daily = yf.download(
                    ticker,
                    period="7d",
                    interval="1d",
                    prepost=False,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    timeout=settings.request_timeout_seconds,
                )
            if daily is not None and not daily.empty:
                daily = normalize_columns(daily)
                daily = daily.dropna(subset=["Close"]) if "Close" in daily else pd.DataFrame()
                if not daily.empty:
                    today_et = pd.Timestamp.now(tz="America/New_York").date()
                    daily_dates = pd.to_datetime(daily.index).date
                    prior_days = daily.loc[daily_dates < today_et]
                    source = prior_days if not prior_days.empty else daily
                    previous_close = valid_price(source["Close"].iloc[-1])
        except Exception:
            previous_close = None

    with state.lock:
        state.previous_close_cache[ticker] = (now, previous_close)
        if len(state.previous_close_cache) > 3000:
            state.previous_close_cache = dict(list(state.previous_close_cache.items())[-1500:])
    return previous_close


def get_average_volume(ticker: str, state: AppState, settings: ScannerSettings) -> Optional[float]:
    now = time.monotonic()
    with state.lock:
        cached = state.average_volume_cache.get(ticker)
        if cached and now - cached[0] <= 900:
            return cached[1]

    average_volume: Optional[float] = None

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fast_info = yf.Ticker(ticker).fast_info
        for key in ("ten_day_average_volume", "three_month_average_volume"):
            try:
                average_volume = valid_price(fast_info.get(key))  # type: ignore[attr-defined]
            except AttributeError:
                average_volume = valid_price(getattr(fast_info, key, None))
            if average_volume is not None:
                break
    except Exception:
        average_volume = None

    if average_volume is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                daily = yf.download(
                    ticker,
                    period="30d",
                    interval="1d",
                    prepost=False,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    timeout=settings.request_timeout_seconds,
                )
            if daily is not None and not daily.empty:
                daily = normalize_columns(daily)
                daily = daily.dropna(subset=["Volume"]) if "Volume" in daily else pd.DataFrame()
                if not daily.empty:
                    today_et = pd.Timestamp.now(tz="America/New_York").date()
                    daily_dates = pd.to_datetime(daily.index).date
                    prior_days = daily.loc[daily_dates < today_et]
                    source = prior_days.tail(20) if not prior_days.empty else daily.tail(20)
                    average_volume = valid_price(source["Volume"].mean())
        except Exception:
            average_volume = None

    with state.lock:
        state.average_volume_cache[ticker] = (now, average_volume)
        if len(state.average_volume_cache) > 3000:
            state.average_volume_cache = dict(list(state.average_volume_cache.items())[-1500:])
    return average_volume


def latest_session_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    eastern = working.index.tz_convert("America/New_York")
    working["session_date"] = eastern.date
    today = working[working["session_date"] == working["session_date"].max()].copy()
    regular = today.between_time("09:30", "16:00")
    return today, regular


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty or float(df["Volume"].sum()) <= 0:
        return np.nan
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return round(float((typical * df["Volume"]).sum() / df["Volume"].sum()), 4)


def estimate_relative_volume(volume: int, regular: pd.DataFrame) -> float:
    if regular.empty:
        return 0.0
    bars_seen = max(1, min(len(regular), 78))
    rolling_bar_volume = float(regular["Volume"].tail(20).mean())
    baseline = rolling_bar_volume * 78
    if baseline <= 0:
        return 0.0
    projected = volume * 78 / bars_seen
    return projected / baseline


def detect_volume_acceleration(regular: pd.DataFrame, threshold: float) -> tuple[float, bool]:
    if regular.empty or len(regular) < 5 or "Volume" not in regular:
        return 0.0, False
    last_two_avg = float(regular["Volume"].tail(2).mean())
    previous = regular["Volume"].iloc[:-2].tail(12)
    previous_avg = float(previous.mean()) if not previous.empty else 0.0
    if previous_avg <= 0:
        return 0.0, False
    ratio = last_two_avg / previous_avg
    return round(ratio, 2), ratio >= threshold


def detect_price_compression(regular: pd.DataFrame, day_high: float, current_price: float) -> tuple[float, bool]:
    if regular.empty or len(regular) < 5 or current_price <= 0:
        return 100.0, False
    recent = regular.tail(8)
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())
    compression_pct = ((recent_high - recent_low) / current_price) * 100
    near_high_pct = ((day_high - current_price) / current_price) * 100
    return round(compression_pct, 2), compression_pct <= 4.0 and near_high_pct <= 3.0


def detect_higher_lows(regular: pd.DataFrame) -> bool:
    if regular.empty or len(regular) < 6:
        return False
    lows = regular["Low"].tail(6).reset_index(drop=True)
    early_low = float(lows.iloc[:3].min())
    late_low = float(lows.iloc[3:].min())
    return late_low > early_low and float(lows.iloc[-1]) >= float(lows.iloc[-3])


def detect_halt_spike(regular: pd.DataFrame, current_price: float, threshold_pct: float) -> bool:
    if regular.empty or len(regular) < 3 or current_price <= 0:
        return False
    recent = regular.tail(4)
    recent_low = float(recent["Low"].min())
    recent_high = max(float(recent["High"].max()), current_price)
    move_pct = ((recent_high - recent_low) / recent_low) * 100 if recent_low > 0 else 0.0
    return move_pct >= threshold_pct


def has_spac_low_float_boost(ticker: str) -> bool:
    name = SECURITY_NAME_CACHE.get(str(ticker).upper(), "").lower()
    return any(term in name for term in ("acquisition", "capital", "holdings"))


def position_score(
    rvol: float,
    near_high_pct: float,
    volume_acceleration: float,
    compression_pct: float,
    tight_consolidation: bool,
    higher_lows: bool,
    halt_candidate: bool,
    spac_boost: bool,
    above_vwap_pct: float,
) -> float:
    rvol_score = min(max(rvol, 0.0), 12.0) * 12
    proximity_score = max(0.0, 3.0 - near_high_pct) * 14
    acceleration_score = min(max(volume_acceleration, 0.0), 5.0) * 10
    compression_score = max(0.0, 5.0 - compression_pct) * 8
    structure_score = (18 if tight_consolidation else 0) + (14 if higher_lows else 0)
    runner_score = 18 if halt_candidate else 0
    spac_score = 10 if spac_boost else 0
    vwap_score = max(0.0, min(above_vwap_pct, 5.0)) * 2
    return round(
        rvol_score
        + proximity_score
        + acceleration_score
        + compression_score
        + structure_score
        + runner_score
        + spac_score
        + vwap_score,
        2,
    )


def extract_candidate(ticker: str, state: AppState, settings: ScannerSettings) -> Optional[dict[str, object]]:
    if not is_valid_ticker(ticker):
        return None
    intraday = cached_download(ticker, state, settings)
    if intraday.empty:
        return None

    today, regular = latest_session_rows(intraday)
    analysis_rows = regular if not regular.empty else today
    if today.empty or analysis_rows.empty:
        return None

    current_price = float(today["Close"].iloc[-1])
    open_price = float(regular["Open"].iloc[0]) if not regular.empty else float(today["Open"].iloc[0])
    last_regular_price = float(regular["Close"].iloc[-1]) if not regular.empty else current_price
    previous_close = get_previous_close(ticker, state, settings)
    if previous_close is None:
        return None
    historical_average_volume = get_average_volume(ticker, state, settings)
    day_high = float(today["High"].max())
    day_low = float(today["Low"].min())
    volume = int(today["Volume"].sum())

    if volume < settings.min_volume:
        return None

    gap_pct = ((current_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0.0
    gain_pct = gap_pct
    premarket_gain_pct = gap_pct if gap_pct > 0 else 0.0
    after_hours_gain_pct = ((current_price - last_regular_price) / last_regular_price) * 100 if last_regular_price > 0 and len(today) > len(regular) else 0.0
    intraday_relative_volume = estimate_relative_volume(volume, analysis_rows)
    daily_relative_volume = volume / historical_average_volume if historical_average_volume and historical_average_volume > 0 else 0.0
    relative_volume = max(intraday_relative_volume, daily_relative_volume)
    near_high_pct = ((day_high - current_price) / current_price) * 100 if current_price > 0 else 100.0
    volume_acceleration, candle_volume_spike = detect_volume_acceleration(analysis_rows, settings.volume_acceleration_threshold)
    volume_spike = candle_volume_spike or relative_volume >= settings.min_relative_volume
    compression_pct, tight_consolidation = detect_price_compression(analysis_rows, day_high, current_price)
    higher_lows = detect_higher_lows(analysis_rows)
    halt_candidate = detect_halt_spike(analysis_rows, current_price, settings.halt_spike_threshold_pct)
    spac_boost = has_spac_low_float_boost(ticker)
    pre_breakout_setup = tight_consolidation or higher_lows
    vwap = calculate_vwap(analysis_rows)
    above_vwap_pct = ((current_price - vwap) / vwap) * 100 if vwap and not np.isnan(vwap) and vwap > 0 else 0.0
    premarket_runner = gap_pct >= 8.0
    high_price_runner_exception = (
        current_price > 20.0
        and gap_pct >= 8.0
        and premarket_gain_pct >= 5.0
        and volume_spike
        and relative_volume >= settings.min_relative_volume
    )

    price_in_range = current_price >= settings.min_price and current_price <= settings.max_price
    if not price_in_range and not high_price_runner_exception:
        return None

    early_momentum = (
        premarket_runner
        or
        relative_volume >= settings.min_relative_volume
        or near_high_pct <= settings.near_high_threshold_pct
        or volume_spike
        or pre_breakout_setup
        or halt_candidate
    )
    if not early_momentum:
        return None

    reason: list[str] = []
    if gap_pct >= 8.0:
        reason.append("gap up")
    if premarket_runner:
        reason.append("premarket runner")
    if volume_spike:
        reason.append("volume spike")
    if near_high_pct <= settings.near_high_threshold_pct:
        reason.append("near high")

    scan_reason: list[str] = []
    scan_reason.extend(reason)
    if near_high_pct <= settings.near_high_threshold_pct and (tight_consolidation or higher_lows):
        scan_reason.append("early breakout setup")
    if early_momentum:
        scan_reason.append("pre-momentum")
    if relative_volume >= settings.min_relative_volume:
        scan_reason.append(f"RVOL {relative_volume:.2f}x")
    if volume_spike:
        scan_reason.append("volume expansion")
    if near_high_pct <= settings.near_high_threshold_pct:
        scan_reason.append("near high pressure")
    if tight_consolidation:
        scan_reason.append("tight consolidation")
    if higher_lows:
        scan_reason.append("higher lows")
    if halt_candidate:
        scan_reason.append("halt candidate")
    if spac_boost:
        scan_reason.append("SPAC/low-float boost")
    if current_price >= vwap and day_low <= vwap:
        scan_reason.append("vwap reclaim setup")
    if settings.min_gain_pct > 0 and gain_pct >= settings.min_gain_pct:
        scan_reason.append(f"gain {gain_pct:.1f}%")
    scan_reason.append(f"volume {volume:,}")
    if not reason:
        reason.append("pre-momentum")

    return {
        "Ticker": ticker,
        "Current Price": round(current_price, 4),
        "Previous Close": round(previous_close, 4),
        "Open": round(open_price, 4),
        "Day High": round(day_high, 4),
        "Day Low": round(day_low, 4),
        "Gap %": round(gap_pct, 2),
        "Premarket Gain %": round(premarket_gain_pct, 2),
        "Intraday Gain %": round(gain_pct, 2),
        "After Hours Gain %": round(after_hours_gain_pct, 2),
        "Volume": volume,
        "Average Volume": int(historical_average_volume or float(analysis_rows["Volume"].tail(20).mean()) * 78),
        "Relative Volume": round(relative_volume, 2),
        "Near High %": round(near_high_pct, 2),
        "Volume Acceleration": volume_acceleration,
        "Volume Spike": volume_spike,
        "Compression %": compression_pct,
        "Tight Consolidation": tight_consolidation,
        "Higher Lows": higher_lows,
        "Pre-Breakout Setup": pre_breakout_setup,
        "Halt Candidate": halt_candidate,
        "SPAC Low Float Boost": spac_boost,
        "Premarket Runner": premarket_runner,
        "Runner Label": "PREMARKET RUNNER" if premarket_runner else "",
        "VWAP": vwap,
        "Above VWAP %": round(above_vwap_pct, 2),
        "Early Momentum": early_momentum,
        "Momentum Score": position_score(
            relative_volume,
            near_high_pct,
            volume_acceleration,
            compression_pct,
            tight_consolidation,
            higher_lows,
            halt_candidate,
            spac_boost,
            above_vwap_pct,
        ),
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Reason": ", ".join(reason),
        "Scan Reason": ", ".join(scan_reason),
    }


def scan_market(state: AppState, settings: ScannerSettings) -> tuple[pd.DataFrame, int, int]:
    tickers = build_scan_universe(state, settings)
    rows: list[dict[str, object]] = []
    skipped = 0
    workers = max(10, min(settings.max_workers, 25))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_candidate, ticker, state, settings): ticker for ticker in tickers}
        for future in as_completed(futures):
            try:
                candidate = future.result(timeout=0)
            except Exception:
                candidate = None
            if candidate is None:
                skipped += 1
            else:
                rows.append(candidate)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["Ticker"], keep="last")
        df["_runner_rank"] = np.where(df["Runner Label"].astype(str).eq("PREMARKET RUNNER"), 0, 1)
        df = df.sort_values(
            [
                "_runner_rank",
                "Early Momentum",
                "Gap %",
                "Relative Volume",
                "Near High %",
                "Volume Acceleration",
                "Compression %",
                "Momentum Score",
                "Volume",
            ],
            ascending=[True, False, False, False, True, False, True, False, False],
        ).drop(columns=["_runner_rank"]).head(settings.output_limit)
        blank_runner = df["Runner Label"].astype(str).eq("")
        top_blank_indexes = df[blank_runner].head(3).index
        df.loc[top_blank_indexes, "Runner Label"] = "HIGH POTENTIAL RUNNERS"
    return df, skipped, len(tickers)


def get_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def rr_ratio(entry: float, stop: float, target: float) -> Optional[float]:
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def price_distance_pct(current_price: float, entry: Optional[float]) -> float:
    if entry is None or entry <= 0:
        return 999.0
    return abs(current_price - entry) / entry * 100


def calculate_trade_plan(row: pd.Series) -> dict[str, object]:
    ticker = str(row["Ticker"]).upper()
    current_price = round_cent(get_value(row, "Current Price"))
    open_price = round_cent(get_value(row, "Open", current_price))
    day_high = round_cent(get_value(row, "Day High", current_price))
    day_low = round_cent(get_value(row, "Day Low", current_price))
    vwap = round_cent(get_value(row, "VWAP", current_price))
    gap_pct = get_value(row, "Gap %")
    premarket_gain_pct = get_value(row, "Premarket Gain %")
    gain_pct = get_value(row, "Intraday Gain %")
    after_hours_gain_pct = get_value(row, "After Hours Gain %")
    volume = int(get_value(row, "Volume"))
    relative_volume = get_value(row, "Relative Volume")
    near_high_pct = get_value(row, "Near High %", 100.0)
    volume_acceleration = get_value(row, "Volume Acceleration")
    compression_pct = get_value(row, "Compression %", 100.0)
    volume_spike = bool(row.get("Volume Spike", False))
    tight_consolidation = bool(row.get("Tight Consolidation", False))
    higher_lows = bool(row.get("Higher Lows", False))
    pre_breakout_setup = bool(row.get("Pre-Breakout Setup", False))
    halt_candidate = bool(row.get("Halt Candidate", False))
    spac_boost = bool(row.get("SPAC Low Float Boost", False))
    premarket_runner = bool(row.get("Premarket Runner", False))
    runner_label = str(row.get("Runner Label", "") or "")
    scan_reason = str(row.get("Reason", "") or row.get("Scan Reason", ""))

    support_low = round_cent(max(day_low, current_price * 0.94))
    support_high = round_cent(max(support_low + 0.01, current_price * 0.975))
    breakout_level = round_cent(max(day_high, current_price * 1.015))
    spike_high = round_cent(max(day_high, breakout_level, current_price * 1.06))

    distance_from_support_pct = ((current_price - support_high) / current_price) * 100 if current_price > 0 else 999
    distance_to_breakout_pct = ((breakout_level - current_price) / current_price) * 100 if current_price > 0 else 999
    extended_pct = ((current_price - support_high) / support_high) * 100 if support_high > 0 else 999
    distance_to_vwap_pct = abs(current_price - vwap) / current_price * 100 if current_price > 0 else 999

    near_support = 0 <= distance_from_support_pct <= 3.5
    near_breakout = -1.0 <= distance_to_breakout_pct <= 2.5
    strong_volume = relative_volume >= 1.2 or volume_spike or volume_acceleration >= 1.5
    price_above_open = current_price >= open_price
    price_above_vwap = current_price >= vwap
    vwap_reclaim_confirmed = price_above_vwap and distance_to_vwap_pct <= 2.5 and strong_volume
    after_hours_reclaim = after_hours_gain_pct > 0 and current_price >= min(open_price, vwap) and strong_volume
    bullish_momentum_gate = gain_pct >= 0 or price_above_open or vwap_reclaim_confirmed or after_hours_reclaim
    bearish_momentum = gain_pct < -2
    strongly_red = gain_pct < -3
    below_open_and_vwap = current_price < open_price and current_price < vwap
    recent_bounce_reclaim = near_support and (higher_lows or volume_spike or volume_acceleration >= 1.5) and bullish_momentum_gate
    near_vwap_reclaim = vwap_reclaim_confirmed

    if bearish_momentum:
        setup_type = "WEAK / NO LONG"
    elif below_open_and_vwap:
        setup_type = "WAIT FOR REVERSAL"
    elif premarket_runner:
        setup_type = "PREMARKET RUNNER"
    elif current_price < support_low:
        setup_type = "Fake breakdown reclaim"
    elif after_hours_reclaim:
        setup_type = "AFTER-HOURS RECLAIM"
    elif halt_candidate:
        setup_type = "Halt candidate"
    elif pre_breakout_setup or near_breakout:
        setup_type = "PRE-BREAKOUT"
    elif recent_bounce_reclaim:
        setup_type = "Pullback-to-support reclaim"
    elif near_vwap_reclaim:
        setup_type = "VWAP reclaim"
    elif extended_pct > 12:
        setup_type = "Wait for pullback"
    else:
        setup_type = "WAIT FOR REVERSAL"

    status = "WAIT"
    reason = ""
    buffer = max(round_cent(current_price * 0.015), 0.02)
    stop_buffer = max(round_cent(current_price * 0.025), 0.03)

    if bearish_momentum:
        status = "WAIT FOR REVERSAL"
        reason = "Needs reclaim above VWAP/day open before any long."
    elif below_open_and_vwap:
        status = "WAIT FOR REVERSAL"
        reason = "Price is below both day open and VWAP; no active long until reclaim confirms."
    elif not bullish_momentum_gate:
        status = "WEAK / NO LONG"
        reason = "Bullish momentum gate failed: price is not green, above open, reclaiming VWAP, or holding positive after-hours action."
    elif current_price < support_low:
        status = "NO TRADE"
        reason = f"Price is below support low at {support_low:.2f}; long structure is broken."
    elif extended_pct > 12 and setup_type not in {"PRE-BREAKOUT", "Halt candidate", "PREMARKET RUNNER", "AFTER-HOURS RECLAIM"}:
        status = "WAIT FOR PULLBACK"
        reason = f"Price is {extended_pct:.1f}% above support, so buying now would chase the move."
    elif support_high < current_price < breakout_level and not near_vwap_reclaim:
        status = "WAIT"
        reason = f"Only valid if price reclaims {breakout_level:.2f} and holds."

    if setup_type == "VWAP reclaim":
        entry = round_cent(vwap + 0.01)
        stop = round_cent(min(vwap - buffer, support_low - 0.01))
        confirmation = f"Active only while price holds reclaimed VWAP at {vwap:.2f} with RVOL above 1.2x."
        invalidation = f"No long if VWAP reclaim fails and price closes below {vwap:.2f}."
    elif setup_type in {"PRE-BREAKOUT", "Halt candidate", "PREMARKET RUNNER"}:
        entry = round_cent(breakout_level + 0.02)
        stop = round_cent(breakout_level - buffer)
        confirmation = f"Only valid if price reclaims {breakout_level:.2f} and holds."
        invalidation = f"No long if breakout over {breakout_level:.2f} rejects back below it."
    elif setup_type == "AFTER-HOURS RECLAIM":
        entry = round_cent(max(current_price, vwap + 0.01, open_price + 0.01))
        stop = round_cent(min(vwap, open_price) - buffer)
        confirmation = f"Active only if after-hours gain stays positive and price holds above VWAP/day open."
        invalidation = f"No long if price loses VWAP {vwap:.2f} or day open {open_price:.2f}."
    elif setup_type == "Pullback-to-support reclaim":
        entry = round_cent(max(current_price, support_high + 0.01))
        stop = round_cent(support_low - stop_buffer)
        confirmation = f"Active only if price holds reclaimed support near {support_high:.2f} with bullish candles and RVOL above 1.2x."
        invalidation = f"No long if reclaimed support fails below {support_low:.2f}."
    elif setup_type in {"WAIT FOR REVERSAL", "WEAK / NO LONG"}:
        entry = round_cent(max(vwap, open_price) + 0.01)
        stop = None
        confirmation = "Needs reclaim above VWAP/day open before any long."
        invalidation = f"No long while price remains below VWAP {vwap:.2f} and day open {open_price:.2f}."
    else:
        entry = round_cent(support_high + 0.01)
        stop = round_cent(support_low - stop_buffer)
        confirmation = f"Only valid if price reclaims {max(support_high, vwap, open_price):.2f} and holds."
        invalidation = f"No long if price loses {support_low:.2f}."

    target_1 = round_cent(max(day_high, breakout_level))
    target_2 = round_cent(max(spike_high, target_1))
    risk = entry - stop if stop is not None else 0
    if risk > 0 and target_1 <= entry:
        target_1 = round_cent(entry + risk * 2)
    if risk > 0 and target_2 <= target_1:
        target_2 = round_cent(target_1 + risk * 2)

    rr = rr_ratio(entry, stop, target_1) if entry is not None and stop is not None and target_1 is not None else None
    entry_distance_pct = price_distance_pct(current_price, entry)
    data_consistent = entry_distance_pct <= 20
    price_near_entry = entry_distance_pct <= 10

    if not data_consistent:
        status = "INVALID DATA"
        setup_type = "INVALID DATA"
        reason = f"Current price is {entry_distance_pct:.1f}% away from planned entry, so this setup is based on stale or inconsistent data."
        confirmation = "Refresh scanner data before considering any trade."
        invalidation = "Trade plan hidden because current price and entry are inconsistent."
    elif current_price < support_low:
        status = "NO TRADE"
        reason = f"Price is below support low at {support_low:.2f}; long structure is broken."
        confirmation = f"No long while price is below support low {support_low:.2f}."
    elif bearish_momentum:
        status = "WAIT FOR REVERSAL"
        reason = "Needs reclaim above VWAP/day open before any long."
        confirmation = "Needs reclaim above VWAP/day open before any long."
    elif not price_near_entry:
        status = "WAIT"
        reason = f"Current price is {entry_distance_pct:.1f}% away from planned entry; wait for price to get within 10% of entry."
        confirmation = f"Only valid if price reclaims {entry:.2f} and holds near the trigger."

    if status == "WAIT" and setup_type in {"VWAP reclaim", "Pullback-to-support reclaim", "AFTER-HOURS RECLAIM"}:
        status = "VALID TRADE"
    if status == "VALID TRADE" and not (price_near_entry and current_price >= support_low and data_consistent and gain_pct >= 0):
        status = "WAIT FOR REVERSAL" if bearish_momentum else "WAIT"
        reason = "Valid-trade gate failed: price must be near entry, above support, data-consistent, and non-negative momentum."
        confirmation = f"Only valid if price holds above {support_low:.2f} and trades near entry {entry:.2f}."
    if status == "VALID TRADE" and (rr is None or rr < 2):
        status = "WAIT"
        reason = "Risk/reward to Target 1 is below 1:2."
        confirmation = f"Only valid if price reclaims {entry:.2f} and holds."
    if relative_volume < 1.2 and status == "VALID TRADE":
        status = "WAIT"
        reason = "Relative volume is below the 1.2x minimum for a valid long."
        confirmation = f"Only valid if price reclaims {entry:.2f} and holds with RVOL above 1.2x."
    if bearish_momentum and status == "VALID TRADE":
        status = "WAIT FOR REVERSAL"
        reason = "Needs reclaim above VWAP/day open before any long."
        confirmation = "Needs reclaim above VWAP/day open before any long."
    if below_open_and_vwap and status == "VALID TRADE":
        status = "WAIT FOR REVERSAL"
        reason = "Price is below day open and VWAP; reclaim is not confirmed."
        confirmation = "Needs reclaim above VWAP/day open before any long."
    if status in {"NO TRADE", "WEAK / NO LONG", "INVALID DATA"}:
        entry = stop = target_1 = target_2 = None

    probability = "Low"
    if status == "VALID TRADE" and rr is not None and rr >= 2:
        probability = "High" if relative_volume >= 2.5 and (volume_spike or near_high_pct <= 3.0) else "Medium"
    elif status.startswith("WAIT") and not bearish_momentum and relative_volume >= 1.5:
        probability = "Medium"
    if bearish_momentum or status == "INVALID DATA":
        probability = "Low"

    score = 0.0
    score += min(relative_volume, 12) * 12
    score += max(0.0, 3.0 - near_high_pct) * 14
    score += min(max(volume_acceleration, 0), 5) * 10
    score += max(0.0, 5.0 - compression_pct) * 8
    score += min(volume / 1_000_000, 10) * 4
    score += min(rr or 0, 6) * 8
    score += 18 if tight_consolidation else 0
    score += 14 if higher_lows else 0
    score += 18 if halt_candidate else 0
    score += 24 if premarket_runner else 0
    score += 10 if spac_boost else 0
    score += 10 if near_vwap_reclaim else 0
    score += 10 if near_breakout else 0
    score += 20 if status == "VALID TRADE" else 5 if status.startswith("WAIT") else -20
    if gain_pct < 0:
        score -= 30
    if gain_pct < -3:
        score -= 50
    if not price_above_open:
        score -= 20
    if relative_volume < 1:
        score -= 20
    if extended_pct > 12:
        score -= min(extended_pct - 12, 30) * 1.5
    score = round(max(score, 0), 1)

    why = reason or (
        f"{ticker} is showing pre-breakout behavior: RVOL {relative_volume:.2f}x, "
        f"{near_high_pct:.2f}% from the day high, {volume_acceleration:.2f}x candle volume acceleration, "
        f"and {compression_pct:.2f}% recent price compression."
    )
    avoid = f"Avoid if price loses {support_low:.2f}, relative volume fades, price gets more than 12% extended, or breakout level {breakout_level:.2f} rejects."
    rr_text = "N/A" if entry is None or stop is None or target_1 is None or rr is None else f"1:{rr:.2f}"

    return {
        "Ticker": ticker,
        "Status": status,
        "Setup Type": setup_type,
        "Current Price": current_price,
        "Gap %": round(gap_pct, 2),
        "Premarket Gain %": round(premarket_gain_pct, 2),
        "Intraday Gain %": round(gain_pct, 2),
        "Volume": volume,
        "Relative Volume": round(relative_volume, 2),
        "Entry": entry,
        "Stop": stop,
        "Target 1": target_1,
        "Target 2": target_2,
        "Risk/Reward": rr_text,
        "Probability": probability,
        "Momentum Score": score,
        "Entry Distance %": round(entry_distance_pct, 2),
        "Data Consistent": data_consistent,
        "Price Near Entry": price_near_entry,
        "Runner Label": runner_label,
        "Reason": scan_reason,
        "Near High %": round(near_high_pct, 2),
        "Volume Acceleration": round(volume_acceleration, 2),
        "Compression %": round(compression_pct, 2),
        "Volume Spike": volume_spike,
        "Pre-Breakout Setup": pre_breakout_setup,
        "Halt Candidate": halt_candidate,
        "Confirmation": confirmation,
        "Invalidation": invalidation,
        "Why This Works": why,
        "Avoid Trade": avoid,
        "Day High": day_high,
        "VWAP": vwap,
        "Last Updated": datetime.now().isoformat(timespec="seconds"),
    }


def analyze_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame()
    plans = [calculate_trade_plan(row) for _, row in candidates_df.iterrows()]
    df = pd.DataFrame(plans)
    if df.empty:
        return df
    if "Runner Label" in df.columns:
        runner_labels = df["Runner Label"].astype(str)
        df["_runner_rank"] = np.select(
            [runner_labels.eq("PREMARKET RUNNER"), runner_labels.eq("HIGH POTENTIAL RUNNERS")],
            [0, 1],
            default=2,
        )
    else:
        df["_runner_rank"] = 1
    status_rank = {
        "VALID TRADE": 0,
        "WAIT": 1,
        "WAIT FOR PULLBACK": 2,
        "WAIT FOR REVERSAL": 3,
        "WEAK / NO LONG": 4,
        "INVALID DATA": 5,
        "NO TRADE": 6,
    }
    if "Status" in df.columns:
        df["_status_rank"] = df["Status"].replace(status_rank).where(df["Status"].isin(status_rank), 9)
    else:
        df["_status_rank"] = 9
    return df.sort_values(["_runner_rank", "_status_rank", "Momentum Score"], ascending=[True, True, False]).drop(columns=["_runner_rank", "_status_rank"])


def scanner_loop(state: AppState) -> None:
    while not state.stop_event.is_set():
        with state.lock:
            settings = state.settings
            state.last_scan_started = datetime.now().isoformat(timespec="seconds")
            state.error = None

        started = time.monotonic()
        try:
            candidates_df, skipped, scanned = scan_market(state, settings)
            plans_df = analyze_candidates(candidates_df)
            elapsed = time.monotonic() - started
            with state.lock:
                state.candidates_df = candidates_df
                state.plans_df = plans_df
                state.scanned_count = scanned
                state.skipped_count = skipped
                state.last_scan_seconds = elapsed
                state.last_scan_finished = datetime.now().isoformat(timespec="seconds")
                state.cycle_count += 1
                state.error = None
        except Exception as exc:
            with state.lock:
                state.error = f"{type(exc).__name__}: {exc}"

        sleep_until = time.monotonic() + max(1, settings.scan_interval_seconds)
        while time.monotonic() < sleep_until and not state.stop_event.is_set():
            time.sleep(0.2)


def ensure_scanner_running(state: AppState) -> None:
    with state.lock:
        thread = state.scanner_thread
        if thread and thread.is_alive():
            return
        state.stop_event.clear()
        thread = threading.Thread(target=scanner_loop, args=(state,), daemon=True, name="momentum-scanner")
        state.scanner_thread = thread
        thread.start()


def restart_scanner(state: AppState, settings: ScannerSettings) -> None:
    with state.lock:
        state.settings = settings
        old_thread = state.scanner_thread
        state.stop_event.set()
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=1.5)
    with state.lock:
        state.stop_event = threading.Event()
        state.candidates_df = pd.DataFrame()
        state.plans_df = pd.DataFrame()
        state.error = None
        state.scanner_thread = threading.Thread(target=scanner_loop, args=(state,), daemon=True, name="momentum-scanner")
        state.scanner_thread.start()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #090d14; color: #e5e7eb; }
            [data-testid="stSidebar"] { background: #0d1420; }
            .block-container { padding-top: 1.35rem; }
            h1, h2, h3 { color: #f8fafc; letter-spacing: 0; }
            .subtle { color: #94a3b8; font-size: 0.9rem; }
            div[data-testid="stMetric"] {
                background: #111827;
                border: 1px solid #243244;
                border-radius: 8px;
                padding: 0.7rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_setup_card(row: pd.Series) -> None:
    with st.container(border=True):
        title_cols = st.columns([0.32, 0.38, 0.30])
        title_cols[0].markdown(f"**{safe_format_value(row.get('Ticker', 'N/A'))}**")
        title_cols[1].markdown(status_icon(str(row.get("Status", "WAIT"))))
        runner_label = str(row.get("Runner Label", "") or "")
        if runner_label:
            title_cols[2].markdown(f":orange[{runner_label}]")
        else:
            title_cols[2].caption(str(row.get("Setup Type", "WAIT")))

        metric_cols = st.columns(4)
        metric_cols[0].metric("Price", f"${fmt_price(row.get('Current Price'))}")
        metric_cols[1].metric("Gain", f"{float(row.get('Intraday Gain %', 0)):.2f}%")
        metric_cols[2].metric("RVOL", f"{float(row.get('Relative Volume', 0)):.2f}x")
        metric_cols[3].metric("Score", f"{safe_format_value(row.get('Momentum Score'))}")

        signal_cols = st.columns(3)
        signal_cols[0].caption(f"Accel {float(row.get('Volume Acceleration', 0)):.2f}x")
        signal_cols[1].caption(f"Near high {float(row.get('Near High %', 0)):.2f}%")
        signal_cols[2].caption(f"Volume {fmt_num(row.get('Volume'))}")

        trade_cols = st.columns(4)
        trade_cols[0].caption(f"Entry {fmt_price(row.get('Entry'))}")
        trade_cols[1].caption(f"Stop {fmt_price(row.get('Stop'))}")
        trade_cols[2].caption(f"T1 {fmt_price(row.get('Target 1'))}")
        trade_cols[3].caption(f"T2 {fmt_price(row.get('Target 2'))}")


def snapshot_state(state: AppState) -> dict[str, object]:
    with state.lock:
        return {
            "settings": state.settings,
            "candidates_df": state.candidates_df.copy(),
            "plans_df": state.plans_df.copy(),
            "last_scan_started": state.last_scan_started,
            "last_scan_finished": state.last_scan_finished,
            "last_scan_seconds": state.last_scan_seconds,
            "scanned_count": state.scanned_count,
            "skipped_count": state.skipped_count,
            "cycle_count": state.cycle_count,
            "error": state.error,
            "thread_alive": bool(state.scanner_thread and state.scanner_thread.is_alive()),
        }


def sync_session_state(snapshot: dict[str, object]) -> None:
    st.session_state["scanner_results"] = snapshot["candidates_df"]
    st.session_state["trade_plans"] = snapshot["plans_df"]
    st.session_state["scanner_metrics"] = {
        "thread_alive": snapshot["thread_alive"],
        "cycle_count": snapshot["cycle_count"],
        "scanned_count": snapshot["scanned_count"],
        "candidate_count": len(snapshot["candidates_df"]),  # type: ignore[arg-type]
        "last_scan_seconds": snapshot["last_scan_seconds"],
        "last_scan_finished": snapshot["last_scan_finished"],
        "error": snapshot["error"],
    }
    st.session_state["last_update_time"] = datetime.now().isoformat(timespec="seconds")


def remove_html_tags(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"<[^>]*>", "", value)
    return value


def safe_format_value(value: object) -> object:
    value = remove_html_tags(value)
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return round(value, 2)
    return value


def render_dynamic_sections(
    state: AppState,
    metrics_placeholder: st.delta_generator.DeltaGenerator,
    status_placeholder: st.delta_generator.DeltaGenerator,
    table_placeholder: st.delta_generator.DeltaGenerator,
    cards_placeholder: st.delta_generator.DeltaGenerator,
    detail_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    snap = snapshot_state(state)
    sync_session_state(snap)

    metrics_data = st.session_state["scanner_metrics"]
    plans_df: pd.DataFrame = st.session_state["trade_plans"]
    candidates_df: pd.DataFrame = st.session_state["scanner_results"]

    with metrics_placeholder.container():
        metrics = st.columns(5)
        metrics[0].metric("Scanner", "RUNNING" if metrics_data["thread_alive"] else "STOPPED")
        metrics[1].metric("Cycles", f"{int(metrics_data['cycle_count']):,}")
        metrics[2].metric("Scanned", f"{int(metrics_data['scanned_count']):,}")
        metrics[3].metric("Candidates", f"{len(candidates_df):,}")
        metrics[4].metric("Last Scan", f"{float(metrics_data['last_scan_seconds']):.1f}s")

    with status_placeholder.container():
        st.caption(f"Last updated: {st.session_state['last_update_time']} | Last scan finished: {metrics_data['last_scan_finished'] or 'Starting...'}")
        if metrics_data["error"]:
            st.warning(f"Last scanner error: {metrics_data['error']}")

    if plans_df.empty:
        table_placeholder.info("No trade setups available yet. Scanner is still collecting data.")
        cards_placeholder.empty()
        detail_placeholder.empty()
        return

    top10 = plans_df.head(10)
    if top10.empty:
        table_placeholder.info("No trade setups available yet. Scanner is still collecting data.")
        cards_placeholder.empty()
        detail_placeholder.empty()
        return

    display_cols = [
        "Runner Label",
        "Ticker",
        "Status",
        "Setup Type",
        "Reason",
        "Current Price",
        "Gap %",
        "Premarket Gain %",
        "Intraday Gain %",
        "Relative Volume",
        "Near High %",
        "Volume Acceleration",
        "Compression %",
        "Volume Spike",
        "Entry",
        "Stop",
        "Target 1",
        "Target 2",
        "Momentum Score",
        "Entry Distance %",
        "Data Consistent",
        "Price Near Entry",
    ]

    try:
        available_cols = [col for col in display_cols if col in top10.columns]
        if not available_cols:
            table_placeholder.info("No trade setups available yet. Scanner is still collecting data.")
            cards_placeholder.empty()
            detail_placeholder.empty()
            return

        table_df = top10[available_cols].copy()
        for col in table_df.columns:
            table_df[col] = table_df[col].apply(safe_format_value)

        table_placeholder.dataframe(table_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        table_placeholder.warning(f"Could not format trade setup table yet: {type(exc).__name__}. Scanner is still collecting data.")
        cards_placeholder.empty()
        detail_placeholder.empty()
        return

    with cards_placeholder.container():
        for _, row in top10.iterrows():
            render_setup_card(row)

    selected_ticker = st.session_state.get("selected_ticker")
    if "Ticker" not in top10.columns:
        detail_placeholder.info("No trade setup details available yet.")
        return

    if selected_ticker not in set(top10["Ticker"].astype(str)):
        selected_ticker = str(top10.iloc[0]["Ticker"])

    row = top10[top10["Ticker"].astype(str) == selected_ticker].iloc[0]
    with detail_placeholder.container():
        if "Status" in top10.columns:
            st.markdown(status_icon(str(row.get("Status", "WAIT"))))
        else:
            st.markdown(status_icon("WAIT"))
        trade_cols = st.columns(4)
        trade_cols[0].metric("Entry", fmt_price(row.get("Entry")))
        trade_cols[1].metric("Stop", fmt_price(row.get("Stop")))
        trade_cols[2].metric("Target 1", fmt_price(row.get("Target 1")))
        trade_cols[3].metric("Target 2", fmt_price(row.get("Target 2")))
        st.write(f"**Confirmation:** {safe_format_value(row.get('Confirmation'))}")
        st.write(f"**Invalidation:** {safe_format_value(row.get('Invalidation'))}")
        st.write(f"**Why this works:** {safe_format_value(row.get('Why This Works'))}")
        st.write(f"**Avoid trade:** {safe_format_value(row.get('Avoid Trade'))}")


def main() -> None:
    st.set_page_config(page_title="Live Momentum Trade Setups", layout="wide")
    apply_theme()

    state = get_state()
    ensure_scanner_running(state)
    snap = snapshot_state(state)
    sync_session_state(snap)
    settings: ScannerSettings = snap["settings"]  # type: ignore[assignment]

    st.title("Live Small-Cap Momentum Trade Setups")
    st.caption("Background scanner runs automatically. Data is held in memory and refreshes in the browser.")

    with st.sidebar:
        st.header("Controls")
        fast_mode = st.toggle("FAST MODE", value=settings.fast_mode)
        scan_interval = st.slider("Scan interval seconds", min_value=5, max_value=30, value=int(settings.scan_interval_seconds), step=1)
        max_tickers = st.slider("Tickers per scan", min_value=100, max_value=1500, value=int(settings.max_tickers), step=50)
        max_price = st.slider("Maximum price", min_value=1.0, max_value=200.0, value=float(settings.max_price), step=1.0)
        min_volume = st.number_input("Minimum volume", min_value=0, value=int(settings.min_volume), step=25_000)
        max_workers = st.slider("Workers", min_value=10, max_value=25, value=int(settings.max_workers), step=1)
        live_updates = st.toggle("Live placeholder updates", value=True)
        pause_updates = st.toggle("Pause Updates", value=False)
        ui_update_seconds = st.slider("UI update seconds", min_value=1, max_value=10, value=2, step=1)

        new_settings = ScannerSettings(
            fast_mode=fast_mode,
            scan_interval_seconds=scan_interval,
            max_price=float(max_price),
            min_volume=int(min_volume),
            max_tickers=int(max_tickers),
            max_workers=int(max_workers),
        )

        if st.button("Restart scanner", type="primary", use_container_width=True):
            restart_scanner(state, new_settings)

        with state.lock:
            state.settings = new_settings

    metrics_placeholder = st.empty()
    status_placeholder = st.empty()
    st.subheader("Top 10 Momentum Stocks")
    table_col, cards_col = st.columns([0.58, 0.42])
    with table_col:
        table_placeholder = st.empty()
    with cards_col:
        cards_placeholder = st.empty()

    plans_for_selector: pd.DataFrame = st.session_state.get("trade_plans", pd.DataFrame())
    if not plans_for_selector.empty and "Ticker" in plans_for_selector.columns:
        tickers = plans_for_selector.head(10)["Ticker"].astype(str).tolist()
        current_selected = st.session_state.get("selected_ticker")
        if current_selected in tickers:
            selected_index = tickers.index(current_selected)
        else:
            selected_index = 0
        st.selectbox("Open trade plan", tickers, index=selected_index, key="selected_ticker")
    else:
        st.selectbox("Open trade plan", ["Waiting for data"], disabled=True)

    detail_placeholder = st.empty()
    render_dynamic_sections(
        state,
        metrics_placeholder,
        status_placeholder,
        table_placeholder,
        cards_placeholder,
        detail_placeholder,
    )

    if live_updates and not pause_updates:
        while True:
            render_dynamic_sections(
                state,
                metrics_placeholder,
                status_placeholder,
                table_placeholder,
                cards_placeholder,
                detail_placeholder,
            )
            time.sleep(ui_update_seconds)


if __name__ == "__main__":
    main()
