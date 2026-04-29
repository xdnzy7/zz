from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
import io
from io import StringIO
import logging
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


CANDIDATES_FILE = Path("momentum_candidates.xlsx")
LOG_FILE = Path("scanner.log")
CACHE_DIR = Path(".yfinance_cache")
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
LOGGER = logging.getLogger("scanner")

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

KNOWN_BAD_TICKERS = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "TEST",
}

_download_cache: dict[tuple[str, str, str, bool], tuple[float, pd.DataFrame]] = {}
_market_cap_cache: dict[str, tuple[float, Optional[float]]] = {}
_ticker_universe_cache: tuple[float, list[str]] | None = None
_ticker_name_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


@dataclass
class ScannerConfig:
    min_price: float = 0.50
    max_price: float = 20.00
    min_gain_pct: float = 0.0
    min_volume: int = 150_000
    min_relative_volume: float = 1.3
    near_high_threshold_pct: float = 3.0
    volume_acceleration_threshold: float = 1.5
    halt_spike_threshold_pct: float = 15.0
    max_market_cap: int = 2_000_000_000
    batch_size: int = 40
    max_workers: int = 20
    max_tickers: int = 500
    active_movers_count: int = 200
    explore_tickers_count: int = 300
    output_limit: int = 100
    fast_mode: bool = True
    cache_seconds: int = 20
    save_interval_seconds: int = 20
    ticker_timeout_seconds: int = 4
    batch_timeout_seconds: int = 6
    universe_cache_seconds: int = 300
    include_market_cap_check: bool = False
    stream_results: bool = True
    scan_interval_seconds: int = 5


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_request_text(url: str, timeout: int = 5) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def is_valid_ticker(ticker: str) -> bool:
    ticker = str(ticker).upper().strip()
    return bool(VALID_TICKER_RE.fullmatch(ticker)) and ticker not in KNOWN_BAD_TICKERS


def get_us_tickers() -> pd.DataFrame:
    """Download NASDAQ/NYSE/AMEX symbols from public NASDAQ Trader files."""
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
        return pd.DataFrame({"Ticker": FALLBACK_TICKERS, "Name": "", "Exchange": "Fallback", "ETF": "N", "Test Issue": "N"})

    if not frames:
        return pd.DataFrame({"Ticker": FALLBACK_TICKERS, "Name": "", "Exchange": "Fallback", "ETF": "N", "Test Issue": "N"})
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])


def clean_ticker_list(ticker_df: pd.DataFrame) -> list[str]:
    """Remove non-common-stock instruments and symbols likely to fail data fetches."""
    if ticker_df.empty:
        return FALLBACK_TICKERS

    df = ticker_df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].fillna("").astype(str).str.lower()
    df = df[df["Ticker"].map(is_valid_ticker)]
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

    preferred_exchanges = {"NASDAQ", "NYSE", "NYSE American", "NYSE Arca"}
    if "Fallback" not in set(df["Exchange"]):
        df = df[df["Exchange"].isin(preferred_exchanges)]

    cleaned = df["Ticker"].drop_duplicates().tolist()
    return cleaned if cleaned else FALLBACK_TICKERS


def get_clean_universe(config: ScannerConfig) -> list[str]:
    global _ticker_name_cache
    global _ticker_universe_cache
    now = time.monotonic()
    with _cache_lock:
        if _ticker_universe_cache and now - _ticker_universe_cache[0] <= config.universe_cache_seconds:
            return list(_ticker_universe_cache[1])

    ticker_df = get_us_tickers()
    if not ticker_df.empty and {"Ticker", "Name"}.issubset(ticker_df.columns):
        names = ticker_df.copy()
        names["Ticker"] = names["Ticker"].astype(str).str.upper().str.strip()
        names["Name"] = names["Name"].fillna("").astype(str)
        _ticker_name_cache = dict(zip(names["Ticker"], names["Name"]))
    universe = clean_ticker_list(ticker_df)
    with _cache_lock:
        _ticker_universe_cache = (now, universe)
    return list(universe)


def get_active_movers(max_count: int = 250) -> list[str]:
    """Seed scans with Yahoo gainers, but never depend on this list alone."""
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"scrIds": "day_gainers", "count": min(max_count, 250)}
    try:
        response = requests.get(YAHOO_GAINERS_URL, params=params, headers=headers, timeout=4)
        response.raise_for_status()
        quotes = response.json()["finance"]["result"][0]["quotes"]
    except Exception:
        return []

    tickers: list[str] = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper().strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        market = str(quote.get("market", "")).lower()
        if quote_type == "EQUITY" and market in {"us_market", ""} and is_valid_ticker(symbol):
            tickers.append(symbol)
    return list(dict.fromkeys(tickers))


def build_scan_universe(config: ScannerConfig) -> list[str]:
    all_tickers = get_clean_universe(config)
    all_set = set(all_tickers)
    active = [ticker for ticker in get_active_movers(config.active_movers_count) if ticker in all_set]

    explore_pool = [ticker for ticker in all_tickers if ticker not in set(active)]
    explore_size = min(len(explore_pool), config.explore_tickers_count)
    explore = random.sample(explore_pool, explore_size) if explore_size else []

    tickers = list(dict.fromkeys(active + explore))
    if len(tickers) < min(config.max_tickers, len(all_tickers)):
        remaining = [ticker for ticker in all_tickers if ticker not in set(tickers)]
        fill_size = min(len(remaining), config.max_tickers - len(tickers))
        if fill_size > 0:
            tickers.extend(random.sample(remaining, fill_size))
    return [ticker for ticker in tickers[: min(config.max_tickers, len(all_tickers))] if is_valid_ticker(ticker)]


def cached_download(ticker: str, config: ScannerConfig) -> pd.DataFrame:
    """Fast yfinance wrapper with short in-memory cache and hard per-request timeout."""
    key = (ticker, "1d", "5m", True)
    now = time.monotonic()
    with _cache_lock:
        cached = _download_cache.get(key)
        if cached and now - cached[0] <= config.cache_seconds:
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
                timeout=config.ticker_timeout_seconds,
            )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        df = pd.DataFrame()
    else:
        df = normalize_columns(df)
        df = df.dropna(subset=["Close"]) if "Close" in df else pd.DataFrame()

    with _cache_lock:
        _download_cache[key] = (now, df.copy())
    return df


def fetch_market_cap_cached(ticker: str, config: ScannerConfig) -> Optional[float]:
    now = time.monotonic()
    with _cache_lock:
        cached = _market_cap_cache.get(ticker)
        if cached and now - cached[0] <= max(config.cache_seconds, 30):
            return cached[1]

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            market_cap = yf.Ticker(ticker).fast_info.get("market_cap")
        value = float(market_cap) if market_cap else None
    except Exception:
        value = None

    with _cache_lock:
        _market_cap_cache[ticker] = (now, value)
    return value


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty or float(df["Volume"].sum()) <= 0:
        return np.nan
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return round(float((typical * df["Volume"]).sum() / df["Volume"].sum()), 4)


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
    compression_pct = ((recent_high - recent_low) / current_price) * 100 if current_price > 0 else 100.0
    near_high_pct = ((day_high - current_price) / current_price) * 100 if current_price > 0 else 100.0
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
    name = _ticker_name_cache.get(str(ticker).upper(), "").lower()
    return any(term in name for term in ("acquisition", "capital", "holdings"))


def calculate_position_score(
    relative_volume: float,
    near_high_pct: float,
    volume_acceleration: float,
    compression_pct: float,
    tight_consolidation: bool,
    higher_lows: bool,
    halt_candidate: bool,
    spac_boost: bool,
    above_vwap_pct: float,
) -> float:
    rvol_score = min(max(relative_volume, 0.0), 12.0) * 12
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


def latest_session_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    eastern = working.index.tz_convert("America/New_York")
    working["session_date"] = eastern.date
    today = working[working["session_date"] == working["session_date"].max()].copy()
    regular = today.between_time("09:30", "16:00")
    return today, regular if not regular.empty else today


def estimate_relative_volume(today_volume: int, regular: pd.DataFrame, config: ScannerConfig) -> float:
    if regular.empty:
        return 0.0
    bars_seen = max(len(regular), 1)
    expected_full_day_bars = 78
    projected_day_volume = today_volume * expected_full_day_bars / min(bars_seen, expected_full_day_bars)
    rolling_bar_volume = float(regular["Volume"].tail(20).mean()) if "Volume" in regular and not regular.empty else 0.0
    baseline = rolling_bar_volume * expected_full_day_bars
    if baseline <= 0:
        return 0.0
    return projected_day_volume / baseline


def extract_candidate(ticker: str, config: ScannerConfig) -> Optional[dict[str, object]]:
    ticker = str(ticker).upper().strip()
    if not is_valid_ticker(ticker):
        return None

    intraday = cached_download(ticker, config)
    if intraday.empty:
        return None

    today, regular = latest_session_rows(intraday)
    if today.empty or regular.empty:
        return None

    current_price = float(today["Close"].iloc[-1])
    open_price = float(regular["Open"].iloc[0])
    previous_close = open_price
    day_high = float(regular["High"].max())
    day_low = float(regular["Low"].min())
    volume = int(today["Volume"].sum())

    if not config.min_price <= current_price <= config.max_price:
        return None
    if volume < config.min_volume:
        return None

    avg_volume = int(float(regular["Volume"].tail(20).mean()) * 78) if "Volume" in regular and not regular.empty else 0
    relative_volume = estimate_relative_volume(volume, regular, config)
    intraday_gain_pct = ((current_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0.0
    near_high_pct = ((day_high - current_price) / current_price) * 100 if current_price > 0 else 100.0
    volume_acceleration, volume_spike = detect_volume_acceleration(regular, config.volume_acceleration_threshold)
    compression_pct, tight_consolidation = detect_price_compression(regular, day_high, current_price)
    higher_lows = detect_higher_lows(regular)
    halt_candidate = detect_halt_spike(regular, current_price, config.halt_spike_threshold_pct)
    spac_boost = has_spac_low_float_boost(ticker)
    pre_breakout_setup = tight_consolidation or higher_lows

    vwap = np.nan
    above_vwap_pct = 0.0
    if not config.fast_mode:
        vwap = calculate_vwap(regular)
        above_vwap_pct = ((current_price - vwap) / vwap) * 100 if vwap and not np.isnan(vwap) and vwap > 0 else 0.0

    early_momentum = (
        relative_volume >= config.min_relative_volume
        or near_high_pct <= config.near_high_threshold_pct
        or volume_spike
        or pre_breakout_setup
        or halt_candidate
    )

    if not early_momentum:
        return None

    market_cap = None
    if config.include_market_cap_check and not config.fast_mode:
        market_cap = fetch_market_cap_cached(ticker, config)
        if market_cap is not None and market_cap > config.max_market_cap:
            return None

    scan_reason: list[str] = []
    if near_high_pct <= config.near_high_threshold_pct and (tight_consolidation or higher_lows):
        scan_reason.append("early breakout setup")
    if early_momentum:
        scan_reason.append("pre-momentum")
    if relative_volume >= config.min_relative_volume:
        scan_reason.append(f"RVOL {relative_volume:.2f}x")
    if volume_spike:
        scan_reason.append("volume expansion")
    if near_high_pct <= config.near_high_threshold_pct:
        scan_reason.append("near high pressure")
    if tight_consolidation:
        scan_reason.append("tight consolidation")
    if higher_lows:
        scan_reason.append("higher lows")
    if halt_candidate:
        scan_reason.append("halt candidate")
    if spac_boost:
        scan_reason.append("SPAC/low-float boost")
    if not config.fast_mode and vwap and not np.isnan(vwap) and current_price >= vwap and day_low <= vwap:
        scan_reason.append("vwap reclaim setup")
    if config.min_gain_pct > 0 and intraday_gain_pct >= config.min_gain_pct:
        scan_reason.append(f"gain {intraday_gain_pct:.1f}%")
    scan_reason.append(f"volume {volume:,}")

    return {
        "Ticker": ticker,
        "Current Price": round(current_price, 4),
        "Previous Close": round(previous_close, 4),
        "Open": round(open_price, 4),
        "Day High": round(day_high, 4),
        "Day Low": round(day_low, 4),
        "Intraday Gain %": round(intraday_gain_pct, 2),
        "Volume": volume,
        "Average Volume": avg_volume,
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
        "Runner Label": "",
        "Market Cap": market_cap,
        "VWAP": vwap,
        "Above VWAP %": round(above_vwap_pct, 2),
        "Early Momentum": early_momentum,
        "Momentum Score": calculate_position_score(
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
        "Scan Reason": ", ".join(scan_reason),
    }


def relaxed_configs(config: ScannerConfig) -> list[ScannerConfig]:
    return [
        config,
        replace(config, min_gain_pct=0.0, min_volume=100_000, min_relative_volume=1.2),
        replace(config, min_gain_pct=0.0, min_volume=50_000, min_relative_volume=1.1),
    ]


def sort_candidates(rows: list[dict[str, object]], limit: int) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["Ticker"], keep="last")
    df = df.sort_values(
        [
            "Early Momentum",
            "Relative Volume",
            "Near High %",
            "Volume Acceleration",
            "Compression %",
            "Momentum Score",
            "Volume",
        ],
        ascending=[False, False, True, False, True, False, False],
    )
    df = df.head(limit).copy()
    df["Runner Label"] = ""
    df.iloc[: min(3, len(df)), df.columns.get_loc("Runner Label")] = "HIGH POTENTIAL RUNNERS"
    return df


def save_candidates(df: pd.DataFrame) -> None:
    columns = [
        "Ticker",
        "Current Price",
        "Previous Close",
        "Open",
        "Day High",
        "Day Low",
        "Intraday Gain %",
        "Volume",
        "Average Volume",
        "Relative Volume",
        "Near High %",
        "Volume Acceleration",
        "Volume Spike",
        "Compression %",
        "Tight Consolidation",
        "Higher Lows",
        "Pre-Breakout Setup",
        "Halt Candidate",
        "SPAC Low Float Boost",
        "Runner Label",
        "Market Cap",
        "VWAP",
        "Above VWAP %",
        "Early Momentum",
        "Momentum Score",
        "Timestamp",
        "Scan Reason",
    ]
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan
        df = df[columns]
    df.to_excel(CANDIDATES_FILE, index=False, engine="openpyxl")


def iter_batches(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def scan_config_stream(
    tickers: list[str],
    config: ScannerConfig,
    on_candidate: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    skipped = 0
    last_save = time.monotonic()
    workers = min(max(config.max_workers, 10), 25)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in iter_batches(tickers, config.batch_size):
            futures = {executor.submit(extract_candidate, ticker, config): ticker for ticker in batch}
            processed: set[object] = set()
            try:
                completed = as_completed(futures, timeout=config.batch_timeout_seconds)
                for future in completed:
                    processed.add(future)
                    ticker = futures[future]
                    try:
                        candidate = future.result(timeout=0)
                    except Exception:
                        candidate = None

                    if candidate is None:
                        skipped += 1
                    else:
                        rows.append(candidate)
                        if on_candidate:
                            on_candidate(candidate)

                    if rows and time.monotonic() - last_save >= config.save_interval_seconds:
                        save_candidates(sort_candidates(rows, config.output_limit))
                        last_save = time.monotonic()
            except TimeoutError:
                for future, ticker in futures.items():
                    if future in processed:
                        continue
                    if future.done():
                        try:
                            candidate = future.result(timeout=0)
                        except Exception:
                            candidate = None
                        if candidate is None:
                            skipped += 1
                        else:
                            rows.append(candidate)
                            if on_candidate:
                                on_candidate(candidate)
                    else:
                        future.cancel()
                        skipped += 1

            if config.stream_results and rows:
                top = sort_candidates(rows, min(5, config.output_limit))
                tickers_text = ", ".join(top["Ticker"].astype(str).tolist())
                print(f"Live top candidates: {tickers_text}")

    return sort_candidates(rows, config.output_limit), skipped


def scan_market(
    config: ScannerConfig,
    on_candidate: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[pd.DataFrame, int, int]:
    tickers = build_scan_universe(config)
    total_skipped = 0

    for active_config in relaxed_configs(config):
        df, skipped = scan_config_stream(tickers, active_config, on_candidate)
        total_skipped = skipped
        if not df.empty:
            save_candidates(df)
            return df, total_skipped, len(tickers)

    save_candidates(pd.DataFrame())
    return pd.DataFrame(), total_skipped, len(tickers)


def print_candidate(candidate: dict[str, object]) -> None:
    if not candidate.get("Early Momentum"):
        return
    ticker = candidate["Ticker"]
    price = candidate["Current Price"]
    gain = candidate["Intraday Gain %"]
    volume = candidate["Volume"]
    reason = candidate["Scan Reason"]
    label = candidate.get("Runner Label") or ""
    prefix = f"{label} | " if label else ""
    print(f"{prefix}{ticker}: ${price} | {gain}% | vol {volume:,} | {reason}")


def setup_logging(daemon: bool = False) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    if not daemon:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

    LOGGER.propagate = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous small-cap early momentum scanner.")
    parser.add_argument("--daemon", action="store_true", help="Run silently with file logging only.")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit.")
    parser.add_argument("--interval", type=int, default=ScannerConfig.scan_interval_seconds, help="Seconds between scan cycles.")
    parser.add_argument("--workers", type=int, default=ScannerConfig.max_workers, help="Parallel worker count, clamped to 10-25.")
    parser.add_argument("--batch-size", type=int, default=ScannerConfig.batch_size, help="Tickers per parallel batch.")
    parser.add_argument("--max-tickers", type=int, default=ScannerConfig.max_tickers, help="Maximum tickers per scan cycle.")
    parser.add_argument("--full-mode", action="store_true", help="Disable fast mode and calculate slower optional fields.")
    return parser


def config_from_args(args: argparse.Namespace) -> ScannerConfig:
    return ScannerConfig(
        batch_size=max(20, min(args.batch_size, 50)),
        max_workers=max(10, min(args.workers, 25)),
        max_tickers=max(1, args.max_tickers),
        fast_mode=not args.full_mode,
        stream_results=not args.daemon,
        scan_interval_seconds=max(1, args.interval),
    )


def run_scan_cycle(config: ScannerConfig, daemon: bool = False) -> tuple[pd.DataFrame, int, int, float]:
    mode = "FAST MODE" if config.fast_mode else "FULL MODE"
    LOGGER.info("Scan started (%s)", mode)
    started = time.monotonic()
    candidates, skipped, scanned = scan_market(config, None if daemon else print_candidate if config.stream_results else None)
    elapsed = time.monotonic() - started
    LOGGER.info(
        "Scan finished in %.1fs; scanned=%s skipped=%s candidates=%s output=%s",
        elapsed,
        scanned,
        skipped,
        len(candidates),
        CANDIDATES_FILE,
    )
    return candidates, skipped, scanned, elapsed


def run_continuous(config: ScannerConfig, daemon: bool = False) -> None:
    if not daemon:
        print("Scanner started (continuous mode)")
        print("Press CTRL+C to stop")
    LOGGER.info("Scanner service started; interval=%ss daemon=%s", config.scan_interval_seconds, daemon)

    while True:
        cycle_started = time.monotonic()
        try:
            run_scan_cycle(config, daemon=daemon)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Scan cycle failed; restarting next cycle")

        elapsed = time.monotonic() - cycle_started
        sleep_seconds = max(0.0, config.scan_interval_seconds - elapsed)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(daemon=args.daemon)
    config = config_from_args(args)

    try:
        if args.once:
            run_scan_cycle(config, daemon=args.daemon)
        else:
            run_continuous(config, daemon=args.daemon)
    except KeyboardInterrupt:
        LOGGER.info("Scanner stopped safely")
        if not args.daemon:
            print("Scanner stopped safely")


if __name__ == "__main__":
    main()
