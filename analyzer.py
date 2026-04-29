from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
import io
import logging
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


CANDIDATES_FILE = Path("momentum_candidates.xlsx")
ANALYZED_FILE = Path("analyzed_trade_plans.xlsx")
CACHE_DIR = Path(".yfinance_cache")
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@dataclass
class Levels:
    ticker: str
    current_price: float
    day_high: float
    day_low: float
    support_low: float
    support_high: float
    breakout_level: float
    spike_high: float
    vwap: float
    intraday_gain_pct: float
    volume: int
    average_volume: int
    relative_volume: float
    near_high_pct: float
    volume_acceleration: float
    compression_pct: float
    volume_spike: bool
    pre_breakout_setup: bool
    halt_candidate: bool
    runner_label: str
    market_cap: Optional[float]
    distance_from_support_pct: float
    distance_to_breakout_pct: float
    extended_pct: float
    near_support: bool
    near_breakout: bool
    near_vwap_reclaim: bool


def round_cent(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_download(ticker: str) -> pd.DataFrame:
    """Short intraday refresh only. Analyzer stays fast because candidate fundamentals are already in Excel."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                ticker,
                period="2d",
                interval="5m",
                prepost=True,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=4,
            )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_columns(df)
    return df.dropna(subset=["Close"]) if "Close" in df else pd.DataFrame()


def latest_session_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    eastern = working.index.tz_convert("America/New_York")
    working["session_date"] = eastern.date
    today = working[working["session_date"] == working["session_date"].max()].copy()
    regular = today.between_time("09:30", "16:00")
    return today, regular if not regular.empty else today


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty or float(df["Volume"].sum()) <= 0:
        return np.nan
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return round_cent(float((typical * df["Volume"]).sum() / df["Volume"].sum()))


def get_value(row: pd.Series, column: str, default: float = 0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def calculate_levels(row: pd.Series, intraday: pd.DataFrame) -> Optional[Levels]:
    ticker = str(row["Ticker"]).upper()
    current_price = get_value(row, "Current Price")
    day_high = get_value(row, "Day High")
    day_low = get_value(row, "Day Low")
    vwap = get_value(row, "VWAP", current_price)

    if intraday.empty:
        regular = pd.DataFrame()
    else:
        _, regular = latest_session_rows(intraday)

    if not regular.empty:
        current_price = round_cent(float(regular["Close"].iloc[-1]))
        day_high = round_cent(max(day_high, float(regular["High"].max())))
        day_low = round_cent(min(day_low, float(regular["Low"].min())))
        vwap = calculate_vwap(regular)
        recent = regular.tail(24) if len(regular) >= 24 else regular
        previous_bars = regular.iloc[:-1].tail(24) if len(regular) > 2 else regular
        support_low = round_cent(float(recent["Low"].quantile(0.20)))
        support_high = round_cent(float(recent["Low"].quantile(0.45)))
        breakout_level = round_cent(float(previous_bars["High"].max()))
        spike_high = round_cent(float(intraday["High"].tail(160).max()))
    else:
        support_low = round_cent(max(day_low, current_price * 0.94))
        support_high = round_cent(max(support_low + 0.01, current_price * 0.975))
        breakout_level = round_cent(max(day_high, current_price * 1.03))
        spike_high = round_cent(max(day_high, breakout_level))

    if pd.isna(vwap):
        vwap = current_price
    if support_high <= support_low:
        support_high = round_cent(support_low + max(current_price * 0.01, 0.01))

    intraday_gain_pct = get_value(row, "Intraday Gain %")
    volume = int(get_value(row, "Volume"))
    average_volume = int(get_value(row, "Average Volume"))
    relative_volume = get_value(row, "Relative Volume")
    near_high_pct = get_value(row, "Near High %", 100.0)
    volume_acceleration = get_value(row, "Volume Acceleration")
    compression_pct = get_value(row, "Compression %", 100.0)
    volume_spike = bool(row.get("Volume Spike", False))
    pre_breakout_setup = bool(row.get("Pre-Breakout Setup", False))
    halt_candidate = bool(row.get("Halt Candidate", False))
    runner_label = str(row.get("Runner Label", "") or "")
    market_cap_value = row.get("Market Cap")
    market_cap = None if pd.isna(market_cap_value) else float(market_cap_value)

    distance_from_support_pct = ((current_price - support_high) / current_price) * 100 if current_price > 0 else 999
    distance_to_breakout_pct = ((breakout_level - current_price) / current_price) * 100 if current_price > 0 else 999
    extended_pct = ((current_price - support_high) / support_high) * 100 if support_high > 0 else 999
    distance_to_vwap_pct = abs(current_price - vwap) / current_price * 100 if current_price > 0 else 999

    return Levels(
        ticker=ticker,
        current_price=round_cent(current_price),
        day_high=round_cent(day_high),
        day_low=round_cent(day_low),
        support_low=support_low,
        support_high=support_high,
        breakout_level=max(breakout_level, support_high + 0.01),
        spike_high=max(spike_high, day_high, breakout_level),
        vwap=round_cent(vwap),
        intraday_gain_pct=intraday_gain_pct,
        volume=volume,
        average_volume=average_volume,
        relative_volume=relative_volume,
        near_high_pct=near_high_pct,
        volume_acceleration=volume_acceleration,
        compression_pct=compression_pct,
        volume_spike=volume_spike,
        pre_breakout_setup=pre_breakout_setup,
        halt_candidate=halt_candidate,
        runner_label=runner_label,
        market_cap=market_cap,
        distance_from_support_pct=distance_from_support_pct,
        distance_to_breakout_pct=distance_to_breakout_pct,
        extended_pct=extended_pct,
        near_support=0 <= distance_from_support_pct <= 3.5,
        near_breakout=-1.0 <= distance_to_breakout_pct <= 2.5,
        near_vwap_reclaim=current_price >= vwap and distance_to_vwap_pct <= 2.5,
    )


def rr_ratio(entry: float, stop: float, target: float) -> Optional[float]:
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def setup_type_for(levels: Levels) -> str:
    if levels.halt_candidate:
        return "Halt candidate"
    if levels.pre_breakout_setup and levels.near_breakout:
        return "Early breakout setup"
    if levels.pre_breakout_setup:
        return "Pre-momentum"
    if levels.current_price < levels.support_low:
        return "Fake breakdown reclaim"
    if levels.near_support:
        return "Pullback-to-support reclaim"
    if levels.near_vwap_reclaim:
        return "VWAP reclaim"
    if levels.near_breakout:
        return "Early breakout setup"
    if levels.extended_pct > 12:
        return "Wait for pullback"
    return "Mid-range wait"


def momentum_score(levels: Levels, rr: Optional[float], status: str) -> float:
    score = 0.0
    score += min(levels.relative_volume, 12) * 12
    score += max(0.0, 3.0 - levels.near_high_pct) * 14
    score += min(max(levels.volume_acceleration, 0), 5) * 10
    score += max(0.0, 5.0 - levels.compression_pct) * 8
    score += min(levels.volume / 1_000_000, 10) * 4
    score += min(rr or 0, 6) * 8
    score += 18 if levels.pre_breakout_setup else 0
    score += 18 if levels.halt_candidate else 0
    score += 12 if levels.near_support else 0
    score += 10 if levels.near_vwap_reclaim else 0
    score += 10 if levels.near_breakout else 0
    score += 20 if status == "VALID TRADE" else 5 if status.startswith("WAIT") else -20
    if levels.extended_pct > 12:
        score -= min(levels.extended_pct - 12, 30) * 1.5
    return round(max(score, 0), 1)


def generate_trade_plan(levels: Levels) -> dict[str, object]:
    setup_type = setup_type_for(levels)
    status = "VALID TRADE"
    reason = ""
    buffer = max(round_cent(levels.current_price * 0.015), 0.02)
    stop_buffer = max(round_cent(levels.current_price * 0.025), 0.03)

    if levels.current_price < levels.support_low:
        status = "NO TRADE"
        reason = f"Price is below support low at {levels.support_low:.2f}; long structure is broken."
    elif levels.extended_pct > 12 and setup_type not in {"Early breakout setup", "Halt candidate"}:
        status = "WAIT FOR PULLBACK"
        reason = f"Price is {levels.extended_pct:.1f}% above support, so buying now would chase the move."
    elif levels.support_high < levels.current_price < levels.breakout_level and not levels.near_vwap_reclaim:
        status = "WAIT"
        reason = "Price is mid-range between support and breakout; wait for support reclaim or breakout hold."

    if setup_type == "Pullback-to-support reclaim":
        entry = round_cent(levels.support_high + 0.01)
        stop = round_cent(levels.support_low - stop_buffer)
        confirmation = f"5m reclaim and hold above {levels.support_high:.2f}, then bid support holds into {entry:.2f}."
        invalidation = f"No long if price accepts below {levels.support_low:.2f}."
    elif setup_type == "VWAP reclaim":
        entry = round_cent(levels.vwap + 0.01)
        stop = round_cent(min(levels.vwap - buffer, levels.support_low - 0.01))
        confirmation = f"5m candle reclaims VWAP at {levels.vwap:.2f} and next pullback holds above VWAP."
        invalidation = f"No long if VWAP reclaim fails and price closes below {levels.vwap:.2f}."
    elif setup_type in {"Early breakout setup", "Halt candidate"}:
        entry = round_cent(levels.breakout_level + 0.02)
        stop = round_cent(levels.breakout_level - buffer)
        confirmation = f"Break above {levels.breakout_level:.2f} with volume expansion, then retest holds."
        invalidation = f"No long if breakout over {levels.breakout_level:.2f} rejects back below it."
    elif setup_type == "Pre-momentum":
        entry = round_cent(max(levels.current_price, levels.support_high + 0.01))
        stop = round_cent(levels.support_low - stop_buffer)
        confirmation = f"Hold higher lows near {levels.support_high:.2f} and keep RVOL above 1.3x before attacking {levels.breakout_level:.2f}."
        invalidation = f"No long if compression breaks down below {levels.support_low:.2f} or volume expansion fades."
    elif setup_type == "Fake breakdown reclaim":
        entry = round_cent(levels.support_low + 0.02)
        stop = round_cent(levels.day_low - 0.01)
        confirmation = f"Failed breakdown below {levels.support_low:.2f}, then reclaim and hold above {entry:.2f}."
        invalidation = f"No long if price remains below {levels.support_low:.2f}."
    else:
        entry = round_cent(levels.support_high + 0.01)
        stop = round_cent(levels.support_low - stop_buffer)
        confirmation = f"Wait for pullback into {levels.support_low:.2f}-{levels.support_high:.2f} or breakout over {levels.breakout_level:.2f}."
        invalidation = f"No long if price loses {levels.support_low:.2f}."

    target_1 = round_cent(max(levels.day_high, levels.breakout_level))
    target_2 = round_cent(max(levels.spike_high, target_1))
    risk = entry - stop
    if risk > 0 and target_1 <= entry:
        target_1 = round_cent(entry + risk * 2)
    if risk > 0 and target_2 <= target_1:
        target_2 = round_cent(target_1 + risk * 2)

    rr = rr_ratio(entry, stop, target_1)
    if status == "VALID TRADE" and (rr is None or rr < 2):
        status = "WAIT"
        reason = "Risk/reward to Target 1 is below 1:2."
    if levels.relative_volume < 1.3 and not levels.volume_spike and status == "VALID TRADE":
        status = "WAIT"
        reason = "Relative volume is below the pre-momentum trigger and no candle volume spike is present."

    if status == "NO TRADE":
        entry = stop = target_1 = target_2 = None

    probability = "Low"
    if status == "VALID TRADE" and rr is not None and rr >= 2:
        probability = "High" if levels.relative_volume >= 2.5 and (levels.volume_spike or levels.near_high_pct <= 3.0) else "Medium"
    elif status.startswith("WAIT") and levels.relative_volume >= 1.5:
        probability = "Medium"

    score = momentum_score(levels, rr, status)
    why = reason or (
        f"{levels.ticker} is showing pre-breakout behavior: RVOL {levels.relative_volume:.2f}x, "
        f"{levels.near_high_pct:.2f}% from the day high, {levels.volume_acceleration:.2f}x candle volume acceleration, "
        f"and {levels.compression_pct:.2f}% recent price compression."
    )
    avoid = (
        f"Avoid if price loses {levels.support_low:.2f}, relative volume fades, price gets more than 12% extended, "
        f"or breakout level {levels.breakout_level:.2f} rejects."
    )

    rr_text = "N/A" if entry is None or stop is None or target_1 is None or rr is None else f"1:{rr:.2f}"

    return {
        "Ticker": levels.ticker,
        "Status": status,
        "Setup Type": setup_type,
        "Current Price": levels.current_price,
        "Entry": entry,
        "Stop": stop,
        "Target 1": target_1,
        "Target 2": target_2,
        "Risk/Reward": rr_text,
        "Probability": probability,
        "Momentum Score": score,
        "Runner Label": levels.runner_label,
        "Near High %": round(levels.near_high_pct, 2),
        "Volume Acceleration": round(levels.volume_acceleration, 2),
        "Compression %": round(levels.compression_pct, 2),
        "Volume Spike": levels.volume_spike,
        "Pre-Breakout Setup": levels.pre_breakout_setup,
        "Halt Candidate": levels.halt_candidate,
        "Confirmation": confirmation,
        "Invalidation": invalidation,
        "Why This Works": why,
        "Avoid Trade": avoid,
        "Last Updated": datetime.now().isoformat(timespec="seconds"),
    }


def analyze_candidates() -> pd.DataFrame:
    if not CANDIDATES_FILE.exists():
        return pd.DataFrame()

    candidates = pd.read_excel(CANDIDATES_FILE, engine="openpyxl")
    if candidates.empty or "Ticker" not in candidates.columns:
        return pd.DataFrame()

    plans: list[dict[str, object]] = []
    for _, row in candidates.drop_duplicates(subset=["Ticker"]).iterrows():
        ticker = str(row["Ticker"]).upper()
        intraday = safe_download(ticker)
        levels = calculate_levels(row, intraday)
        if levels is None:
            continue
        plans.append(generate_trade_plan(levels))

    if not plans:
        return pd.DataFrame()
    df = pd.DataFrame(plans)
    df["_runner_rank"] = df["Runner Label"].astype(str).eq("HIGH POTENTIAL RUNNERS").map({True: 0, False: 1})
    status_rank = {"VALID TRADE": 0, "WAIT": 1, "WAIT FOR PULLBACK": 2, "NO TRADE": 3}
    df["_status_rank"] = df["Status"].map(status_rank).fillna(9)
    df = df.sort_values(["_runner_rank", "_status_rank", "Momentum Score"], ascending=[True, True, False]).drop(columns=["_runner_rank", "_status_rank"])
    return df


def save_analysis(df: pd.DataFrame) -> None:
    columns = [
        "Ticker",
        "Status",
        "Setup Type",
        "Current Price",
        "Entry",
        "Stop",
        "Target 1",
        "Target 2",
        "Risk/Reward",
        "Probability",
        "Momentum Score",
        "Runner Label",
        "Near High %",
        "Volume Acceleration",
        "Compression %",
        "Volume Spike",
        "Pre-Breakout Setup",
        "Halt Candidate",
        "Confirmation",
        "Invalidation",
        "Why This Works",
        "Avoid Trade",
        "Last Updated",
    ]
    if df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_excel(ANALYZED_FILE, index=False, engine="openpyxl")


def run_once() -> None:
    df = analyze_candidates()
    save_analysis(df)
    print(f"Analyzed {len(df)} plans into {ANALYZED_FILE}.")


def watch(interval_seconds: int = 2) -> None:
    print(f"Watching {CANDIDATES_FILE}; writing {ANALYZED_FILE}. Press Ctrl+C to stop.")
    while True:
        try:
            run_once()
        except Exception as exc:
            print(f"Analyzer skipped one cycle: {exc}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    watch(2)
