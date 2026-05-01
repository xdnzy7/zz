from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


APP_TITLE = "Professional Momentum Scanner"
APP_VERSION = "bilingual-playbook-scanner-2026-05-01"

YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

CLOUD_MAX_TICKERS = 150
CLOUD_WORKERS = 8
CLOUD_SCAN_TIMEOUT = 20
CLOUD_SCAN_INTERVAL = 60

FALLBACK_RUNNERS = [
    "SOUN", "BBAI", "KULR", "OPEN", "PLUG", "MARA", "RIOT", "HOLO",
    "GNS", "FFIE", "LUCY", "WISA", "TIVC", "ATER", "GFAI", "CXAI",
    "WULF", "BITF", "QBTS", "IONQ", "RGTI", "SERV", "LUNR", "ACHR",
    "AISP", "RILY", "LCID", "RIVN", "DNA", "JOBY", "RKLB", "ASTS",
]

SETUP_AR = {
    "BREAKOUT WATCH": "مراقبة اختراق",
    "DIP BUY ZONE": "منطقة شراء على النزول",
    "MOMENTUM TRIGGER": "تفعيل الزخم",
    "NHOD MOMENTUM": "أعلى سعر جديد اليوم",
    "SCALP ONLY": "مضاربة سريعة فقط",
    "WAIT FOR PULLBACK": "انتظر رجوع السعر",
    "WAIT FOR REVERSAL": "انتظر انعكاس",
    "TOO LATE / AVOID": "متأخر / تجنب",
    "INVALID DATA": "بيانات غير صالحة",
}

TEXT = {
    "English": {
        "app_title": "Professional Small-Cap Momentum Scanner",
        "subtitle": "Automatic discovery, key levels, setup classification, and disciplined risk checks. No fake prediction.",
        "language": "Language",
        "scan_now": "Scan now",
        "cloud_fast": "Cloud fast mode",
        "advanced_watchlist": "Optional advanced watchlist",
        "advanced_help": "Optional only. The scanner does not require manual ticker input.",
        "last_scan": "Last scan",
        "scanned": "Scanned",
        "valid": "Valid plans",
        "hot": "Hot movers",
        "sources": "Discovery sources",
        "compact": "Compact table",
        "details": "Extra details",
        "no_results": "No clean setups in this section. The scanner is intentionally selective.",
        "explosive": "Explosive Runners Now",
        "breakout": "Breakout Watch",
        "dip": "Dip Buy Zones",
        "watched": "Watched Movers",
        "ticker": "Ticker",
        "setup": "Setup",
        "current": "Current price",
        "break": "Break level",
        "dip_zone": "Dip zone",
        "stop": "Stop",
        "target1": "Target 1",
        "target2": "Target 2",
        "rr": "Risk/reward",
        "confirmation": "Confirmation",
        "invalidation": "Invalidation",
        "why": "Why this works",
        "avoid": "Avoid if",
        "price": "Price",
        "gain": "Gain",
        "rvol": "RVOL",
        "vol_acc": "Volume acceleration",
        "near_high": "Near high",
        "gap": "Gap",
        "status": "Status",
        "source": "Source",
        "mode": "Scanner mode",
        "footer": "Not financial advice. This app filters live conditions and builds risk-defined plans; it does not predict outcomes.",
        "valid_trade": "VALID TRADE",
        "watch": "WATCH",
        "wait": "WAIT",
        "caution": "CAUTION",
        "invalid": "INVALID",
        "scalp": "SCALP",
        "flat": "No change",
    },
    "Arabic": {
        "app_title": "ماسح احترافي لأسهم الزخم الصغيرة",
        "subtitle": "اكتشاف تلقائي، مستويات رئيسية، تصنيف فرص، وفحص مخاطرة منضبط. بدون تنبؤ وهمي.",
        "language": "اللغة",
        "scan_now": "افحص الآن",
        "cloud_fast": "وضع السحابة السريع",
        "advanced_watchlist": "قائمة مراقبة متقدمة اختيارية",
        "advanced_help": "اختياري فقط. الماسح لا يحتاج إلى إدخال يدوي للرموز.",
        "last_scan": "آخر فحص",
        "scanned": "المفحوصة",
        "valid": "خطط صالحة",
        "hot": "أسهم نشطة",
        "sources": "مصادر الاكتشاف",
        "compact": "جدول مختصر",
        "details": "تفاصيل إضافية",
        "no_results": "لا توجد فرص نظيفة في هذا القسم. الماسح انتقائي عمدا.",
        "explosive": "الأسهم المنفجرة الآن",
        "breakout": "مراقبة الاختراق",
        "dip": "مناطق الشراء على النزول",
        "watched": "أسهم تحت المراقبة",
        "ticker": "الرمز",
        "setup": "نوع الفرصة",
        "current": "السعر الحالي",
        "break": "مستوى الاختراق",
        "dip_zone": "منطقة الشراء على النزول",
        "stop": "وقف الخسارة",
        "target1": "الهدف الأول",
        "target2": "الهدف الثاني",
        "rr": "العائد مقابل المخاطرة",
        "confirmation": "التأكيد",
        "invalidation": "الإلغاء",
        "why": "لماذا قد تعمل الفرصة",
        "avoid": "تجنب إذا",
        "price": "السعر",
        "gain": "الصعود",
        "rvol": "الحجم النسبي",
        "vol_acc": "تسارع الحجم",
        "near_high": "القرب من القمة",
        "gap": "الفجوة",
        "status": "الحالة",
        "source": "المصدر",
        "mode": "وضع الماسح",
        "footer": "ليست نصيحة مالية. التطبيق يرشح الشروط الحية ويبني خطة مخاطرة؛ ولا يتنبأ بالنتائج.",
        "valid_trade": "صفقة صالحة",
        "watch": "مراقبة",
        "wait": "انتظار",
        "caution": "حذر",
        "invalid": "غير صالح",
        "scalp": "مضاربة",
        "flat": "بدون تغيير",
    },
}


@dataclass(frozen=True)
class ScanConfig:
    max_tickers: int = CLOUD_MAX_TICKERS
    workers: int = CLOUD_WORKERS
    scan_timeout: int = CLOUD_SCAN_TIMEOUT
    scan_interval: int = CLOUD_SCAN_INTERVAL
    min_price: float = 0.2
    max_price: float = 50.0
    min_volume: int = 100_000
    request_timeout: int = 5
    max_cards: int = 5


@dataclass(frozen=True)
class DiscoveryResult:
    tickers: list[str]
    sources: dict[str, int]
    message: str


def is_arabic(lang: str) -> bool:
    return lang == "Arabic"


def tr(key: str, lang: str) -> str:
    return TEXT[lang][key]


def setup_text(label: str, lang: str) -> str:
    return SETUP_AR.get(label, label) if is_arabic(lang) else label


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(value, np.nan)
    return default if np.isnan(number) else int(number)


def clean_symbol(value: Any) -> str | None:
    symbol = str(value or "").upper().strip().replace(".", "-")
    if VALID_TICKER_RE.fullmatch(symbol):
        return symbol
    return None


def dedupe(symbols: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in symbols:
        symbol = clean_symbol(item)
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def fmt_money(value: Any) -> str:
    number = safe_float(value, np.nan)
    if np.isnan(number):
        return "N/A"
    places = 4 if abs(number) < 1 else 2
    return f"${number:,.{places}f}"


def fmt_pct(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.1f}%"


def fmt_x(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.2f}x"


def fmt_rr(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"1:{number:.2f}"


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def utc_clock(ts: float | None) -> str:
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%H:%M:%S UTC")


def request_json(url: str, params: dict[str, Any] | None = None, timeout: int = 5) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params or {},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def yahoo_screener(screen_id: str, count: int, timeout: int) -> list[str]:
    payload = request_json(
        YAHOO_SCREENER_URL,
        {"scrIds": screen_id, "count": count},
        timeout,
    )
    quotes = payload.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    return dedupe([quote.get("symbol") for quote in quotes if quote.get("quoteType", "EQUITY") in {"EQUITY", ""}])


def yahoo_trending(timeout: int) -> list[str]:
    payload = request_json(YAHOO_TRENDING_URL, timeout=timeout)
    quotes = payload.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    return dedupe([quote.get("symbol") for quote in quotes])


def yahoo_premarket(timeout: int) -> list[str]:
    found: list[str] = []
    for screen_id in ("premarket_gainers", "premarket_movers"):
        try:
            found.extend(yahoo_screener(screen_id, 40, timeout))
        except Exception:
            continue
    return dedupe(found)


def parse_watchlist(value: str) -> list[str]:
    return dedupe(re.split(r"[\s,;]+", value or ""))


@st.cache_data(ttl=60, show_spinner=False)
def discover_tickers(optional_watchlist: str, max_tickers: int, request_timeout: int) -> DiscoveryResult:
    sources: dict[str, list[str]] = {}
    loaders = {
        "Yahoo day_gainers": lambda: yahoo_screener("day_gainers", 80, request_timeout),
        "Yahoo most_actives": lambda: yahoo_screener("most_actives", 80, request_timeout),
        "Yahoo trending": lambda: yahoo_trending(request_timeout),
        "Yahoo premarket": lambda: yahoo_premarket(request_timeout),
    }
    for name, loader in loaders.items():
        try:
            sources[name] = loader()
        except Exception:
            sources[name] = []

    optional = parse_watchlist(optional_watchlist)
    if optional:
        sources["Optional watchlist"] = optional

    merged: list[str] = []
    for symbols in sources.values():
        merged.extend(symbols)

    if not merged:
        sources["Fallback runners"] = FALLBACK_RUNNERS.copy()
        merged = FALLBACK_RUNNERS.copy()
        message = "Fallback runner list active"
    else:
        fallback_fill = [ticker for ticker in FALLBACK_RUNNERS if ticker not in merged]
        sources["Fallback runners"] = fallback_fill[:30]
        merged.extend(fallback_fill[:30])
        message = "Live discovery active"

    tickers = dedupe(merged)[:max_tickers]
    return DiscoveryResult(
        tickers=tickers,
        sources={name: len(dedupe(symbols)) for name, symbols in sources.items()},
        message=message,
    )


@st.cache_data(ttl=20, show_spinner=False)
def quote_batch(symbols: tuple[str, ...], request_timeout: int) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    payload = request_json(
        YAHOO_QUOTE_URL,
        {"symbols": ",".join(symbols)},
        request_timeout,
    )
    quotes = payload.get("quoteResponse", {}).get("result", [])
    return {quote["symbol"]: quote for quote in quotes if quote.get("symbol")}


@st.cache_data(ttl=20, show_spinner=False)
def history_intraday(symbol: str) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        prepost=True,
        threads=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]
    return frame.dropna(how="all")


@st.cache_data(ttl=1800, show_spinner=False)
def history_daily(symbol: str) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        period="30d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]
    return frame.dropna(how="all")


def latest_bar_age_seconds(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("inf")
    index = frame.index[-1]
    if getattr(index, "tzinfo", None) is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return max(0.0, now_ts() - index.timestamp())


def calculate_vwap(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    volume = frame["Volume"].replace(0, np.nan)
    total_volume = safe_float(volume.sum(), 0)
    if total_volume <= 0:
        return np.nan
    return safe_float((typical * volume).sum() / total_volume, np.nan)


def volume_acceleration(frame: pd.DataFrame) -> float:
    if len(frame) < 8:
        return np.nan
    recent = safe_float(frame["Volume"].tail(5).mean(), np.nan)
    prior = safe_float(frame["Volume"].iloc[:-5].tail(20).mean(), np.nan)
    if np.isnan(recent) or np.isnan(prior) or prior <= 0:
        return np.nan
    return recent / prior


def relative_volume(frame: pd.DataFrame, daily: pd.DataFrame) -> float:
    if frame.empty or daily.empty:
        return np.nan
    current_volume = safe_float(frame["Volume"].sum(), np.nan)
    avg_daily_volume = safe_float(daily["Volume"].tail(20).mean(), np.nan)
    if np.isnan(current_volume) or np.isnan(avg_daily_volume) or avg_daily_volume <= 0:
        return np.nan
    elapsed_minutes = min(max(len(frame), 1), 390)
    expected_volume = avg_daily_volume * (elapsed_minutes / 390)
    return current_volume / expected_volume if expected_volume > 0 else np.nan


def invalid_row(symbol: str, reason: str = "Invalid data") -> dict[str, Any]:
    return {
        "ticker": symbol,
        "setup": "INVALID DATA",
        "mode": "Watched Movers",
        "trade_status": "INVALID",
        "score": 0.0,
        "current": np.nan,
        "previous_close": np.nan,
        "open": np.nan,
        "day_high": np.nan,
        "day_low": np.nan,
        "nhod_level": np.nan,
        "breakout_level": np.nan,
        "dip_low": np.nan,
        "dip_high": np.nan,
        "vwap": np.nan,
        "support": np.nan,
        "stop": np.nan,
        "target_1": np.nan,
        "target_2": np.nan,
        "rr": np.nan,
        "extension_from_base": np.nan,
        "vol_accel": np.nan,
        "rvol": np.nan,
        "near_high_pct": np.nan,
        "gap_pct": np.nan,
        "gain_pct": np.nan,
        "volume": 0,
        "stale_seconds": float("inf"),
        "mismatch_pct": np.nan,
        "valid_trade": False,
        "reason": reason,
        "last_updated": utc_clock(now_ts()),
    }


def normalize_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    defaults = invalid_row("")
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    frame["valid_trade"] = frame["valid_trade"].fillna(False).astype(bool)
    return frame


def compute_levels(symbol: str, quote: dict[str, Any], config: ScanConfig) -> dict[str, Any]:
    intraday = history_intraday(symbol)
    daily = history_daily(symbol)
    if intraday.empty:
        return invalid_row(symbol, "No intraday data")

    current = safe_float(intraday["Close"].iloc[-1], np.nan)
    quote_price = safe_float(quote.get("regularMarketPrice") or quote.get("preMarketPrice"), np.nan)
    mismatch_pct = 0.0
    if not np.isnan(quote_price) and quote_price > 0 and not np.isnan(current):
        mismatch_pct = abs(current - quote_price) / quote_price * 100

    previous_close = safe_float(
        quote.get("regularMarketPreviousClose")
        or (daily["Close"].iloc[-2] if len(daily) > 1 else np.nan),
        np.nan,
    )
    open_price = safe_float(intraday["Open"].iloc[0], np.nan)
    day_high = safe_float(intraday["High"].max(), np.nan)
    day_low = safe_float(intraday["Low"].min(), np.nan)
    day_volume = safe_int(intraday["Volume"].sum(), 0)
    vwap = calculate_vwap(intraday)

    recent = intraday.tail(min(45, len(intraday)))
    recent_high = safe_float(recent["High"].max(), day_high)
    recent_low = safe_float(recent["Low"].min(), day_low)
    base_low = safe_float(recent["Low"].min(), day_low)

    gain_pct = ((current - previous_close) / previous_close * 100) if previous_close > 0 else np.nan
    gap_pct = ((open_price - previous_close) / previous_close * 100) if previous_close > 0 else np.nan
    near_high_pct = ((day_high - current) / day_high * 100) if day_high > 0 else np.nan
    extension_from_base = ((current - base_low) / base_low * 100) if base_low > 0 else np.nan
    vol_accel = volume_acceleration(intraday)
    rvol = relative_volume(intraday, daily)
    stale_seconds = latest_bar_age_seconds(intraday)

    breakout_level = round(max(day_high, recent_high) * 1.002, 4)
    nhod_level = round(day_high * 1.001, 4)
    support_candidates = [x for x in [recent_low, vwap, day_low] if not np.isnan(safe_float(x, np.nan))]
    support = round(max(day_low, min(support_candidates)) if support_candidates else day_low, 4)
    dip_low = round(max(day_low, (vwap * 0.985 if not np.isnan(vwap) else recent_low), recent_low * 0.995), 4)
    dip_high = round(max(dip_low, (vwap * 1.01 if not np.isnan(vwap) else recent_low * 1.01)), 4)

    stop = round(min(dip_low, support) * 0.985, 4)
    if np.isnan(current) or stop >= current:
        stop = round(current * 0.97, 4) if not np.isnan(current) else np.nan
    risk = current - stop if not np.isnan(current) and not np.isnan(stop) else np.nan
    target_1 = round(current + risk * 2, 4) if risk > 0 else np.nan
    target_2 = round(current + risk * 3, 4) if risk > 0 else np.nan
    rr = ((target_1 - current) / risk) if risk > 0 else np.nan

    above_vwap = not np.isnan(vwap) and current >= vwap
    near_trigger = not np.isnan(near_high_pct) and near_high_pct <= 3.5
    inside_dip_zone = not np.isnan(dip_low) and not np.isnan(dip_high) and dip_low <= current <= dip_high
    active_volume = day_volume >= config.min_volume and (safe_float(rvol, 0) >= 1.2 or safe_float(vol_accel, 0) >= 1.3)
    reclaiming_support = current >= support and current >= open_price * 0.995
    latest_not_stale = stale_seconds <= 20 * 60
    risk_ok = safe_float(rr, 0) >= 2.0 and risk > 0
    valid_trade = (near_trigger or inside_dip_zone) and risk_ok and latest_not_stale and (above_vwap or reclaiming_support) and active_volume

    day_range_pct = ((day_high - day_low) / current * 100) if current > 0 else np.nan
    higher_lows = len(intraday) >= 12 and intraday["Low"].tail(12).iloc[-1] > intraday["Low"].tail(12).min()
    consolidation = safe_float(day_range_pct, 99) <= 9 and safe_float(near_high_pct, 99) <= 6
    volume_ignition = safe_float(vol_accel, 0) >= 1.5 or safe_float(rvol, 0) >= 1.5
    last_bar = intraday.iloc[-1]
    lower_wick_bounce = safe_float(last_bar["Close"], 0) > safe_float(last_bar["Open"], 0) and safe_float(last_bar["Low"], 0) <= dip_high
    resistance_nearby = day_high > 0 and current >= day_high * 0.985

    if (
        np.isnan(current)
        or current <= 0
        or current < config.min_price
        or current > config.max_price
        or np.isnan(previous_close)
        or mismatch_pct > 10
    ):
        setup = "INVALID DATA"
        mode = "Watched Movers"
        trade_status = "INVALID"
        valid_trade = False
    elif safe_float(gain_pct, 0) > 30 and safe_float(extension_from_base, 0) > 18 and current > dip_high * 1.05:
        setup = "WAIT FOR PULLBACK"
        mode = "Watched Movers"
        trade_status = "WAIT"
        valid_trade = False
    elif current < vwap and current < open_price:
        setup = "WAIT FOR REVERSAL"
        mode = "Watched Movers"
        trade_status = "WAIT"
        valid_trade = False
    elif valid_trade and safe_float(gain_pct, 0) >= 10 and near_trigger and safe_float(vol_accel, 0) >= 1.4:
        setup = "NHOD MOMENTUM" if current >= day_high * 0.995 else "MOMENTUM TRIGGER"
        mode = "Explosive Runners Now"
        trade_status = "VALID TRADE"
    elif valid_trade and inside_dip_zone and lower_wick_bounce:
        setup = "DIP BUY ZONE"
        mode = "Dip Buy Zones"
        trade_status = "VALID TRADE"
    elif valid_trade and consolidation and higher_lows and volume_ignition:
        setup = "BREAKOUT WATCH"
        mode = "Breakout Watch"
        trade_status = "VALID TRADE"
    elif safe_float(gain_pct, 0) >= 10 and near_trigger and active_volume:
        setup = "BREAKOUT WATCH"
        mode = "Breakout Watch"
        trade_status = "WATCH"
    elif inside_dip_zone and current >= support:
        setup = "DIP BUY ZONE"
        mode = "Dip Buy Zones"
        trade_status = "WATCH"
    elif resistance_nearby and (safe_float(rvol, 0) < 1.2 or safe_float(day_range_pct, 99) < 5):
        setup = "SCALP ONLY"
        mode = "Watched Movers"
        trade_status = "SCALP"
    elif safe_float(gain_pct, 0) >= 15:
        setup = "WAIT FOR PULLBACK"
        mode = "Watched Movers"
        trade_status = "WAIT"
    else:
        setup = "SCALP ONLY" if active_volume else "TOO LATE / AVOID"
        mode = "Watched Movers"
        trade_status = "CAUTION"

    score = 0.0
    score += min(max(safe_float(gain_pct, 0), 0), 30)
    score += min(max(safe_float(rvol, 0) * 8, 0), 25)
    score += min(max(safe_float(vol_accel, 0) * 8, 0), 20)
    score += max(0, 15 - min(max(safe_float(near_high_pct, 15), 0), 15))
    score += 10 if valid_trade else 0

    return {
        "ticker": symbol,
        "setup": setup,
        "mode": mode,
        "trade_status": trade_status,
        "score": round(score, 1),
        "current": current,
        "previous_close": previous_close,
        "open": open_price,
        "day_high": day_high,
        "day_low": day_low,
        "nhod_level": nhod_level,
        "breakout_level": breakout_level,
        "dip_low": dip_low,
        "dip_high": dip_high,
        "vwap": vwap,
        "support": support,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "rr": rr,
        "extension_from_base": extension_from_base,
        "vol_accel": vol_accel,
        "rvol": rvol,
        "near_high_pct": near_high_pct,
        "gap_pct": gap_pct,
        "gain_pct": gain_pct,
        "volume": day_volume,
        "stale_seconds": stale_seconds,
        "mismatch_pct": mismatch_pct,
        "valid_trade": valid_trade,
        "reason": "",
        "last_updated": utc_clock(now_ts()),
    }


def scan_universe(tickers: list[str], config: ScanConfig, progress=None) -> pd.DataFrame:
    start = time.monotonic()
    deadline = start + config.scan_timeout
    rows: list[dict[str, Any]] = []

    quotes: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(tickers), 50):
        try:
            quotes.update(quote_batch(tuple(tickers[offset : offset + 50]), config.request_timeout))
        except Exception:
            continue

    executor = ThreadPoolExecutor(max_workers=config.workers)
    futures = {
        executor.submit(compute_levels, ticker, quotes.get(ticker, {}), config): ticker
        for ticker in tickers
    }
    pending = set(futures)
    completed = 0
    try:
        while pending and time.monotonic() < deadline:
            done, pending = wait(pending, timeout=0.4, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                try:
                    rows.append(future.result(timeout=0))
                except Exception as exc:
                    rows.append(invalid_row(futures[future], str(exc)[:120]))
                if progress is not None:
                    progress.progress(min(completed / max(len(tickers), 1), 1.0))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    if not rows:
        return pd.DataFrame()

    frame = normalize_scan_frame(pd.DataFrame(rows))
    return frame.sort_values(
        ["valid_trade", "score", "gain_pct", "rvol"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def apply_theme(lang: str) -> None:
    direction = "rtl" if is_arabic(lang) else "ltr"
    align = "right" if is_arabic(lang) else "left"
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: dark; }}
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(22, 163, 127, .22), transparent 30rem),
                linear-gradient(135deg, #07100f 0%, #0b1118 55%, #10140f 100%);
            color: #f4fbf7;
            direction: {direction};
        }}
        .block-container {{
            max-width: 1280px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, p, label, div[data-testid="stMarkdownContainer"] {{ text-align: {align}; }}
        section[data-testid="stSidebar"] {{ background: #08100f; }}
        .stButton button {{
            min-height: 2.5rem;
            border-radius: 8px;
            border: 1px solid rgba(74, 222, 128, .35);
            background: #123c34;
            color: #effff8;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, .055);
            border: 1px solid rgba(255, 255, 255, .10);
            border-radius: 8px;
            padding: .85rem 1rem;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255, 255, 255, .08);
            border-radius: 8px;
            overflow: hidden;
        }}
        .trade-card {{
            border: 1px solid rgba(255, 255, 255, .10);
            background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
            border-radius: 16px;
            padding: 1rem;
            margin: .9rem 0;
            box-shadow: 0 18px 42px rgba(0, 0, 0, .22);
        }}
        .trade-head {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            direction: {direction};
        }}
        .trade-title {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #f6fff9;
            text-align: {align};
        }}
        .trade-sub {{
            margin-top: .2rem;
            color: #aebbb5;
            font-size: .86rem;
            text-align: {align};
        }}
        .status-pill {{
            border-radius: 999px;
            padding: .28rem .65rem;
            font-size: .76rem;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,.13);
            white-space: nowrap;
        }}
        .status-valid {{ color: #86efac; background: rgba(34,197,94,.13); border-color: rgba(34,197,94,.28); }}
        .status-wait {{ color: #fde68a; background: rgba(234,179,8,.12); border-color: rgba(234,179,8,.28); }}
        .status-caution {{ color: #fdba74; background: rgba(249,115,22,.12); border-color: rgba(249,115,22,.28); }}
        .status-invalid {{ color: #fca5a5; background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.28); }}
        .metric-card {{
            min-height: 96px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, .10);
            background: rgba(7, 15, 17, .92);
            padding: .82rem .9rem;
            text-align: {align};
            box-shadow: inset 0 1px 0 rgba(255,255,255,.055), 0 14px 28px rgba(0,0,0,.18);
        }}
        .metric-label {{
            color: #aebbb5;
            font-size: .77rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .metric-value {{
            color: #f4fff9;
            margin-top: .45rem;
            font-size: clamp(1.2rem, 2.1vw, 1.8rem);
            font-weight: 800;
            line-height: 1.05;
            font-variant-numeric: tabular-nums;
        }}
        .note-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .8rem;
            margin-top: .8rem;
        }}
        .note {{
            border-radius: 12px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
            padding: .75rem;
            text-align: {align};
        }}
        .note-label {{
            color: #aebbb5;
            font-size: .78rem;
            margin-bottom: .25rem;
            font-weight: 700;
        }}
        .note-text {{ color: #edf8f1; font-size: .9rem; line-height: 1.45; }}
        @media (max-width: 760px) {{
            .trade-head {{ display: block; }}
            .status-pill {{ display: inline-block; margin-top: .6rem; }}
            .note-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_class(status: str) -> str:
    if status == "VALID TRADE":
        return "status-valid"
    if status in {"WAIT", "WATCH", "SCALP"}:
        return "status-wait"
    if status == "INVALID":
        return "status-invalid"
    return "status-caution"


def status_text(status: str, lang: str) -> str:
    mapping = {
        "VALID TRADE": tr("valid_trade", lang),
        "WATCH": tr("watch", lang),
        "WAIT": tr("wait", lang),
        "CAUTION": tr("caution", lang),
        "INVALID": tr("invalid", lang),
        "SCALP": tr("scalp", lang),
    }
    return mapping.get(status, status)


def plan_comments(row: pd.Series, lang: str) -> dict[str, str]:
    break_level = fmt_money(row["breakout_level"])
    dip_low = fmt_money(row["dip_low"])
    dip_high = fmt_money(row["dip_high"])
    stop = fmt_money(row["stop"])
    vwap = fmt_money(row["vwap"])
    rvol = fmt_x(row["rvol"])
    vol = fmt_x(row["vol_accel"])

    if is_arabic(lang):
        return {
            "confirmation": f"اختراق مستوى {break_level} مع حجم تداول قوي قد يفعّل استمرار الحركة.",
            "invalidation": f"الإلغاء إذا كسر السعر {stop} أو فقد VWAP عند {vwap}.",
            "why": f"السهم قريب من مستوى مهم مع حجم نسبي {rvol} وتسارع حجم {vol}، والخطة محددة المخاطرة.",
            "avoid": f"إذا رفض السعر مستوى {break_level}، تجنب المطاردة. السهم ممتد؛ انتظر رجوع نظيف.",
            "dip": f"منطقة الشراء على النزول بين {dip_low} و {dip_high}، ووقف الخسارة تحت {stop}.",
        }
    return {
        "confirmation": f"Break above {break_level} with volume can trigger continuation.",
        "invalidation": f"Invalid if price loses {stop} or fails VWAP near {vwap}.",
        "why": f"Price is near a key level with {rvol} RVOL and {vol} volume acceleration, while risk is defined.",
        "avoid": f"If it rejects {break_level}, avoid chasing. Already extended; wait for clean pullback.",
        "dip": f"Dip zone is {dip_low}-{dip_high}; stop under {stop}.",
    }


def render_note(label: str, text: str) -> str:
    return f"""
        <div class="note">
            <div class="note-label">{html.escape(label)}</div>
            <div class="note-text">{html.escape(text)}</div>
        </div>
    """


def render_trade_card(row: pd.Series, lang: str) -> None:
    setup = setup_text(str(row["setup"]), lang)
    status = str(row["trade_status"])
    comments = plan_comments(row, lang)

    st.markdown(
        f"""
        <div class="trade-card">
            <div class="trade-head">
                <div>
                    <div class="trade-title">{html.escape(str(row["ticker"]))} · {html.escape(setup)}</div>
                    <div class="trade-sub">{html.escape(tr("mode", lang))}: {html.escape(section_title(str(row["mode"]), lang))}</div>
                </div>
                <div class="status-pill {status_class(status)}">{html.escape(status_text(status, lang))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = [
        (tr("current", lang), fmt_money(row["current"])),
        (tr("break", lang), fmt_money(row["breakout_level"])),
        (tr("dip_zone", lang), f"{fmt_money(row['dip_low'])} - {fmt_money(row['dip_high'])}"),
        (tr("stop", lang), fmt_money(row["stop"])),
        (tr("target1", lang), fmt_money(row["target_1"])),
        (tr("target2", lang), fmt_money(row["target_2"])),
        (tr("rr", lang), fmt_rr(row["rr"])),
        (tr("rvol", lang), fmt_x(row["rvol"])),
    ]
    for start in range(0, len(metrics), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, metrics[start : start + 4]):
            with col:
                render_metric_card(label, value)

    notes_html = "".join(
        [
            render_note(tr("confirmation", lang), comments["confirmation"]),
            render_note(tr("invalidation", lang), comments["invalidation"]),
            render_note(tr("why", lang), comments["why"]),
            render_note(tr("avoid", lang), comments["avoid"]),
        ]
    )
    st.markdown(f'<div class="note-grid">{notes_html}</div>', unsafe_allow_html=True)

    with st.expander(tr("details", lang)):
        details = pd.DataFrame(
            [
                {
                    "Previous close": fmt_money(row["previous_close"]),
                    "Open": fmt_money(row["open"]),
                    "Day high": fmt_money(row["day_high"]),
                    "Day low": fmt_money(row["day_low"]),
                    "NHOD": fmt_money(row["nhod_level"]),
                    "VWAP": fmt_money(row["vwap"]),
                    "Support": fmt_money(row["support"]),
                    "Extension": fmt_pct(row["extension_from_base"]),
                    "Volume acceleration": fmt_x(row["vol_accel"]),
                    "Near high": fmt_pct(row["near_high_pct"]),
                    "Gap": fmt_pct(row["gap_pct"]),
                    "Volume": f"{safe_int(row['volume']):,}",
                    "Data age": f"{safe_float(row['stale_seconds'], 0) / 60:.1f} min",
                }
            ]
        )
        st.dataframe(details, use_container_width=True, hide_index=True)


def section_title(mode: str, lang: str) -> str:
    lookup = {
        "Explosive Runners Now": tr("explosive", lang),
        "Breakout Watch": tr("breakout", lang),
        "Dip Buy Zones": tr("dip", lang),
        "Watched Movers": tr("watched", lang),
    }
    return lookup.get(mode, mode)


def section_rows(frame: pd.DataFrame, section: str, max_cards: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    if section == "Explosive Runners Now":
        return frame[frame["mode"].eq(section)].head(max_cards)
    if section == "Breakout Watch":
        return frame[frame["setup"].isin(["BREAKOUT WATCH", "MOMENTUM TRIGGER", "NHOD MOMENTUM"])].head(max_cards)
    if section == "Dip Buy Zones":
        return frame[frame["setup"].eq("DIP BUY ZONE")].head(max_cards)
    return frame[~frame["mode"].isin(["Explosive Runners Now", "Dip Buy Zones"])].head(max_cards)


def compact_table(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                tr("ticker", lang): row["ticker"],
                tr("setup", lang): setup_text(str(row["setup"]), lang),
                tr("price", lang): fmt_money(row["current"]),
                tr("gain", lang): fmt_pct(row["gain_pct"]),
                tr("rvol", lang): fmt_x(row["rvol"]),
                tr("vol_acc", lang): fmt_x(row["vol_accel"]),
                tr("near_high", lang): fmt_pct(row["near_high_pct"]),
                tr("rr", lang): fmt_rr(row["rr"]),
                tr("status", lang): status_text(str(row["trade_status"]), lang),
            }
            for _, row in frame.iterrows()
        ]
    )


def render_section(frame: pd.DataFrame, section: str, lang: str, max_cards: int) -> None:
    st.header(section_title(section, lang))
    subset = section_rows(frame, section, max_cards)
    if subset.empty:
        st.info(tr("no_results", lang))
        return
    st.dataframe(compact_table(subset, lang), use_container_width=True, hide_index=True)
    for _, row in subset.iterrows():
        render_trade_card(row, lang)


def initialize_state() -> None:
    defaults = {
        "scan_df": pd.DataFrame(),
        "last_scan_ts": None,
        "source_counts": {},
        "discovery_message": "",
        "optional_watchlist": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def should_scan(interval: int) -> bool:
    last_scan = st.session_state.get("last_scan_ts")
    if not last_scan:
        return True
    return now_ts() - float(last_scan) >= interval


def run_scan(config: ScanConfig, optional_watchlist: str, lang: str) -> None:
    discovery = discover_tickers(optional_watchlist, config.max_tickers, config.request_timeout)
    progress = st.progress(0, text="Scanning..." if not is_arabic(lang) else "جار الفحص...")
    frame = scan_universe(discovery.tickers, config, progress)
    progress.empty()
    st.session_state.scan_df = frame
    st.session_state.last_scan_ts = now_ts()
    st.session_state.source_counts = discovery.sources
    st.session_state.discovery_message = discovery.message


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    initialize_state()

    with st.sidebar:
        lang = st.radio("Language / اللغة", ["English", "Arabic"], horizontal=True, key="language")
        apply_theme(lang)
        st.caption(APP_VERSION)
        st.divider()
        st.subheader("Scanner" if not is_arabic(lang) else "الماسح")
        cloud_fast = st.toggle(tr("cloud_fast", lang), value=True)
        config = ScanConfig() if cloud_fast else ScanConfig(max_tickers=220, workers=10, scan_timeout=28)
        with st.expander(tr("advanced_watchlist", lang)):
            st.session_state.optional_watchlist = st.text_area(
                tr("advanced_watchlist", lang),
                value=st.session_state.optional_watchlist,
                help=tr("advanced_help", lang),
                height=90,
                label_visibility="collapsed",
            )
        scan_clicked = st.button(tr("scan_now", lang), use_container_width=True)
        st.caption(f"{tr('next_refresh', lang)}: {config.scan_interval}s")

    apply_theme(lang)
    st.title(tr("app_title", lang))
    st.caption(tr("subtitle", lang))

    if scan_clicked or should_scan(config.scan_interval):
        run_scan(config, st.session_state.optional_watchlist, lang)

    frame: pd.DataFrame = st.session_state.scan_df
    valid_count = int(frame["valid_trade"].sum()) if not frame.empty and "valid_trade" in frame else 0
    hot_count = int((frame["gain_pct"].fillna(0) >= 10).sum()) if not frame.empty and "gain_pct" in frame else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("last_scan", lang), utc_clock(st.session_state.last_scan_ts))
    k2.metric(tr("scanned", lang), f"{len(frame):,}")
    k3.metric(tr("valid", lang), f"{valid_count:,}")
    k4.metric(tr("hot", lang), f"{hot_count:,}")

    with st.expander(tr("sources", lang)):
        st.write(st.session_state.discovery_message)
        st.dataframe(
            pd.DataFrame([{"Source": name, "Tickers": count} for name, count in st.session_state.source_counts.items()]),
            use_container_width=True,
            hide_index=True,
        )

    if frame.empty:
        st.warning(tr("no_results", lang))
    else:
        render_section(frame, "Explosive Runners Now", lang, config.max_cards)
        render_section(frame, "Breakout Watch", lang, config.max_cards)
        render_section(frame, "Dip Buy Zones", lang, config.max_cards)
        render_section(frame, "Watched Movers", lang, config.max_cards)
        with st.expander(tr("compact", lang)):
            st.dataframe(compact_table(frame.head(60), lang), use_container_width=True, hide_index=True)

    st.caption(tr("footer", lang))

    elapsed = now_ts() - float(st.session_state.last_scan_ts or now_ts())
    if elapsed >= config.scan_interval:
        st.rerun()


if __name__ == "__main__":
    main()
