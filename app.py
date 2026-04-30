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
APP_VERSION = "bilingual-professional-momentum-scanner-2026-04-30"

YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

CLOUD_MAX_TICKERS = 150
CLOUD_WORKERS = 8
CLOUD_SCAN_TIMEOUT = 20
CLOUD_SCAN_INTERVAL = 60

FALLBACK_RUNNERS = [
    "SOUN",
    "BBAI",
    "KULR",
    "OPEN",
    "PLUG",
    "MARA",
    "RIOT",
    "HOLO",
    "GNS",
    "FFIE",
    "LUCY",
    "WISA",
    "TIVC",
    "ATER",
    "GFAI",
    "CXAI",
    "WULF",
    "BITF",
    "QBTS",
    "IONQ",
    "RGTI",
    "SERV",
    "LUNR",
    "ACHR",
]

SETUP_LABELS = {
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
        "page_title": "Professional Small-Cap Momentum Scanner",
        "subtitle": "Automatic hot ticker discovery, conservative setup classification, and trade plans built from live price, levels, volume, and risk.",
        "language": "Language",
        "refresh": "Scan now",
        "last_scan": "Last scan",
        "next_refresh": "Auto refresh window",
        "universe": "Tickers scanned",
        "valid": "Valid trade plans",
        "hot": "Hot movers",
        "stale": "stale",
        "scanner_controls": "Scanner controls",
        "cloud_fast": "Cloud fast mode",
        "advanced_watchlist": "Optional advanced watchlist",
        "advanced_help": "Optional only. The scanner already discovers tickers automatically.",
        "sources": "Discovery sources",
        "explosive": "Explosive Runners Now",
        "breakout": "Breakout Watch",
        "dip": "Dip Buy Zones",
        "watched": "Watched Movers",
        "table": "Compact table",
        "details": "Extra details",
        "no_results": "No clean setups in this section yet. The scanner is intentionally selective.",
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
        "vol_acc": "Vol accel",
        "near_high": "Near high",
        "gap": "Gap",
        "status": "Trade status",
        "mode": "Scanner mode",
        "source": "Source",
        "score": "Pre-Move Score",
        "footer": "Not financial advice. The app does not predict outcomes; it filters conditions and forces risk checks.",
    },
    "Arabic": {
        "page_title": "ماسح احترافي لأسهم الزخم الصغيرة",
        "subtitle": "اكتشاف تلقائي للأسهم النشطة، تصنيف محافظ للفرص، وخطة تداول مبنية على السعر والمستويات والحجم والمخاطرة.",
        "language": "اللغة",
        "refresh": "افحص الآن",
        "last_scan": "آخر فحص",
        "next_refresh": "نافذة التحديث التلقائي",
        "universe": "عدد الرموز المفحوصة",
        "valid": "خطط صالحة",
        "hot": "أسهم نشطة",
        "stale": "قديم",
        "scanner_controls": "إعدادات الماسح",
        "cloud_fast": "وضع السحابة السريع",
        "advanced_watchlist": "قائمة مراقبة متقدمة اختيارية",
        "advanced_help": "اختياري فقط. الماسح يكتشف الرموز تلقائيا.",
        "sources": "مصادر الاكتشاف",
        "explosive": "الأسهم المنفجرة الآن",
        "breakout": "مراقبة الاختراق",
        "dip": "مناطق الشراء على النزول",
        "watched": "أسهم تحت المراقبة",
        "table": "جدول مختصر",
        "details": "تفاصيل إضافية",
        "no_results": "لا توجد فرص نظيفة في هذا القسم حاليا. الماسح انتقائي عمدا.",
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
        "status": "حالة الخطة",
        "mode": "وضع الماسح",
        "source": "المصدر",
        "score": "درجة ما قبل الحركة",
        "footer": "ليست نصيحة مالية. التطبيق لا يتنبأ؛ بل يرشح الشروط ويفرض فحص المخاطرة.",
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
    max_cards_per_section: int = 5


@dataclass
class DiscoveryResult:
    tickers: list[str]
    sources: dict[str, int]
    message: str


def tr(key: str, lang: str) -> str:
    return TEXT[lang][key]


def ar_setup(label: str) -> str:
    return SETUP_LABELS.get(label, label)


def is_arabic(lang: str) -> bool:
    return lang == "Arabic"


def setup_text(label: str, lang: str) -> str:
    return ar_setup(label) if is_arabic(lang) else label


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(value, np.nan)
    return default if np.isnan(number) else int(number)


def clean_symbol(symbol: Any) -> str | None:
    value = str(symbol or "").upper().strip().replace(".", "-")
    if VALID_TICKER_RE.fullmatch(value):
        return value
    return None


def dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        symbol = clean_symbol(item)
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def fmt_money(value: Any) -> str:
    number = safe_float(value, np.nan)
    if np.isnan(number):
        return "N/A"
    digits = 4 if number < 1 else 2
    return f"${number:,.{digits}f}"


def fmt_pct(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.1f}%"


def fmt_x(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"{number:.2f}x"


def fmt_rr(value: Any) -> str:
    number = safe_float(value, np.nan)
    return "N/A" if np.isnan(number) else f"1:{number:.2f}"


def now_utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def utc_clock(ts: float | None) -> str:
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%H:%M:%S UTC")


def request_json(url: str, params: dict[str, Any] | None = None, timeout: int = 5) -> dict[str, Any]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def yahoo_screener(scr_id: str, count: int, timeout: int) -> list[str]:
    payload = request_json(
        YAHOO_SCREENER_URL,
        params={"scrIds": scr_id, "count": count},
        timeout=timeout,
    )
    quotes = (
        payload.get("finance", {})
        .get("result", [{}])[0]
        .get("quotes", [])
    )
    return dedupe([quote.get("symbol") for quote in quotes])


def yahoo_trending(timeout: int) -> list[str]:
    payload = request_json(YAHOO_TRENDING_URL, timeout=timeout)
    quotes = payload.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    return dedupe([quote.get("symbol") for quote in quotes])


def yahoo_premarket(timeout: int) -> list[str]:
    candidates = ["premarket_gainers", "premarket_movers", "day_losers"]
    found: list[str] = []
    for scr_id in candidates:
        try:
            found.extend(yahoo_screener(scr_id, 35, timeout))
        except Exception:
            continue
    return dedupe(found)


def parse_watchlist(value: str) -> list[str]:
    parts = re.split(r"[\s,;]+", value or "")
    return dedupe(parts)


@st.cache_data(ttl=60, show_spinner=False)
def discover_tickers(optional_watchlist: str, max_tickers: int, request_timeout: int) -> DiscoveryResult:
    sources: dict[str, list[str]] = {}
    source_calls = {
        "Yahoo day_gainers": lambda: yahoo_screener("day_gainers", 80, request_timeout),
        "Yahoo most_actives": lambda: yahoo_screener("most_actives", 80, request_timeout),
        "Yahoo trending": lambda: yahoo_trending(request_timeout),
        "Yahoo premarket": lambda: yahoo_premarket(request_timeout),
    }

    for name, loader in source_calls.items():
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
        merged = FALLBACK_RUNNERS.copy()
        sources["Fallback runners"] = merged
        message = "Fallback runner list active"
    else:
        fallback_fill = [ticker for ticker in FALLBACK_RUNNERS if ticker not in merged]
        merged.extend(fallback_fill[:30])
        sources["Fallback runners"] = fallback_fill[:30]
        message = "Live discovery active"

    tickers = dedupe(merged)[:max_tickers]
    return DiscoveryResult(
        tickers=tickers,
        sources={name: len(dedupe(values)) for name, values in sources.items()},
        message=message,
    )


@st.cache_data(ttl=20, show_spinner=False)
def quote_batch(symbols: tuple[str, ...], request_timeout: int) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    payload = request_json(
        YAHOO_QUOTE_URL,
        params={"symbols": ",".join(symbols)},
        timeout=request_timeout,
    )
    quotes = payload.get("quoteResponse", {}).get("result", [])
    return {quote.get("symbol"): quote for quote in quotes if quote.get("symbol")}


@st.cache_data(ttl=20, show_spinner=False)
def history_1d(symbol: str) -> pd.DataFrame:
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
    return max(0.0, now_utc_ts() - index.timestamp())


def vwap(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    volume = frame["Volume"].replace(0, np.nan)
    total_volume = safe_float(volume.sum(), 0)
    if total_volume <= 0:
        return np.nan
    return safe_float((typical * volume).sum() / total_volume, np.nan)


def rolling_base_low(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    lookback = frame.tail(min(45, len(frame)))
    return safe_float(lookback["Low"].min(), np.nan)


def volume_acceleration(frame: pd.DataFrame) -> float:
    if len(frame) < 8:
        return np.nan
    recent = frame["Volume"].tail(5).mean()
    prior = frame["Volume"].iloc[:-5].tail(20).mean()
    if prior <= 0 or np.isnan(prior):
        return np.nan
    return safe_float(recent / prior, np.nan)


def relative_volume(frame: pd.DataFrame, daily: pd.DataFrame) -> float:
    current_volume = safe_float(frame["Volume"].sum(), np.nan) if not frame.empty else np.nan
    if daily.empty or np.isnan(current_volume):
        return np.nan
    avg_daily_volume = safe_float(daily["Volume"].tail(20).mean(), np.nan)
    if avg_daily_volume <= 0 or np.isnan(avg_daily_volume):
        return np.nan
    market_minutes = 390
    elapsed = min(max(len(frame), 1), market_minutes)
    expected = avg_daily_volume * (elapsed / market_minutes)
    return safe_float(current_volume / expected, np.nan) if expected > 0 else np.nan


def invalid_row(symbol: str, reason: str = "Invalid data") -> dict[str, Any]:
    return {
        "ticker": symbol,
        "setup": "INVALID DATA",
        "mode": "Watched Movers",
        "trade_status": "Invalid",
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
        "source_price": np.nan,
        "last_updated": utc_clock(now_utc_ts()),
        "reason": reason,
    }


def normalize_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    defaults = invalid_row("")
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    frame["valid_trade"] = frame["valid_trade"].fillna(False).astype(bool)
    return frame


def compute_levels(symbol: str, quote: dict[str, Any], config: ScanConfig) -> dict[str, Any]:
    intraday = history_1d(symbol)
    daily = history_daily(symbol)

    if intraday.empty:
        return invalid_row(symbol, "No intraday data")

    current = safe_float(intraday["Close"].iloc[-1], np.nan)
    quote_price = safe_float(quote.get("regularMarketPrice") or quote.get("preMarketPrice"), np.nan)
    if not np.isnan(quote_price) and quote_price > 0:
        mismatch = abs(current - quote_price) / quote_price
    else:
        mismatch = 0.0

    previous_close = safe_float(
        quote.get("regularMarketPreviousClose")
        or (daily["Close"].iloc[-2] if len(daily) > 1 else np.nan),
        np.nan,
    )
    open_price = safe_float(intraday["Open"].iloc[0], np.nan)
    day_high = safe_float(intraday["High"].max(), np.nan)
    day_low = safe_float(intraday["Low"].min(), np.nan)
    day_volume = safe_int(intraday["Volume"].sum(), 0)
    vw = vwap(intraday)
    base_low = rolling_base_low(intraday)

    gain_pct = ((current - previous_close) / previous_close * 100) if previous_close > 0 else np.nan
    gap_pct = ((open_price - previous_close) / previous_close * 100) if previous_close > 0 else np.nan
    near_high_pct = ((day_high - current) / day_high * 100) if day_high > 0 else np.nan
    extension_from_base = ((current - base_low) / base_low * 100) if base_low > 0 else np.nan
    vol_accel = volume_acceleration(intraday)
    rvol = relative_volume(intraday, daily)
    stale_seconds = latest_bar_age_seconds(intraday)

    recent_high = safe_float(intraday["High"].tail(min(20, len(intraday))).max(), day_high)
    recent_low = safe_float(intraday["Low"].tail(min(20, len(intraday))).min(), day_low)
    breakout_level = round(max(recent_high, day_high) * 1.002, 4)
    nhod_level = round(day_high * 1.001, 4)
    support = round(max(day_low, min(vw if not np.isnan(vw) else day_low, recent_low)), 4)
    dip_low = round(max(support, vw * 0.985 if not np.isnan(vw) else support), 4)
    dip_high = round(max(dip_low, vw * 1.01 if not np.isnan(vw) else recent_low), 4)

    stop = round(min(dip_low, support) * 0.985, 4)
    if stop >= current:
        stop = round(current * 0.97, 4)
    risk = max(current - stop, 0)
    target_1 = round(current + risk * 2, 4) if risk > 0 else np.nan
    target_2 = round(current + risk * 3, 4) if risk > 0 else np.nan
    rr = ((target_1 - current) / risk) if risk > 0 else np.nan

    above_vwap = bool(current >= vw) if not np.isnan(vw) else False
    near_trigger = not np.isnan(near_high_pct) and near_high_pct <= 3.5
    inside_dip_zone = dip_low <= current <= dip_high if all(not np.isnan(x) for x in [dip_low, dip_high]) else False
    active_volume = (
        day_volume >= config.min_volume
        and (safe_float(rvol, 0) >= 1.2 or safe_float(vol_accel, 0) >= 1.3)
    )
    reclaiming_support = current >= support and current >= open_price * 0.995
    latest_not_stale = stale_seconds <= 20 * 60
    risk_ok = safe_float(rr, 0) >= 2.0 and risk > 0
    valid_trade = (near_trigger or inside_dip_zone) and risk_ok and latest_not_stale and (above_vwap or reclaiming_support) and active_volume

    tight_range = ((day_high - day_low) / current * 100) if current > 0 else np.nan
    recent_closes = intraday["Close"].tail(min(12, len(intraday)))
    higher_lows = len(intraday) >= 12 and bool(intraday["Low"].tail(12).iloc[-1] > intraday["Low"].tail(12).min())
    volume_ignition = safe_float(vol_accel, 0) >= 1.5 or safe_float(rvol, 0) >= 1.5
    lower_wick_bounce = bool(intraday["Close"].iloc[-1] > intraday["Open"].iloc[-1] and intraday["Low"].iloc[-1] <= dip_high)
    consolidation = safe_float(tight_range, 99) <= 9 and safe_float(near_high_pct, 99) <= 6
    resistance_nearby = current >= day_high * 0.985 if day_high > 0 else False

    if (
        np.isnan(current)
        or current <= 0
        or current < config.min_price
        or current > config.max_price
        or mismatch > 0.10
        or np.isnan(previous_close)
    ):
        setup = "INVALID DATA"
        mode = "Watched Movers"
        trade_status = "Invalid"
    elif safe_float(gain_pct, 0) > 30 and safe_float(extension_from_base, 0) > 18 and current > dip_high * 1.05:
        setup = "WAIT FOR PULLBACK"
        mode = "Watched Movers"
        trade_status = "Wait"
    elif current < vw and current < open_price:
        setup = "WAIT FOR REVERSAL"
        mode = "Watched Movers"
        trade_status = "Wait"
    elif valid_trade and safe_float(gain_pct, 0) >= 10 and near_trigger and safe_float(vol_accel, 0) >= 1.4:
        setup = "NHOD MOMENTUM" if current >= day_high * 0.995 else "MOMENTUM TRIGGER"
        mode = "Explosive Runners Now"
        trade_status = "Valid trade"
    elif valid_trade and inside_dip_zone and lower_wick_bounce:
        setup = "DIP BUY ZONE"
        mode = "Dip Buy Zones"
        trade_status = "Valid trade"
    elif valid_trade and consolidation and higher_lows and volume_ignition:
        setup = "BREAKOUT WATCH"
        mode = "Breakout Watch"
        trade_status = "Valid trade"
    elif resistance_nearby and (safe_float(rvol, 0) < 1.2 or safe_float(tight_range, 99) < 5):
        setup = "SCALP ONLY"
        mode = "Watched Movers"
        trade_status = "Scalp only"
    elif safe_float(gain_pct, 0) >= 10 and near_trigger and active_volume:
        setup = "BREAKOUT WATCH"
        mode = "Breakout Watch"
        trade_status = "Watch"
    elif inside_dip_zone and current >= support:
        setup = "DIP BUY ZONE"
        mode = "Dip Buy Zones"
        trade_status = "Watch"
    elif safe_float(gain_pct, 0) >= 15 and not near_trigger:
        setup = "WAIT FOR PULLBACK"
        mode = "Watched Movers"
        trade_status = "Wait"
    else:
        setup = "SCALP ONLY" if active_volume else "TOO LATE / AVOID"
        mode = "Watched Movers"
        trade_status = "Caution"

    score = 0
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
        "vwap": vw,
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
        "mismatch_pct": mismatch * 100,
        "valid_trade": valid_trade,
        "source_price": quote_price,
        "last_updated": utc_clock(now_utc_ts()),
    }


def plan_comments(row: pd.Series, lang: str) -> dict[str, str]:
    b = fmt_money(row["breakout_level"])
    d1 = fmt_money(row["dip_low"])
    d2 = fmt_money(row["dip_high"])
    stop = fmt_money(row["stop"])
    current = fmt_money(row["current"])
    v = fmt_money(row["vwap"])

    if is_arabic(lang):
        return {
            "confirmation": f"اختراق مستوى {b} مع حجم تداول قوي قد يفعّل استمرار الحركة.",
            "invalidation": f"الإلغاء إذا كسر السعر {stop} أو فقد VWAP عند {v}.",
            "why": f"السعر الحالي {current} قريب من الزخم، والحجم النسبي {fmt_x(row['rvol'])} مع تسارع حجم {fmt_x(row['vol_accel'])}.",
            "avoid": f"إذا رفض السعر مستوى {b}، تجنب المطاردة. السهم ممتد؛ انتظر رجوع نظيف عند الحاجة.",
            "dip": f"منطقة الشراء على النزول بين {d1} و {d2}، ووقف الخسارة تحت {stop}.",
        }
    return {
        "confirmation": f"Break above {b} with volume can trigger continuation.",
        "invalidation": f"Invalid if price loses {stop} or fails VWAP near {v}.",
        "why": f"Current price {current} is near the active level with {fmt_x(row['rvol'])} RVOL and {fmt_x(row['vol_accel'])} volume acceleration.",
        "avoid": f"If it rejects {b}, avoid chasing. Already extended names need a clean pullback.",
        "dip": f"Dip zone is {d1}-{d2}; stop under {stop}.",
    }


def scan_universe(tickers: list[str], config: ScanConfig, progress=None) -> pd.DataFrame:
    started = time.monotonic()
    deadline = started + config.scan_timeout
    rows: list[dict[str, Any]] = []
    scanned = 0

    quotes: dict[str, dict[str, Any]] = {}
    for i in range(0, len(tickers), 50):
        batch = tuple(tickers[i : i + 50])
        try:
            quotes.update(quote_batch(batch, config.request_timeout))
        except Exception:
            continue

    executor = ThreadPoolExecutor(max_workers=config.workers)
    futures = {
        executor.submit(compute_levels, ticker, quotes.get(ticker, {}), config): ticker
        for ticker in tickers
    }
    pending = set(futures)
    try:
        while pending and time.monotonic() < deadline:
            done, pending = wait(
                pending,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                if time.monotonic() > deadline:
                    break
                scanned += 1
                try:
                    row = future.result(timeout=0)
                except Exception as exc:
                    row = invalid_row(futures[future], str(exc)[:100])
                rows.append(row)
                if progress is not None:
                    progress.progress(min(scanned / max(len(tickers), 1), 1.0))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = normalize_scan_frame(frame)
    for column in ["score", "gain_pct", "rvol", "vol_accel", "near_high_pct", "rr", "volume"]:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.sort_values(
        by=["valid_trade", "score", "gain_pct", "rvol"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return frame.reset_index(drop=True)


def apply_theme(lang: str) -> None:
    direction = "rtl" if is_arabic(lang) else "ltr"
    align = "right" if is_arabic(lang) else "left"
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: dark;
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(19, 92, 92, .24), transparent 30rem),
                linear-gradient(135deg, #08100f 0%, #0b1118 52%, #10130f 100%);
            color: #f2f7f4;
            direction: {direction};
        }}
        .block-container {{
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, p, label, div[data-testid="stMarkdownContainer"] {{
            text-align: {align};
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, .055);
            border: 1px solid rgba(255, 255, 255, .10);
            border-radius: 8px;
            padding: .85rem 1rem;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: #aebbb5;
        }}
        div[data-testid="stMetricValue"] {{
            color: #f4fff9;
        }}
        section[data-testid="stSidebar"] {{
            background: #09100f;
        }}
        .stButton button {{
            border-radius: 8px;
            border: 1px solid rgba(64, 224, 171, .35);
            background: #123c34;
            color: #effff8;
            min-height: 2.5rem;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255, 255, 255, .08);
            border-radius: 8px;
            overflow: hidden;
        }}
        .live-metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .75rem;
            direction: {direction};
            margin: .75rem 0 1rem;
        }}
        .live-metric-card {{
            position: relative;
            min-height: 112px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, .11);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, .075), rgba(255, 255, 255, .035)),
                rgba(7, 15, 17, .92);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, .06), 0 16px 34px rgba(0, 0, 0, .24);
            padding: .85rem .95rem;
            overflow: hidden;
            text-align: {align};
            transition: border-color .28s ease, box-shadow .28s ease, transform .28s ease;
        }}
        .live-metric-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0;
            transition: opacity .35s ease;
        }}
        .live-metric-card.up {{
            border-color: rgba(45, 212, 128, .72);
            animation: pulseGreen 900ms ease-out 1, glowGreen 1200ms ease-out 1;
        }}
        .live-metric-card.down {{
            border-color: rgba(248, 113, 113, .72);
            animation: pulseRed 900ms ease-out 1, glowRed 1200ms ease-out 1;
        }}
        .live-metric-card.up::after {{
            background: linear-gradient(90deg, rgba(34, 197, 94, .18), transparent 70%);
            opacity: 1;
            animation: fadeFlash 900ms ease-out 1 forwards;
        }}
        .live-metric-card.down::after {{
            background: linear-gradient(90deg, rgba(239, 68, 68, .18), transparent 70%);
            opacity: 1;
            animation: fadeFlash 900ms ease-out 1 forwards;
        }}
        .live-metric-label {{
            color: #aebbb5;
            font-size: .78rem;
            line-height: 1.2;
            margin-bottom: .45rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .live-metric-value {{
            color: #f5fff9;
            font-size: clamp(1.35rem, 2.4vw, 2.05rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.05;
            font-variant-numeric: tabular-nums;
            transition: color .25s ease, transform .25s ease;
        }}
        .live-metric-card.up .live-metric-value {{
            color: #86efac;
            transform: translateY(-1px);
        }}
        .live-metric-card.down .live-metric-value {{
            color: #fca5a5;
            transform: translateY(1px);
        }}
        .live-metric-delta {{
            display: inline-flex;
            align-items: center;
            gap: .25rem;
            margin-top: .55rem;
            min-height: 1.35rem;
            border-radius: 999px;
            padding: .18rem .5rem;
            font-size: .78rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            border: 1px solid rgba(255, 255, 255, .10);
            color: #9aa8a2;
            background: rgba(255, 255, 255, .045);
        }}
        .live-metric-delta.up {{
            color: #86efac;
            background: rgba(34, 197, 94, .12);
            border-color: rgba(34, 197, 94, .28);
        }}
        .live-metric-delta.down {{
            color: #fca5a5;
            background: rgba(239, 68, 68, .12);
            border-color: rgba(239, 68, 68, .28);
        }}
        .live-metric-delta.flat {{
            color: #9aa8a2;
        }}
        @keyframes pulseGreen {{
            0% {{ background-color: rgba(34, 197, 94, .24); }}
            100% {{ background-color: rgba(7, 15, 17, .92); }}
        }}
        @keyframes pulseRed {{
            0% {{ background-color: rgba(239, 68, 68, .24); }}
            100% {{ background-color: rgba(7, 15, 17, .92); }}
        }}
        @keyframes glowGreen {{
            0% {{ box-shadow: 0 0 0 rgba(255,255,255,0), 0 16px 34px rgba(0,0,0,.24); }}
            35% {{ box-shadow: 0 0 28px rgba(45, 212, 128, .18), 0 18px 42px rgba(0,0,0,.30); }}
            100% {{ box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 16px 34px rgba(0,0,0,.24); }}
        }}
        @keyframes glowRed {{
            0% {{ box-shadow: 0 0 0 rgba(255,255,255,0), 0 16px 34px rgba(0,0,0,.24); }}
            35% {{ box-shadow: 0 0 28px rgba(248, 113, 113, .20), 0 18px 42px rgba(0,0,0,.30); }}
            100% {{ box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 16px 34px rgba(0,0,0,.24); }}
        }}
        @keyframes fadeFlash {{
            0% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
        @media (max-width: 980px) {{
            .live-metric-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 560px) {{
            .live-metric-grid {{
                grid-template-columns: 1fr;
            }}
            .live-metric-card {{
                min-height: 96px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_theme(lang: str) -> None:
    apply_theme(lang)


def format_live_value(value: Any) -> str:
    number = safe_float(value, np.nan)
    if np.isnan(number):
        return "N/A"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) < 1:
        return f"{number:.4f}"
    return f"{number:,.2f}"


def begin_live_metric_cycle() -> None:
    if "live_metric_previous_values" not in st.session_state:
        st.session_state.live_metric_previous_values = {}
    st.session_state.live_metric_pending_values = {}


def commit_live_metric_cycle() -> None:
    pending = st.session_state.get("live_metric_pending_values", {})
    previous = st.session_state.get("live_metric_previous_values", {})
    st.session_state.live_metric_previous_values = {**previous, **pending}


def get_metric_direction(key: str, current_value: Any) -> tuple[str, float]:
    current = safe_float(current_value, np.nan)
    pending = st.session_state.setdefault("live_metric_pending_values", {})
    previous_values = st.session_state.setdefault("live_metric_previous_values", {})
    pending[key] = current

    previous = safe_float(previous_values.get(key), np.nan)
    if np.isnan(current) or np.isnan(previous):
        return "flat", 0.0

    tolerance = max(abs(previous), 1.0) * 0.00001
    delta = current - previous
    if delta > tolerance:
        return "up", delta
    if delta < -tolerance:
        return "down", delta
    return "flat", 0.0


def render_live_metric_card(
    label: str,
    value: Any,
    key: str,
    prefix: str = "",
    suffix: str = "",
) -> None:
    direction, delta = get_metric_direction(key, value)
    arrow = {"up": "▲", "down": "▼", "flat": ""}[direction]
    lang = st.session_state.get("language", "English")
    delta_text = ("بدون تغيير" if is_arabic(lang) else "No change") if direction == "flat" else f"{arrow} {delta:+.2f}"
    formatted = format_live_value(value)
    value_text = value if isinstance(value, str) else ("N/A" if formatted == "N/A" else f"{prefix}{formatted}{suffix}")
    st.markdown(
        f"""
        <div class="live-metric-card {direction}">
            <div class="live-metric-label">{html.escape(label)}</div>
            <div class="live-metric-value">{html.escape(value_text)}</div>
            <div class="live-metric-delta {direction}">{html.escape(delta_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_metric_grid(metrics: list[dict[str, Any]]) -> None:
    cards = []
    lang = st.session_state.get("language", "English")
    for metric in metrics:
        direction, delta = get_metric_direction(metric["key"], metric["value"])
        arrow = {"up": "▲", "down": "▼", "flat": ""}[direction]
        delta_text = ("بدون تغيير" if is_arabic(lang) else "No change") if direction == "flat" else f"{arrow} {delta:+.2f}"
        raw_value = metric["value"]
        formatted = format_live_value(raw_value)
        value_text = str(raw_value) if isinstance(raw_value, str) else (
            "N/A" if formatted == "N/A" else f"{metric.get('prefix', '')}{formatted}{metric.get('suffix', '')}"
        )
        cards.append(
            f"""
            <div class="live-metric-card {direction}">
                <div class="live-metric-label">{html.escape(str(metric["label"]))}</div>
                <div class="live-metric-value">{html.escape(value_text)}</div>
                <div class="live-metric-delta {direction}">{html.escape(delta_text)}</div>
            </div>
            """
        )
    st.markdown(f"<div class=\"live-metric-grid\">{''.join(cards)}</div>", unsafe_allow_html=True)


def render_trade_card(row: pd.Series, lang: str) -> None:
    comments = plan_comments(row, lang)
    setup = setup_text(str(row["setup"]), lang)
    status_color = "green" if bool(row.get("valid_trade", False)) else "orange"
    if row["setup"] in {"INVALID DATA", "TOO LATE / AVOID"}:
        status_color = "red"

    with st.container(border=True):
        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            st.subheader(f"{row['ticker']} · {setup}")
            st.caption(f"{tr('mode', lang)}: {tr(mode_key(str(row['mode'])), lang)}")
        with top_right:
            st.markdown(f":{status_color}[{row['trade_status']}]")

        ticker_key = str(row["ticker"])
        render_live_metric_grid(
            [
                {
                    "label": tr("current", lang),
                    "value": row["current"],
                    "key": f"{ticker_key}:current",
                    "prefix": "$",
                },
                {
                    "label": tr("break", lang),
                    "value": row["breakout_level"],
                    "key": f"{ticker_key}:breakout_level",
                    "prefix": "$",
                },
                {
                    "label": tr("stop", lang),
                    "value": row["stop"],
                    "key": f"{ticker_key}:stop",
                    "prefix": "$",
                },
                {
                    "label": tr("rvol", lang),
                    "value": row["rvol"],
                    "key": f"{ticker_key}:rvol",
                    "suffix": "x",
                },
                {
                    "label": tr("vol_acc", lang),
                    "value": row["vol_accel"],
                    "key": f"{ticker_key}:vol_accel",
                    "suffix": "x",
                },
                {
                    "label": tr("target1", lang),
                    "value": row["target_1"],
                    "key": f"{ticker_key}:target_1",
                    "prefix": "$",
                },
                {
                    "label": tr("target2", lang),
                    "value": row["target_2"],
                    "key": f"{ticker_key}:target_2",
                    "prefix": "$",
                },
                {
                    "label": tr("score", lang),
                    "value": row["score"],
                    "key": f"{ticker_key}:score",
                    "suffix": "/100",
                },
            ]
        )

        info = {
            tr("ticker", lang): row["ticker"],
            tr("setup", lang): setup,
            tr("current", lang): fmt_money(row["current"]),
            tr("break", lang): fmt_money(row["breakout_level"]),
            tr("dip_zone", lang): f"{fmt_money(row['dip_low'])} - {fmt_money(row['dip_high'])}",
            tr("stop", lang): fmt_money(row["stop"]),
            tr("target1", lang): fmt_money(row["target_1"]),
            tr("target2", lang): fmt_money(row["target_2"]),
            tr("rr", lang): fmt_rr(row["rr"]),
        }
        st.dataframe(pd.DataFrame([info]), use_container_width=True, hide_index=True)

        st.write(f"**{tr('confirmation', lang)}**")
        st.write(comments["confirmation"])
        st.write(f"**{tr('invalidation', lang)}**")
        st.write(comments["invalidation"])
        st.write(f"**{tr('why', lang)}**")
        st.write(comments["why"])
        st.write(f"**{tr('avoid', lang)}**")
        st.write(comments["avoid"])

        with st.expander(tr("details", lang)):
            detail = pd.DataFrame(
                [
                    {
                        "Open": fmt_money(row["open"]),
                        "Day high": fmt_money(row["day_high"]),
                        "Day low": fmt_money(row["day_low"]),
                        "NHOD": fmt_money(row["nhod_level"]),
                        "VWAP": fmt_money(row["vwap"]),
                        "Support": fmt_money(row["support"]),
                        "Extension": fmt_pct(row["extension_from_base"]),
                        "Gap": fmt_pct(row["gap_pct"]),
                        "Volume": f"{safe_int(row['volume']):,}",
                        "Data age": f"{safe_float(row['stale_seconds'], 0) / 60:.1f} min",
                    }
                ]
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)


def render_plan_card(row: pd.Series, lang: str) -> None:
    render_trade_card(row, lang)


def mode_key(mode: str) -> str:
    return {
        "Explosive Runners Now": "explosive",
        "Breakout Watch": "breakout",
        "Dip Buy Zones": "dip",
        "Watched Movers": "watched",
    }.get(mode, "watched")


def section_filter(frame: pd.DataFrame, section: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if section == "Explosive Runners Now":
        return frame[frame["mode"].eq(section)].head(5)
    if section == "Breakout Watch":
        return frame[frame["setup"].isin(["BREAKOUT WATCH", "MOMENTUM TRIGGER", "NHOD MOMENTUM"])].head(5)
    if section == "Dip Buy Zones":
        return frame[frame["setup"].eq("DIP BUY ZONE")].head(5)
    return frame[~frame["mode"].isin(["Explosive Runners Now", "Dip Buy Zones"])].head(5)


def compact_table(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                tr("ticker", lang): row["ticker"],
                tr("setup", lang): setup_text(str(row["setup"]), lang),
                tr("price", lang): fmt_money(row["current"]),
                tr("gain", lang): fmt_pct(row["gain_pct"]),
                tr("rvol", lang): fmt_x(row["rvol"]),
                tr("near_high", lang): fmt_pct(row["near_high_pct"]),
                tr("rr", lang): fmt_rr(row["rr"]),
                tr("status", lang): row["trade_status"],
            }
        )
    return pd.DataFrame(rows)


def render_section(frame: pd.DataFrame, title_key: str, source_mode: str, lang: str, max_cards: int) -> None:
    st.header(tr(title_key, lang))
    subset = section_filter(frame, source_mode).head(max_cards)
    if subset.empty:
        st.info(tr("no_results", lang))
        return
    st.dataframe(compact_table(subset, lang), use_container_width=True, hide_index=True)
    for _, row in subset.iterrows():
        render_plan_card(row, lang)


def initialize_state() -> None:
    defaults = {
        "scan_df": pd.DataFrame(),
        "last_scan_ts": None,
        "source_counts": {},
        "discovery_message": "",
        "optional_watchlist": "",
        "live_metric_previous_values": {},
        "live_metric_pending_values": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def should_scan(interval: int) -> bool:
    last = st.session_state.get("last_scan_ts")
    if not last:
        return True
    return now_utc_ts() - float(last) >= interval


def run_scan(config: ScanConfig, optional_watchlist: str, lang: str) -> None:
    discovery = discover_tickers(optional_watchlist, config.max_tickers, config.request_timeout)
    progress = st.progress(0, text="Scanning..." if not is_arabic(lang) else "جار الفحص...")
    frame = scan_universe(discovery.tickers, config, progress=progress)
    progress.empty()
    st.session_state.scan_df = frame
    st.session_state.last_scan_ts = now_utc_ts()
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
        st.subheader(tr("scanner_controls", lang))
        cloud_fast = st.toggle(tr("cloud_fast", lang), value=True)
        config = ScanConfig() if cloud_fast else ScanConfig(max_tickers=220, workers=10, scan_timeout=28)
        with st.expander(tr("advanced_watchlist", lang)):
            optional_watchlist = st.text_area(
                tr("advanced_watchlist", lang),
                value=st.session_state.optional_watchlist,
                help=tr("advanced_help", lang),
                height=90,
                label_visibility="collapsed",
            )
            st.session_state.optional_watchlist = optional_watchlist
        scan_clicked = st.button(tr("refresh", lang), use_container_width=True)
        st.caption(f"{tr('next_refresh', lang)}: {config.scan_interval}s")

    apply_theme(lang)

    st.title(tr("page_title", lang))
    st.caption(tr("subtitle", lang))

    if scan_clicked or should_scan(config.scan_interval):
        run_scan(config, st.session_state.optional_watchlist, lang)

    frame: pd.DataFrame = st.session_state.scan_df
    valid_count = int(frame.get("valid_trade", pd.Series(dtype=bool)).fillna(False).sum()) if not frame.empty else 0
    hot_count = int((frame.get("gain_pct", pd.Series(dtype=float)).fillna(0) >= 10).sum()) if not frame.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("last_scan", lang), utc_clock(st.session_state.last_scan_ts))
    k2.metric(tr("universe", lang), f"{len(frame):,}")
    k3.metric(tr("valid", lang), f"{valid_count:,}")
    k4.metric(tr("hot", lang), f"{hot_count:,}")

    with st.expander(tr("sources", lang)):
        sources = pd.DataFrame(
            [{"Source": name, "Tickers": count} for name, count in st.session_state.source_counts.items()]
        )
        st.write(st.session_state.discovery_message)
        st.dataframe(sources, use_container_width=True, hide_index=True)

    begin_live_metric_cycle()
    if frame.empty:
        st.warning(tr("no_results", lang))
    else:
        render_section(frame, "explosive", "Explosive Runners Now", lang, config.max_cards_per_section)
        render_section(frame, "breakout", "Breakout Watch", lang, config.max_cards_per_section)
        render_section(frame, "dip", "Dip Buy Zones", lang, config.max_cards_per_section)
        render_section(frame, "watched", "Watched Movers", lang, config.max_cards_per_section)

        with st.expander(tr("table", lang)):
            st.dataframe(compact_table(frame.head(50), lang), use_container_width=True, hide_index=True)
    commit_live_metric_cycle()

    st.caption(tr("footer", lang))

    elapsed = now_utc_ts() - float(st.session_state.last_scan_ts or now_utc_ts())
    if elapsed >= config.scan_interval:
        st.rerun()


if __name__ == "__main__":
    main()
