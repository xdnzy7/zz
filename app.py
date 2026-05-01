from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
#  Momentum Playbook Scanner  ·  Professional Bilingual Edition
#  Not financial advice — educational / trade planning only.
# ═══════════════════════════════════════════════════════════════

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
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


# ── Constants ────────────────────────────────────────────────

APP_VERSION = "momentum-playbook-bilingual-2026-05-01"

YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
YAHOO_QUOTE_URL    = "https://query1.finance.yahoo.com/v7/finance/quote"
VALID_TICKER_RE    = re.compile(r"^[A-Z]{1,5}$")

CLOUD_MAX_TICKERS  = 150
CLOUD_WORKERS      = 8
CLOUD_SCAN_TIMEOUT = 20
CLOUD_SCAN_INTERVAL = 45
HOT_POOL_TTL       = 3600
HOT_POOL_MAX       = 80

PSYCH_LEVELS = [
    0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00,
    7.50, 10.00, 12.50, 15.00, 20.00, 25.00, 30.00, 40.00, 50.00,
]

FALLBACK_RUNNERS = [
    "AKAN", "HCAI", "AIOS", "CUE", "RDAC", "XTLB", "BIYA", "OSRH",
    "SKLZ", "CANF", "WNW", "MRAM", "MXL", "CTXR", "BYND", "HUN",
    "MICC", "BYON", "SOUN", "BBAI", "KULR", "PLUG", "OPEN", "FFIE",
    "HOLO", "GNS", "LUCY", "WISA", "TIVC", "ATER", "GFAI", "CXAI",
    "WULF", "BITF", "QBTS", "IONQ", "RGTI", "SERV", "LUNR", "ACHR",
]

# ── Setup labels ─────────────────────────────────────────────

SETUP_EN: dict[str, str] = {
    "EARLY FIRE":         "EARLY FIRE",
    "BUILDING MOMENTUM":  "BUILDING MOMENTUM",
    "BREAKOUT WATCH":     "BREAKOUT WATCH",
    "ENTRY READY":        "ENTRY READY",
    "DIP BUY ZONE":       "DIP BUY ZONE",
    "NHOD MOMENTUM":      "NHOD MOMENTUM",
    "EXPLOSIVE RUNNER":   "EXPLOSIVE RUNNER",
    "HOT RUNNER WAIT":    "HOT RUNNER — WAIT FOR PULLBACK",
    "AFTER-HOURS RUNNER": "AFTER-HOURS RUNNER",
    "PREMARKET RUNNER":   "PREMARKET RUNNER",
    "SCALP ONLY":         "SCALP ONLY",
    "WAIT FOR PULLBACK":  "WAIT FOR PULLBACK",
    "WAIT FOR REVERSAL":  "WAIT FOR REVERSAL",
    "TOO LATE / AVOID":   "TOO LATE / AVOID",
    "INVALID DATA":       "INVALID DATA",
}

SETUP_AR: dict[str, str] = {
    "EARLY FIRE":         "بداية اشتعال",
    "BUILDING MOMENTUM":  "زخم يتكوّن",
    "BREAKOUT WATCH":     "مراقبة اختراق",
    "ENTRY READY":        "الدخول جاهز",
    "DIP BUY ZONE":       "منطقة شراء على النزول",
    "NHOD MOMENTUM":      "أعلى سعر جديد اليوم",
    "EXPLOSIVE RUNNER":   "سهم منفجر",
    "HOT RUNNER WAIT":    "سهم ساخن — انتظر رجوع السعر",
    "AFTER-HOURS RUNNER": "سهم يتحرك بعد الإغلاق",
    "PREMARKET RUNNER":   "سهم يتحرك قبل الافتتاح",
    "SCALP ONLY":         "مضاربة سريعة فقط",
    "WAIT FOR PULLBACK":  "انتظر رجوع السعر",
    "WAIT FOR REVERSAL":  "انتظر انعكاس",
    "TOO LATE / AVOID":   "متأخر / تجنب",
    "INVALID DATA":       "بيانات غير صالحة",
}

# ── Bilingual UI text ─────────────────────────────────────────

TEXT: dict[str, dict[str, str]] = {
    "English": {
        "page_title":         "🔥 Momentum Playbook Scanner",
        "subtitle":           "Automatic small-cap runner discovery · early ignition · breakouts · dips · explosive runners · Not financial advice.",
        "language":           "Language",
        "refresh":            "Scan Now",
        "last_scan":          "Last scan",
        "next_refresh":       "Auto-refresh",
        "universe":           "Scanned",
        "valid":              "Valid setups",
        "hot_pool":           "Hot pool",
        "scanner_controls":   "Scanner Controls",
        "cloud_fast":         "Cloud fast mode",
        "advanced_watchlist": "Optional watchlist (Advanced)",
        "advanced_help":      "Optional only. Scanner discovers tickers automatically.",
        "sources":            "Discovery sources",
        "debug":              "Debug diagnostics",
        "early_fire":         "🔥 Early Fire",
        "entry_ready":        "✅ Entry Ready",
        "dip_zone":           "🎯 Dip Buy Zones",
        "breakout_watch":     "📈 Breakout Watch",
        "explosive":          "🚀 Explosive Runners",
        "ah_pm":              "🌙 After-hours / Pre-market Movers",
        "hot_pool_section":   "🟡 Hot Pool",
        "table":              "Compact table (top 60)",
        "details":            "Details",
        "no_results":         "No clean setups here. Scanner is intentionally selective.",
        "ticker":             "Ticker",
        "setup":              "Setup",
        "current":            "Price",
        "gain":               "Gain",
        "session":            "Session",
        "rvol":               "RVOL",
        "vol_acc":            "Vol Accel",
        "near_high":          "Near High",
        "gap":                "Gap",
        "break_level":        "Break",
        "dip_zone_label":     "Dip zone",
        "stop":               "Stop",
        "t1":                 "T1",
        "t2":                 "T2",
        "rr":                 "R/R",
        "data_q":             "Data",
        "score_early":        "Early",
        "score_expl":         "Expl",
        "source":             "Source",
        "confirmation":       "Confirmation",
        "invalidation":       "Invalidation",
        "playbook":           "Playbook",
        "avoid":              "Avoid if",
        "footer":             "Not financial advice. Educational / trade planning only. The scanner filters conditions — it does not predict outcomes.",
        "hot_pool_lbl":       "Hot Pool",
        "trade_status":       "Status",
        "higher_lows":        "Higher lows",
        "tight_consol":       "Tight consol",
        "vwap_reclaim":       "VWAP reclaim",
    },
    "Arabic": {
        "page_title":         "🔥 ماسح خطط الزخم الاحترافي",
        "subtitle":           "اكتشاف تلقائي للأسهم الصغيرة · اشتعال مبكر · اختراقات · مناطق شراء · أسهم منفجرة · ليست نصيحة مالية.",
        "language":           "اللغة",
        "refresh":            "افحص الآن",
        "last_scan":          "آخر فحص",
        "next_refresh":       "التحديث التلقائي",
        "universe":           "عدد المفحوصة",
        "valid":              "فرص صالحة",
        "hot_pool":           "القائمة الحارة",
        "scanner_controls":   "إعدادات الماسح",
        "cloud_fast":         "وضع السحابة السريع",
        "advanced_watchlist": "قائمة مراقبة اختيارية",
        "advanced_help":      "اختياري فقط. الماسح يكتشف الرموز تلقائياً.",
        "sources":            "مصادر الاكتشاف",
        "debug":              "بيانات التشخيص",
        "early_fire":         "🔥 بداية اشتعال",
        "entry_ready":        "✅ الدخول جاهز",
        "dip_zone":           "🎯 مناطق الشراء على النزول",
        "breakout_watch":     "📈 مراقبة الاختراق",
        "explosive":          "🚀 الأسهم المنفجرة",
        "ah_pm":              "🌙 تحركات قبل / بعد السوق",
        "hot_pool_section":   "🟡 القائمة الحارة",
        "table":              "جدول مختصر (أعلى 60)",
        "details":            "تفاصيل",
        "no_results":         "لا توجد فرص نظيفة هنا. الماسح انتقائي عمداً.",
        "ticker":             "الرمز",
        "setup":              "الفرصة",
        "current":            "السعر",
        "gain":               "الصعود",
        "session":            "الجلسة",
        "rvol":               "ح. نسبي",
        "vol_acc":            "تسارع",
        "near_high":          "من القمة",
        "gap":                "فجوة",
        "break_level":        "الاختراق",
        "dip_zone_label":     "منطقة النزول",
        "stop":               "الوقف",
        "t1":                 "ه1",
        "t2":                 "ه2",
        "rr":                 "ع/م",
        "data_q":             "البيانات",
        "score_early":        "مبكر",
        "score_expl":         "انفجار",
        "source":             "المصدر",
        "confirmation":       "التأكيد",
        "invalidation":       "الإلغاء",
        "playbook":           "خطة التداول",
        "avoid":              "تجنب إذا",
        "footer":             "ليست نصيحة مالية. للتعليم والتخطيط فقط. الماسح يرشح الشروط ولا يتنبأ بالنتائج.",
        "hot_pool_lbl":       "القائمة الحارة",
        "trade_status":       "الحالة",
        "higher_lows":        "قيعان متصاعدة",
        "tight_consol":       "تماسك ضيق",
        "vwap_reclaim":       "استعادة VWAP",
    },
}


# ══════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════

def tr(key: str, lang: str) -> str:
    return TEXT[lang].get(key, key)


def is_arabic(lang: str) -> bool:
    return lang == "Arabic"


def setup_label(code: str, lang: str) -> str:
    d = SETUP_AR if is_arabic(lang) else SETUP_EN
    return d.get(code, code)


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def safe_int(v: Any, default: int = 0) -> int:
    n = safe_float(v, float("nan"))
    return default if math.isnan(n) else int(n)


def clean_symbol(s: Any) -> str | None:
    v = str(s or "").upper().strip().replace(".", "-")
    return v if VALID_TICKER_RE.fullmatch(v) else None


def dedupe(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        sym = clean_symbol(item)
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def utc_clock(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%H:%M:%S UTC")


def fmt_money(v: Any) -> str:
    n = safe_float(v, float("nan"))
    if math.isnan(n):
        return "N/A"
    digits = 4 if n < 1 else (3 if n < 5 else 2)
    return f"${n:,.{digits}f}"


def fmt_pct(v: Any) -> str:
    n = safe_float(v, float("nan"))
    if math.isnan(n):
        return "N/A"
    sign = "+" if n >= 0 else ""
    return f"{sign}{n:.1f}%"


def fmt_x(v: Any) -> str:
    n = safe_float(v, float("nan"))
    return "N/A" if math.isnan(n) else f"{n:.2f}x"


def fmt_rr(v: Any) -> str:
    n = safe_float(v, float("nan"))
    return "N/A" if math.isnan(n) else f"1:{n:.1f}"


def nearest_psych_above(price: float) -> float:
    above = [p for p in PSYCH_LEVELS if p > price]
    return above[0] if above else round(price * 1.10, 2)


def breakout_buffer(price: float) -> float:
    if price < 1:
        return 0.008
    if price < 5:
        return 0.02
    if price < 20:
        return 0.05
    return round(price * 0.003, 3)


_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def request_json(url: str, params: dict[str, Any] | None = None, timeout: int = 6) -> dict[str, Any]:
    r = requests.get(url, params=params or {}, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()  # type: ignore[no-any-return]


# ══════════════════════════════════════════════════════════════
#  DISCOVERY LAYER
# ══════════════════════════════════════════════════════════════

@dataclass
class DiscoveredTicker:
    symbol: str
    source: str
    price: float = float("nan")
    gain_pct: float = float("nan")
    session_type: str = "UNKNOWN"


@dataclass
class DiscoveryResult:
    tickers: list[str]
    sources: dict[str, int]
    message: str
    discovered: list[DiscoveredTicker] = field(default_factory=list)


def _parse_quote(q: dict[str, Any], source: str) -> DiscoveredTicker | None:
    sym = clean_symbol(q.get("symbol"))
    if not sym:
        return None
    price = safe_float(
        q.get("regularMarketPrice")
        or q.get("preMarketPrice")
        or q.get("postMarketPrice")
    )
    gain = safe_float(
        q.get("regularMarketChangePercent")
        or q.get("preMarketChangePercent")
        or q.get("postMarketChangePercent")
    )
    state = str(q.get("marketState", "REGULAR")).upper()
    session = "PREMARKET" if state in ("PRE", "PREMARKET") else (
        "AFTER HOURS" if state in ("POST", "POSTMARKET") else "REGULAR"
    )
    return DiscoveredTicker(symbol=sym, source=source, price=price, gain_pct=gain, session_type=session)


def _yahoo_screener_raw(scr_id: str, count: int, timeout: int) -> list[dict[str, Any]]:
    try:
        data = request_json(
            YAHOO_SCREENER_URL,
            params={"scrIds": scr_id, "count": count},
            timeout=timeout,
        )
        return (
            data.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
    except Exception:
        return []


def _yahoo_trending_raw(timeout: int) -> list[dict[str, Any]]:
    try:
        data = request_json(YAHOO_TRENDING_URL, timeout=timeout)
        return data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    except Exception:
        return []


def _scrape_stockanalysis(timeout: int) -> list[str]:
    try:
        r = requests.get(
            "https://stockanalysis.com/markets/gainers/",
            headers=_HEADERS, timeout=timeout,
        )
        return dedupe(re.findall(r'href="/stocks/([A-Z]{1,5})/"', r.text)[:50])
    except Exception:
        return []


def _scrape_finviz(timeout: int) -> list[str]:
    try:
        r = requests.get(
            "https://finviz.com/screener.ashx?v=111&s=ta_topgainers",
            headers=_HEADERS, timeout=timeout,
        )
        return dedupe(re.findall(r'ticker=([A-Z]{1,5})', r.text)[:50])
    except Exception:
        return []


def _scrape_nasdaq(timeout: int) -> list[str]:
    try:
        r = requests.get(
            "https://www.nasdaq.com/market-activity/stocks/most-active",
            headers={**_HEADERS, "Accept": "text/html"},
            timeout=timeout,
        )
        syms = re.findall(r'/market-activity/stocks/([a-z]{1,5})"', r.text)
        return dedupe([s.upper() for s in syms[:50]])
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def discover_tickers(optional_watchlist: str, max_tickers: int, timeout: int) -> DiscoveryResult:
    src_map: dict[str, list[DiscoveredTicker]] = {}

    # Yahoo screeners
    for scr_id, name in [
        ("day_gainers",  "Yahoo day_gainers"),
        ("most_actives", "Yahoo most_actives"),
    ]:
        quotes = _yahoo_screener_raw(scr_id, 80, timeout)
        src_map[name] = [d for q in quotes if (d := _parse_quote(q, name))]

    # Trending
    tq = _yahoo_trending_raw(timeout)
    src_map["Yahoo trending"] = [d for q in tq if (d := _parse_quote(q, "Yahoo trending"))]

    # Premarket screeners (try multiple IDs)
    pm: list[DiscoveredTicker] = []
    for scr_id in ("premarket_gainers", "premarket_movers", "pre_market_gainers", "pre_market_most_actives"):
        for q in _yahoo_screener_raw(scr_id, 40, timeout):
            d = _parse_quote(q, "Yahoo premarket")
            if d:
                pm.append(d)
    src_map["Yahoo premarket"] = pm

    # Web scraping (silent fail)
    src_map["StockAnalysis"] = [DiscoveredTicker(s, "StockAnalysis") for s in _scrape_stockanalysis(timeout)]
    src_map["Finviz"]         = [DiscoveredTicker(s, "Finviz")        for s in _scrape_finviz(timeout)]
    src_map["Nasdaq"]         = [DiscoveredTicker(s, "Nasdaq")        for s in _scrape_nasdaq(timeout)]

    # Optional watchlist
    if optional_watchlist.strip():
        parsed = dedupe(re.split(r"[\s,;]+", optional_watchlist))
        src_map["Watchlist"] = [DiscoveredTicker(s, "Watchlist") for s in parsed]

    # Merge unique
    seen: set[str] = set()
    all_discovered: list[DiscoveredTicker] = []
    for items in src_map.values():
        for d in items:
            if d.symbol not in seen:
                seen.add(d.symbol)
                all_discovered.append(d)

    if not all_discovered:
        fb = [DiscoveredTicker(s, "Fallback") for s in FALLBACK_RUNNERS]
        src_map["Fallback"] = fb
        all_discovered = fb
        message = "⚠️ All live sources failed — using fallback runner list"
    else:
        # Pad with fallback runners not yet discovered
        fb_extra: list[DiscoveredTicker] = []
        for s in FALLBACK_RUNNERS:
            if s not in seen and len(all_discovered) < max_tickers:
                all_discovered.append(DiscoveredTicker(s, "Fallback"))
                fb_extra.append(DiscoveredTicker(s, "Fallback"))
                seen.add(s)
        if fb_extra:
            src_map["Fallback"] = fb_extra
        message = "✅ Live discovery active"

    final_syms = dedupe(d.symbol for d in all_discovered)[:max_tickers]
    final_set  = set(final_syms)
    final_disc = [d for d in all_discovered if d.symbol in final_set]

    return DiscoveryResult(
        tickers   = final_syms,
        sources   = {name: len({d.symbol for d in items}) for name, items in src_map.items() if items},
        message   = message,
        discovered= final_disc,
    )


# ══════════════════════════════════════════════════════════════
#  HOT POOL
# ══════════════════════════════════════════════════════════════

@dataclass
class HotEntry:
    symbol: str
    added_ts: float
    last_seen_ts: float
    score: float
    reason: str
    gain_pct: float = float("nan")
    pm_gain: float  = float("nan")
    ah_gain: float  = float("nan")
    rvol: float     = float("nan")
    source: str     = ""


def _hot_score(gain: float, pm: float, ah: float, rvol: float, vacc: float, near_high: float, fallback: bool) -> float:
    s = 0.0
    g = safe_float(gain, 0)
    if 5 <= g <= 25:
        s += 25
    elif g > 25:
        s += 18
    elif g >= 3:
        s += 10
    if safe_float(pm, 0) >= 5:
        s += 18
    if safe_float(ah, 0) >= 5:
        s += 18
    s += min(safe_float(rvol, 0) * 8, 20)
    s += min(max(safe_float(vacc, 0) - 1, 0) * 10, 15)
    nh = safe_float(near_high, 99)
    if nh <= 3:
        s += 12
    elif nh <= 5:
        s += 7
    if fallback:
        s += 5
    return round(s, 1)


def hot_pool_add(
    symbol: str, gain: float, pm: float, ah: float,
    rvol: float, vacc: float, near_high: float,
    source: str, fallback: bool,
) -> None:
    pool: dict[str, HotEntry] = st.session_state.hot_pool
    qualifies = any([
        safe_float(gain, 0) >= 5,
        safe_float(pm, 0)   >= 5,
        safe_float(ah, 0)   >= 5,
        safe_float(rvol, 0) >= 1.3,
        safe_float(vacc, 0) >= 1.5,
        safe_float(near_high, 99) <= 5,
        fallback,
    ])
    if not qualifies:
        return
    score = _hot_score(gain, pm, ah, rvol, vacc, near_high, fallback)
    parts = []
    if safe_float(gain, 0) >= 5:
        parts.append(f"+{gain:.1f}%")
    if safe_float(pm, 0) >= 5:
        parts.append(f"PM+{pm:.1f}%")
    if safe_float(ah, 0) >= 5:
        parts.append(f"AH+{ah:.1f}%")
    if safe_float(rvol, 0) >= 1.3:
        parts.append(f"RVOL{rvol:.1f}x")
    reason = " ".join(parts) or ("Fallback" if fallback else "")
    ts = now_ts()
    if symbol in pool:
        old = pool[symbol]
        pool[symbol] = HotEntry(
            symbol=symbol, added_ts=old.added_ts, last_seen_ts=ts,
            score=max(score, old.score), reason=reason or old.reason,
            gain_pct=gain, pm_gain=pm, ah_gain=ah, rvol=rvol, source=source,
        )
    else:
        pool[symbol] = HotEntry(
            symbol=symbol, added_ts=ts, last_seen_ts=ts,
            score=score, reason=reason,
            gain_pct=gain, pm_gain=pm, ah_gain=ah, rvol=rvol, source=source,
        )


def hot_pool_expire() -> None:
    pool: dict[str, HotEntry] = st.session_state.hot_pool
    ts = now_ts()
    dead = [sym for sym, e in pool.items() if ts - e.last_seen_ts > HOT_POOL_TTL]
    for sym in dead:
        del pool[sym]
    if len(pool) > HOT_POOL_MAX:
        keep = sorted(pool, key=lambda s: pool[s].score, reverse=True)[:HOT_POOL_MAX]
        for sym in list(pool):
            if sym not in keep:
                del pool[sym]


def is_hot(symbol: str) -> bool:
    return symbol in st.session_state.get("hot_pool", {})


# ══════════════════════════════════════════════════════════════
#  MARKET DATA
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=15, show_spinner=False)
def quote_batch(symbols: tuple[str, ...], timeout: int) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    try:
        data = request_json(YAHOO_QUOTE_URL, params={"symbols": ",".join(symbols)}, timeout=timeout)
        quotes = data.get("quoteResponse", {}).get("result", [])
        return {q["symbol"]: q for q in quotes if q.get("symbol")}
    except Exception:
        return {}


@st.cache_data(ttl=15, show_spinner=False)
def history_1d(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol, period="1d", interval="5m",
            progress=False, auto_adjust=False, prepost=True, threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def history_daily(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol, period="30d", interval="1d",
            progress=False, auto_adjust=False, prepost=False, threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def calc_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, float("nan"))
    total = safe_float(vol.sum(), 0)
    return safe_float((typical * vol).sum() / total) if total > 0 else float("nan")


def calc_vol_accel(df: pd.DataFrame) -> float:
    if len(df) < 8:
        return float("nan")
    recent = safe_float(df["Volume"].tail(4).mean(), float("nan"))
    prior  = safe_float(df["Volume"].iloc[:-4].tail(20).mean(), float("nan"))
    if math.isnan(prior) or prior <= 0:
        return float("nan")
    return safe_float(recent / prior)


def calc_rvol(df: pd.DataFrame, daily: pd.DataFrame) -> float:
    if df.empty or daily.empty:
        return float("nan")
    cur_vol = safe_float(df["Volume"].sum(), float("nan"))
    avg_d   = safe_float(daily["Volume"].tail(20).mean(), float("nan"))
    if math.isnan(cur_vol) or math.isnan(avg_d) or avg_d <= 0:
        return float("nan")
    elapsed  = max(len(df), 1)
    expected = avg_d * (elapsed / 78)  # 78 five-min bars per full session
    return safe_float(cur_vol / expected) if expected > 0 else float("nan")


def bar_age(df: pd.DataFrame) -> float:
    if df.empty:
        return float("inf")
    idx = df.index[-1]
    if getattr(idx, "tzinfo", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return max(0.0, now_ts() - idx.timestamp())


def last2_candle_move(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return float("nan")
    anchor  = safe_float(df["Close"].iloc[-3])
    current = safe_float(df["Close"].iloc[-1])
    if math.isnan(anchor) or anchor <= 0:
        return float("nan")
    return (current - anchor) / anchor * 100


def detect_higher_lows(df: pd.DataFrame, n: int = 12) -> bool:
    if len(df) < n:
        return False
    lows = df["Low"].tail(n).values
    return bool(lows[-1] > lows[:-1].min())


def detect_tight_consolidation(df: pd.DataFrame, n: int = 8) -> bool:
    if len(df) < n:
        return False
    sub   = df.tail(n)
    price = safe_float(sub["Close"].iloc[-1])
    if price <= 0:
        return False
    rng = (safe_float(sub["High"].max()) - safe_float(sub["Low"].min())) / price * 100
    return rng <= 8


def detect_vwap_reclaim(df: pd.DataFrame, vwap_val: float) -> bool:
    if df.empty or math.isnan(vwap_val) or len(df) < 3:
        return False
    closes = df["Close"].tail(3).values
    return bool(closes[-2] < vwap_val <= closes[-1])


def lower_wick_pct_low(df: pd.DataFrame, n: int = 10) -> float:
    if len(df) < 2:
        return float("nan")
    sub = df.tail(min(n, len(df)))
    return safe_float(float(np.percentile(sub["Low"].values, 20)))


def _get_prev_close(quote: dict[str, Any], daily: pd.DataFrame) -> tuple[float, str]:
    v = safe_float(quote.get("regularMarketPreviousClose"))
    if not math.isnan(v) and v > 0:
        return v, "quote"
    if len(daily) > 1:
        v = safe_float(daily["Close"].iloc[-2])
        if not math.isnan(v) and v > 0:
            return v, "daily"
    # Estimate: price / (1 + chg%)
    price = safe_float(quote.get("regularMarketPrice"))
    chg   = safe_float(quote.get("regularMarketChangePercent"))
    if not math.isnan(price) and not math.isnan(chg) and price > 0:
        est = price / (1 + chg / 100)
        if est > 0:
            return round(est, 4), "estimated"
    return float("nan"), "missing"


def _session_type(quote: dict[str, Any]) -> str:
    state = str(quote.get("marketState", "REGULAR")).upper()
    if state in ("PRE", "PREMARKET"):
        return "PREMARKET"
    if state in ("POST", "POSTMARKET"):
        return "AFTER HOURS"
    return "REGULAR"


# ══════════════════════════════════════════════════════════════
#  LEVELS ENGINE
# ══════════════════════════════════════════════════════════════

def compute_breakout(day_high: float, recent_high: float, price: float) -> float:
    buf  = breakout_buffer(price)
    base = max(day_high, recent_high)
    return round(base + buf, 4)


def compute_dip_zone(vwap_v: float, recent_low: float, wick_low: float, current: float) -> tuple[float, float]:
    candidates = [v for v in (vwap_v, recent_low, wick_low) if not math.isnan(v) and 0 < v < current]
    dip_low  = max(candidates) if candidates else current * 0.97
    dip_high = dip_low * 1.015
    return round(dip_low, 4), round(dip_high, 4)


def compute_stop(dip_low: float, vwap_v: float, wick_low: float, base_low: float, current: float) -> float:
    candidates = [
        v for v in (dip_low * 0.985, vwap_v * 0.99, wick_low * 0.99, base_low * 0.99)
        if not math.isnan(v) and 0 < v < current
    ]
    return round(max(candidates), 4) if candidates else round(current * 0.97, 4)


def compute_targets(current: float, stop: float) -> tuple[float, float, float]:
    risk = max(current - stop, 0.001)
    t1   = round(current + risk * 2, 4)
    t2   = round(current + risk * 3, 4)
    t3   = round(current + risk * 4.5, 4)
    # Snap T1 toward nearest psych level if within 8%
    psych = nearest_psych_above(current)
    if abs(t1 - psych) / max(t1, 0.01) < 0.08:
        t1 = psych
    return t1, t2, t3


# ══════════════════════════════════════════════════════════════
#  SIGNAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════

@dataclass
class ScanConfig:
    max_tickers:      int   = CLOUD_MAX_TICKERS
    workers:          int   = CLOUD_WORKERS
    scan_timeout:     int   = CLOUD_SCAN_TIMEOUT
    scan_interval:    int   = CLOUD_SCAN_INTERVAL
    min_price:        float = 0.20
    max_price:        float = 50.0
    min_volume:       int   = 75_000
    request_timeout:  int   = 6
    max_cards:        int   = 5


def _classify(
    *,
    gain: float, pm_gain: float, ah_gain: float, gap: float,
    rvol: float, vacc: float, l2: float, near_high: float,
    ext_vwap: float, ext_base: float,
    above_vwap: bool, vwap_reclaim: bool, higher_lows: bool, tight_consol: bool,
    inside_dip: bool, near_trigger: bool, active_vol: bool, rr_ok: bool,
    not_stale: bool, open_price: float, vwap_val: float, current: float,
    hot: bool,
) -> str:
    nan = math.isnan

    # After-hours runner
    if not nan(ah_gain) and ah_gain >= 10:
        return "AFTER-HOURS RUNNER"

    # Premarket runner
    if not nan(pm_gain) and pm_gain >= 10:
        return "PREMARKET RUNNER"

    # Explosive (already a big move)
    explosive = (
        gain >= 25
        or (not nan(gap)     and gap >= 20)
        or (not nan(ah_gain) and ah_gain >= 20)
        or (not nan(pm_gain) and pm_gain >= 20)
        or (not nan(l2)      and l2 >= 8)
        or vacc >= 3
    )
    if explosive:
        if ext_vwap > 15 and near_high > 5 and not inside_dip:
            return "HOT RUNNER WAIT"
        return "EXPLOSIVE RUNNER"

    # Wait for reversal (below VWAP and below open)
    if not nan(vwap_val) and current < vwap_val and current < open_price * 0.995:
        return "WAIT FOR REVERSAL"

    # Entry ready — tightest, cleanest condition
    if (
        (near_trigger or inside_dip)
        and active_vol
        and rr_ok
        and not_stale
        and (above_vwap or vwap_reclaim)
        and rvol >= 1.3
        and ext_vwap <= 12
    ):
        return "ENTRY READY"

    # Early fire — first tradable ignition
    if (
        5 <= gain <= 25
        and rvol >= 1.3
        and (vacc >= 1.5 or near_high <= 5)
        and (above_vwap or vwap_reclaim)
        and ext_vwap <= 12
        and active_vol
    ):
        return "EARLY FIRE"

    # Dip buy zone
    if inside_dip and (hot or gain >= 5) and active_vol and not nan(vwap_val) and current >= vwap_val * 0.99:
        return "DIP BUY ZONE"

    # NHOD momentum
    if near_high <= 1.0 and gain >= 5 and active_vol and ext_vwap <= 15:
        return "NHOD MOMENTUM"

    # Building momentum
    if gain >= 3 and near_high <= 6 and (higher_lows or tight_consol or vwap_reclaim) and rvol >= 1.1:
        return "BUILDING MOMENTUM"

    # Breakout watch
    if near_high <= 3 and (tight_consol or higher_lows) and active_vol:
        return "BREAKOUT WATCH"
    if gain >= 10 and near_high <= 5 and active_vol:
        return "BREAKOUT WATCH"

    # Scalp only
    if (hot or gain >= 3) and active_vol and near_high <= 10:
        return "SCALP ONLY"

    # Already moved too far
    if gain >= 15 and ext_vwap > 12 and not inside_dip:
        return "WAIT FOR PULLBACK"

    return "TOO LATE / AVOID"


def _early_score(gain: float, rvol: float, vacc: float, near_high: float,
                 vwap_reclaim: bool, above_vwap: bool, higher_lows: bool, tight_consol: bool) -> float:
    s = 0.0
    g = safe_float(gain, 0)
    if 5 <= g <= 15:
        s += 30
    elif 15 < g <= 25:
        s += 18
    elif 3 <= g < 5:
        s += 10
    s += min(safe_float(rvol, 0) * 10, 20)
    s += min(max(safe_float(vacc, 0) - 1, 0) * 12, 15)
    nh = safe_float(near_high, 99)
    if nh <= 1:
        s += 15
    elif nh <= 3:
        s += 10
    elif nh <= 5:
        s += 5
    if vwap_reclaim:
        s += 12
    elif above_vwap:
        s += 6
    if higher_lows:
        s += 6
    if tight_consol:
        s += 6
    return round(min(s, 100), 1)


def _explosion_score(gain: float, gap: float, pm: float, ah: float,
                     l2: float, vacc: float, near_high: float) -> float:
    s = 0.0
    s += min(safe_float(gain, 0) * 1.2, 30)
    s += min(safe_float(gap, 0)  * 0.8, 20)
    s += min(safe_float(pm, 0)   * 0.6, 15)
    s += min(safe_float(ah, 0)   * 0.6, 15)
    s += min(safe_float(l2, 0)   * 2.0, 15)
    s += min(safe_float(vacc, 0) * 3.0, 12)
    if safe_float(near_high, 99) <= 3:
        s += 8
    return round(min(s, 100), 1)


def _setup_section(setup: str) -> str:
    return {
        "EARLY FIRE":         "early_fire",
        "NHOD MOMENTUM":      "early_fire",
        "ENTRY READY":        "entry_ready",
        "DIP BUY ZONE":       "dip_zone",
        "BREAKOUT WATCH":     "breakout_watch",
        "BUILDING MOMENTUM":  "breakout_watch",
        "EXPLOSIVE RUNNER":   "explosive",
        "HOT RUNNER WAIT":    "explosive",
        "AFTER-HOURS RUNNER": "ah_pm",
        "PREMARKET RUNNER":   "ah_pm",
        "SCALP ONLY":         "hot_pool_section",
        "WAIT FOR PULLBACK":  "hot_pool_section",
        "WAIT FOR REVERSAL":  "hot_pool_section",
        "TOO LATE / AVOID":   "hot_pool_section",
        "INVALID DATA":       "hot_pool_section",
    }.get(setup, "hot_pool_section")


def _invalid_row(symbol: str, reason: str = "", quote: dict[str, Any] | None = None) -> dict[str, Any]:
    q = quote or {}
    return {
        "ticker":           symbol,
        "setup":            "INVALID DATA",
        "section":          "hot_pool_section",
        "valid_trade":      False,
        "early_score":      0.0,
        "explosion_score":  0.0,
        "score":            0.0,
        "current":          safe_float(q.get("regularMarketPrice")),
        "previous_close":   float("nan"),
        "open":             float("nan"),
        "day_high":         float("nan"),
        "day_low":          float("nan"),
        "gap_pct":          float("nan"),
        "gain_pct":         float("nan"),
        "pm_gain":          float("nan"),
        "ah_gain":          float("nan"),
        "near_high_pct":    float("nan"),
        "ext_base":         float("nan"),
        "ext_vwap":         float("nan"),
        "vwap":             float("nan"),
        "above_vwap":       False,
        "vwap_reclaim":     False,
        "rvol":             float("nan"),
        "vacc":             float("nan"),
        "l2_move":          float("nan"),
        "higher_lows":      False,
        "tight_consol":     False,
        "volume":           0,
        "breakout_level":   float("nan"),
        "dip_low":          float("nan"),
        "dip_high":         float("nan"),
        "stop":             float("nan"),
        "target_1":         float("nan"),
        "target_2":         float("nan"),
        "target_3":         float("nan"),
        "rr":               float("nan"),
        "stale_secs":       float("inf"),
        "mismatch_pct":     float("nan"),
        "data_quality":     "invalid",
        "session_type":     "UNKNOWN",
        "hot":              False,
        "source":           "N/A",
        "last_updated":     utc_clock(now_ts()),
        "reason":           reason,
    }


def analyze_ticker(symbol: str, quote: dict[str, Any], config: ScanConfig) -> dict[str, Any]:
    """Full per-ticker analysis: fetch → indicators → classify → levels → score."""
    # ── Intraday data ─────────────────────────────────────────
    df    = history_1d(symbol)
    daily = history_daily(symbol)

    if df.empty:
        return _invalid_row(symbol, "No intraday data", quote)

    current = safe_float(df["Close"].iloc[-1])
    if math.isnan(current) or current <= 0:
        return _invalid_row(symbol, "No valid close", quote)

    # Price filter
    if current < config.min_price or current > config.max_price:
        return _invalid_row(symbol, f"Price filter ${current:.2f}", quote)

    # Previous close
    prev_close, prev_src = _get_prev_close(quote, daily)
    if math.isnan(prev_close) or prev_close <= 0:
        return _invalid_row(symbol, "Cannot determine prev close", quote)

    # Mismatch guard
    q_price = safe_float(quote.get("regularMarketPrice"))
    mismatch = float("nan")
    if not math.isnan(q_price) and q_price > 0:
        mismatch = abs(current - q_price) / q_price * 100
        if mismatch > 10:
            return _invalid_row(symbol, f"Price mismatch {mismatch:.1f}%", quote)

    # ── OHLCV ────────────────────────────────────────────────
    open_price = safe_float(df["Open"].iloc[0])
    day_high   = safe_float(df["High"].max())
    day_low    = safe_float(df["Low"].min())
    day_vol    = safe_int(df["Volume"].sum(), 0)
    stale      = bar_age(df)

    # ── Derived price metrics ────────────────────────────────
    gain_pct   = (current - prev_close) / prev_close * 100
    gap_pct    = (open_price - prev_close) / prev_close * 100 if not math.isnan(open_price) else float("nan")
    near_high  = (day_high - current) / day_high * 100 if day_high > 0 else float("nan")
    recent_high= safe_float(df["High"].tail(min(12, len(df))).max(), day_high)
    recent_low = safe_float(df["Low"].tail(min(12, len(df))).min(), day_low)
    base_low   = safe_float(df["Low"].tail(min(40, len(df))).min(), day_low)
    wick_low   = lower_wick_pct_low(df)
    ext_base   = (current - base_low) / base_low * 100 if base_low > 0 else float("nan")

    vwap_v     = calc_vwap(df)
    vacc       = calc_vol_accel(df)
    rvol       = calc_rvol(df, daily)
    l2         = last2_candle_move(df)
    hlows      = detect_higher_lows(df)
    tconsol    = detect_tight_consolidation(df)
    vwap_rcl   = detect_vwap_reclaim(df, vwap_v)
    above_vwap = (not math.isnan(vwap_v)) and current >= vwap_v

    ext_vwap = float("nan")
    if not math.isnan(vwap_v) and vwap_v > 0:
        ext_vwap = (current - vwap_v) / vwap_v * 100

    # ── Pre/after-market gains ───────────────────────────────
    pre_price  = safe_float(quote.get("preMarketPrice"))
    post_price = safe_float(quote.get("postMarketPrice"))
    pre_chg    = safe_float(quote.get("preMarketChangePercent"))
    post_chg   = safe_float(quote.get("postMarketChangePercent"))

    pm_gain = float("nan")
    ah_gain = float("nan")
    if not math.isnan(pre_price) and pre_price > 0 and prev_close > 0:
        pm_gain = (pre_price - prev_close) / prev_close * 100
    elif not math.isnan(pre_chg):
        pm_gain = pre_chg
    if not math.isnan(post_price) and post_price > 0 and current > 0:
        ah_gain = (post_price - current) / current * 100
    elif not math.isnan(post_chg):
        ah_gain = post_chg

    # ── Flags ────────────────────────────────────────────────
    active_vol   = day_vol >= config.min_volume and (safe_float(rvol, 0) >= 1.2 or safe_float(vacc, 0) >= 1.3)
    near_trigger = safe_float(near_high, 99) <= 3.0
    breakout_lvl = compute_breakout(day_high, recent_high, current)
    dip_low, dip_high = compute_dip_zone(vwap_v, recent_low, wick_low, current)
    stop         = compute_stop(dip_low, vwap_v, wick_low, base_low, current)
    t1, t2, t3   = compute_targets(current, stop)
    risk         = max(current - stop, 0.001)
    rr           = (t1 - current) / risk if risk > 0 else float("nan")

    inside_dip   = dip_low <= current <= dip_high
    rr_ok        = safe_float(rr, 0) >= 2.0 and stop < current
    not_stale    = stale <= 25 * 60
    hot          = is_hot(symbol)

    # ── Classify ─────────────────────────────────────────────
    setup = _classify(
        gain=gain_pct, pm_gain=pm_gain, ah_gain=ah_gain, gap=gap_pct,
        rvol=safe_float(rvol, 0), vacc=safe_float(vacc, 0), l2=l2,
        near_high=safe_float(near_high, 99),
        ext_vwap=safe_float(ext_vwap, 0), ext_base=safe_float(ext_base, 0),
        above_vwap=above_vwap, vwap_reclaim=vwap_rcl,
        higher_lows=hlows, tight_consol=tconsol,
        inside_dip=inside_dip, near_trigger=near_trigger,
        active_vol=active_vol, rr_ok=rr_ok, not_stale=not_stale,
        open_price=safe_float(open_price, current), vwap_val=vwap_v,
        current=current, hot=hot,
    )

    early_sc   = _early_score(gain_pct, safe_float(rvol, 0), safe_float(vacc, 0),
                               safe_float(near_high, 99), vwap_rcl, above_vwap, hlows, tconsol)
    expl_sc    = _explosion_score(gain_pct, gap_pct, pm_gain, ah_gain, l2, safe_float(vacc, 0), safe_float(near_high, 99))
    score      = round(early_sc + expl_sc * 0.4, 1)

    dq = prev_src
    if stale > 25 * 60:
        dq = "stale"

    return {
        "ticker":          symbol,
        "setup":           setup,
        "section":         _setup_section(setup),
        "valid_trade":     setup == "ENTRY READY",
        "early_score":     early_sc,
        "explosion_score": expl_sc,
        "score":           score,
        "current":         current,
        "previous_close":  prev_close,
        "open":            open_price,
        "day_high":        day_high,
        "day_low":         day_low,
        "gap_pct":         gap_pct,
        "gain_pct":        gain_pct,
        "pm_gain":         pm_gain,
        "ah_gain":         ah_gain,
        "near_high_pct":   safe_float(near_high),
        "ext_base":        ext_base,
        "ext_vwap":        ext_vwap,
        "vwap":            vwap_v,
        "above_vwap":      above_vwap,
        "vwap_reclaim":    vwap_rcl,
        "rvol":            safe_float(rvol),
        "vacc":            safe_float(vacc),
        "l2_move":         l2,
        "higher_lows":     hlows,
        "tight_consol":    tconsol,
        "volume":          day_vol,
        "breakout_level":  breakout_lvl,
        "dip_low":         dip_low,
        "dip_high":        dip_high,
        "stop":            stop,
        "target_1":        t1,
        "target_2":        t2,
        "target_3":        t3,
        "rr":              rr,
        "stale_secs":      stale,
        "mismatch_pct":    mismatch,
        "data_quality":    dq,
        "session_type":    _session_type(quote),
        "hot":             hot,
        "source":          quote.get("_source", "Yahoo"),
        "last_updated":    utc_clock(now_ts()),
        "reason":          "",
    }


# ══════════════════════════════════════════════════════════════
#  SCAN ORCHESTRATION
# ══════════════════════════════════════════════════════════════

@dataclass
class ScanStats:
    scanned:      int = 0
    passed:       int = 0
    missing_data: int = 0
    stale_data:   int = 0
    price_filter: int = 0
    no_momentum:  int = 0
    invalid_data: int = 0


_SETUP_ORDER: dict[str, int] = {
    "ENTRY READY":        0,
    "EARLY FIRE":         1,
    "DIP BUY ZONE":       2,
    "BREAKOUT WATCH":     3,
    "NHOD MOMENTUM":      4,
    "EXPLOSIVE RUNNER":   5,
    "BUILDING MOMENTUM":  6,
    "AFTER-HOURS RUNNER": 7,
    "PREMARKET RUNNER":   8,
    "HOT RUNNER WAIT":    9,
    "SCALP ONLY":         10,
    "WAIT FOR PULLBACK":  11,
    "WAIT FOR REVERSAL":  12,
    "TOO LATE / AVOID":   13,
    "INVALID DATA":       14,
}


def scan_universe(tickers: list[str], config: ScanConfig, progress: Any = None) -> tuple[list[dict[str, Any]], ScanStats]:
    stats    = ScanStats()
    started  = time.monotonic()
    deadline = started + config.scan_timeout

    # Stage 1 — batch quote fetch
    quotes: dict[str, dict[str, Any]] = {}
    for i in range(0, len(tickers), 50):
        batch = tuple(tickers[i:i + 50])
        try:
            quotes.update(quote_batch(batch, config.request_timeout))
        except Exception:
            pass

    # Sort: prioritise highest movers so they're processed first within timeout
    def _priority(sym: str) -> float:
        q = quotes.get(sym, {})
        return max(
            safe_float(q.get("regularMarketChangePercent"), 0),
            safe_float(q.get("preMarketChangePercent"), 0),
            safe_float(q.get("postMarketChangePercent"), 0),
        )
    ordered = sorted(tickers, key=_priority, reverse=True)

    rows: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=config.workers)
    futures  = {executor.submit(analyze_ticker, sym, quotes.get(sym, {}), config): sym for sym in ordered}
    pending  = set(futures)

    try:
        while pending and time.monotonic() < deadline:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                if time.monotonic() > deadline:
                    break
                stats.scanned += 1
                try:
                    row = f.result(timeout=0)
                except Exception as exc:
                    row = _invalid_row(futures[f], str(exc)[:80])
                    stats.invalid_data += 1

                setup  = row.get("setup", "INVALID DATA")
                reason = row.get("reason", "")
                if setup == "INVALID DATA":
                    if "Price filter" in reason:
                        stats.price_filter += 1
                    elif "stale" in str(row.get("data_quality", "")):
                        stats.stale_data += 1
                    else:
                        stats.missing_data += 1
                elif setup in ("TOO LATE / AVOID", "SCALP ONLY"):
                    stats.no_momentum += 1
                else:
                    stats.passed += 1

                rows.append(row)
                if progress is not None:
                    progress.progress(min(stats.scanned / max(len(tickers), 1), 1.0))
    finally:
        for f in pending:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    # Update hot pool from scan results (main thread — safe)
    for row in rows:
        sym = row.get("ticker", "")
        if not sym:
            continue
        hot_pool_add(
            sym,
            safe_float(row.get("gain_pct")),
            safe_float(row.get("pm_gain")),
            safe_float(row.get("ah_gain")),
            safe_float(row.get("rvol")),
            safe_float(row.get("vacc")),
            safe_float(row.get("near_high_pct")),
            "scanner",
            sym in FALLBACK_RUNNERS,
        )

    # Sort
    rows.sort(key=lambda r: (_SETUP_ORDER.get(r.get("setup", "INVALID DATA"), 14), -r.get("score", 0)))
    return rows, stats


# ══════════════════════════════════════════════════════════════
#  PLAYBOOK TEXT
# ══════════════════════════════════════════════════════════════

def playbook_notes(row: dict[str, Any], lang: str) -> dict[str, str]:
    setup = row.get("setup", "")
    b     = fmt_money(row.get("breakout_level"))
    d1    = fmt_money(row.get("dip_low"))
    d2    = fmt_money(row.get("dip_high"))
    stop  = fmt_money(row.get("stop"))
    t1    = fmt_money(row.get("target_1"))
    t2    = fmt_money(row.get("target_2"))
    vwap  = fmt_money(row.get("vwap"))

    if is_arabic(lang):
        table: dict[str, dict[str, str]] = {
            "ENTRY READY": {
                "confirmation": f"اختراق {b} مع حجم قوي يفعّل الاستمرار.",
                "invalidation": f"يُلغى إذا كسر السعر {stop} أو فقد VWAP {vwap}.",
                "playbook":     f"الدخول قريب. وقف تحت {stop}، الهدف الأول {t1}، الثاني {t2}.",
                "avoid":        "تجنب إذا انخفض الحجم أو كُسر الدعم.",
            },
            "EARLY FIRE": {
                "confirmation": f"البقاء فوق VWAP {vwap} مع استمرار الحجم يؤكد الاشتعال.",
                "invalidation": f"يفشل إذا عاد السعر تحت {vwap} أو تراجع الحجم.",
                "playbook":     f"اشتعال مبكر. كسر {b} يفتح الطريق نحو {t1} ثم {t2}.",
                "avoid":        "تجنب المطاردة إذا ابتعد السهم كثيراً عن VWAP.",
            },
            "DIP BUY ZONE": {
                "confirmation": f"شمعة خضراء فوق {d1} مع حجم متزايد تؤكد الدخول على النزول.",
                "invalidation": f"يفشل إذا كسر السعر تحت {stop}.",
                "playbook":     f"منطقة الشراء بين {d1} و {d2}. وقف تحت {stop}، الهدف {t1}.",
                "avoid":        "تجنب إذا استمر السهم بالهبوط دون شمعة انعكاس.",
            },
            "BREAKOUT WATCH": {
                "confirmation": f"كسر {b} مع حجم أعلى من المتوسط يؤكد الاختراق.",
                "invalidation": f"رفضان متتاليان عند {b} يشيران لاختراق وهمي.",
                "playbook":     f"راقب الاختراق فوق {b}. عند التأكيد: وقف {stop}، هدف {t1}.",
                "avoid":        f"تجنب إذا تراجع الحجم بعد اختراق {b}.",
            },
            "BUILDING MOMENTUM": {
                "confirmation": f"قيعان متصاعدة مع بقاء السعر قرب القمة فوق {vwap}.",
                "invalidation": f"يُلغى إذا كسر السعر {stop}.",
                "playbook":     f"الزخم يتراكم. راقب الاختراق فوق {b} للدخول.",
                "avoid":        "تجنب الدخول إذا خسر VWAP أو تراجع الحجم.",
            },
            "NHOD MOMENTUM": {
                "confirmation": f"أعلى سعر جديد مع حجم قوي يؤكد الاستمرار.",
                "invalidation": f"يُلغى إذا عاد السعر تحت {vwap}.",
                "playbook":     f"أعلى سعر جديد اليوم. وقف تحت {stop}، هدف {t1}.",
                "avoid":        "تجنب إذا كان السهم ممتداً جداً من VWAP.",
            },
            "EXPLOSIVE RUNNER": {
                "confirmation": f"سهم منفجر. انتظر تصحيحاً نحو VWAP {vwap} أو مستوى {d1}–{d2}.",
                "invalidation": f"تجنب إذا اخترق تحت {stop}.",
                "playbook":     f"لا تطارد. انتظر رجوع السعر نحو VWAP {vwap} أو إعادة اختبار {d1}–{d2}.",
                "avoid":        "الدخول في قمة الحركة مخاطرته عالية. انتظر تصحيحاً.",
            },
            "HOT RUNNER WAIT": {
                "confirmation": f"انتظر رجوع السعر إلى {vwap} أو المنطقة {d1}–{d2}.",
                "invalidation": "السهم ممتد. الدخول الآن نسبة مخاطرة عالية.",
                "playbook":     f"سهم ساخن لكن ممتد. انتظر إعادة اختبار VWAP {vwap}.",
                "avoid":        "تجنب المطاردة في قمة الحركة.",
            },
            "PREMARKET RUNNER": {
                "confirmation": f"راقب افتتاحاً قوياً فوق {b} مع حجم كبير.",
                "invalidation": f"تجنب إذا فتح تحت {vwap} أو بدون حجم.",
                "playbook":     f"يتحرك قبل الافتتاح. راقب أول 5 دقائق. مستوى الاختراق {b}.",
                "avoid":        "تجنب الدخول قبل تأكيد افتتاح قوي.",
            },
            "AFTER-HOURS RUNNER": {
                "confirmation": f"راقب افتتاحاً قوياً في الصباح فوق {b}.",
                "invalidation": f"تجنب إذا فتح ضعيفاً أو بدون حجم.",
                "playbook":     f"يتحرك بعد الإغلاق. راقب الافتتاح. مستوى الاختراق {b}.",
                "avoid":        "كثير من الأسهم تعكس اتجاهها بعد فجوة الافتتاح.",
            },
            "SCALP ONLY": {
                "confirmation": f"مضاربة سريعة بين {d1} و {b} فقط.",
                "invalidation": f"وقف تحت {stop}.",
                "playbook":     f"مضاربة فقط. الهدف {t1}، وقف {stop}.",
                "avoid":        "تجنب التداول بكمية كبيرة.",
            },
            "WAIT FOR PULLBACK": {
                "confirmation": f"انتظر رجوع السعر إلى {vwap} أو {d1}–{d2}.",
                "invalidation": "السعر الحالي بعيد جداً عن أي دعم نظيف.",
                "playbook":     f"السهم ممتد. انتظر تصحيحاً أو إعادة اختبار VWAP {vwap}.",
                "avoid":        "لا تطارد الأسهم الممتدة.",
            },
            "WAIT FOR REVERSAL": {
                "confirmation": f"انتظر استعادة VWAP {vwap} أو مستوى الافتتاح.",
                "invalidation": "السعر تحت VWAP والافتتاح. لا يوجد تأكيد بعد.",
                "playbook":     f"ضعيف الآن. انتظر استعادة {vwap} مع حجم.",
                "avoid":        "تجنب الدخول في هبوط بدون انعكاس.",
            },
        }
        generic = {
            "confirmation": f"البقاء فوق {vwap} مع استمرار الحجم.",
            "invalidation": f"الإلغاء عند كسر {stop}.",
            "playbook":     f"راقب {b} للاختراق أو {d1}–{d2} للنزول.",
            "avoid":        "تجنب إذا تراجع الحجم أو انكسر الدعم.",
        }
    else:
        table = {
            "ENTRY READY": {
                "confirmation": f"Break above {b} with strong volume can trigger continuation.",
                "invalidation": f"Invalid if price loses {stop} or fails VWAP near {vwap}.",
                "playbook":     f"Entry near trigger. Stop under {stop}, T1 {t1}, T2 {t2}.",
                "avoid":        "Avoid if volume drops or support breaks.",
            },
            "EARLY FIRE": {
                "confirmation": f"Holding above VWAP {vwap} with sustained volume confirms ignition.",
                "invalidation": f"Fails if price drops back under {vwap} or volume dries up.",
                "playbook":     f"Early ignition. Break above {b} targets {t1} then {t2}.",
                "avoid":        "Don't chase if price gets too far from VWAP.",
            },
            "DIP BUY ZONE": {
                "confirmation": f"Green candle above {d1} with increasing volume confirms dip entry.",
                "invalidation": f"Fails if price closes under {stop}.",
                "playbook":     f"Dip zone {d1}–{d2}. Stop under {stop}, Target {t1}.",
                "avoid":        "Avoid if stock keeps sliding without a reversal candle.",
            },
            "BREAKOUT WATCH": {
                "confirmation": f"Break above {b} with above-average volume confirms breakout.",
                "invalidation": f"Two rejections at {b} signals fakeout risk.",
                "playbook":     f"Watching break above {b}. On confirm: stop {stop}, target {t1}.",
                "avoid":        f"Fakeout risk if volume fades at {b}.",
            },
            "BUILDING MOMENTUM": {
                "confirmation": f"Higher lows and price holding near highs above VWAP {vwap}.",
                "invalidation": f"Invalidated on break of {stop}.",
                "playbook":     f"Momentum building. Watch breakout above {b}.",
                "avoid":        "Avoid if it loses VWAP or volume fades.",
            },
            "NHOD MOMENTUM": {
                "confirmation": f"New high of day with strong volume confirms continuation.",
                "invalidation": f"Invalid if price falls back under {vwap}.",
                "playbook":     f"NHOD. Stop under {stop}, target {t1}.",
                "avoid":        "Avoid chasing if too extended from VWAP.",
            },
            "EXPLOSIVE RUNNER": {
                "confirmation": f"Hot runner. Wait for pullback to VWAP {vwap} or zone {d1}–{d2}.",
                "invalidation": f"Avoid if loses {stop}.",
                "playbook":     f"Already explosive. Don't chase. Wait for VWAP {vwap} or {d1}–{d2} reclaim.",
                "avoid":        "Don't enter at the top. Wait for pullback.",
            },
            "HOT RUNNER WAIT": {
                "confirmation": f"Wait for price to return to VWAP {vwap} or zone {d1}–{d2}.",
                "invalidation": "Already extended. Chasing here has very poor R/R.",
                "playbook":     f"Hot but extended. Wait for VWAP {vwap} retest.",
                "avoid":        "Don't chase at the top of the move.",
            },
            "PREMARKET RUNNER": {
                "confirmation": f"Watch for strong open above {b} with big volume.",
                "invalidation": f"Avoid if opens under {vwap} or with no volume.",
                "playbook":     f"Moving premarket. Watch first 5 minutes. Breakout level {b}.",
                "avoid":        "Don't enter before confirming a strong open.",
            },
            "AFTER-HOURS RUNNER": {
                "confirmation": f"Watch for strong morning open above {b}.",
                "invalidation": f"Avoid if opens weak or with no volume.",
                "playbook":     f"Moving after hours. Watch the open. Breakout level {b}.",
                "avoid":        "Many stocks reverse the gap at open. Wait for confirmation.",
            },
            "SCALP ONLY": {
                "confirmation": f"Scalp between {d1} and {b} only.",
                "invalidation": f"Stop under {stop}.",
                "playbook":     f"Scalp only. Target {t1}, stop {stop}.",
                "avoid":        "Don't size up. Resistance is nearby.",
            },
            "WAIT FOR PULLBACK": {
                "confirmation": f"Wait for pullback to VWAP {vwap} or zone {d1}–{d2}.",
                "invalidation": "Current price is too far from any clean support.",
                "playbook":     f"Already extended. Wait for VWAP {vwap} retest or prior breakout reclaim.",
                "avoid":        "Don't chase extended names.",
            },
            "WAIT FOR REVERSAL": {
                "confirmation": f"Wait for VWAP {vwap} or open reclaim.",
                "invalidation": "Below VWAP and open. No confirmation yet.",
                "playbook":     f"Weak now. Wait for {vwap} reclaim with volume.",
                "avoid":        "Don't enter into a downtrend without reversal confirmation.",
            },
        }
        generic = {
            "confirmation": f"Hold above VWAP {vwap} with sustained volume.",
            "invalidation": f"Invalidated on break of {stop}.",
            "playbook":     f"Watch {b} for breakout or {d1}–{d2} for dip entry.",
            "avoid":        "Avoid if volume fades or support breaks.",
        }

    return table.get(setup, generic)


# ══════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════

def apply_theme(lang: str) -> None:
    d = "rtl" if is_arabic(lang) else "ltr"
    a = "right" if is_arabic(lang) else "left"
    st.markdown(f"""
<style>
:root {{color-scheme:dark}}
.stApp {{
    background:radial-gradient(circle at top left,rgba(19,92,92,.20),transparent 28rem),
               linear-gradient(135deg,#070f0e 0%,#0b1018 52%,#0d1210 100%);
    color:#f2f7f4;direction:{d};
}}
.block-container{{max-width:1300px;padding-top:1.1rem;padding-bottom:2rem}}
h1,h2,h3,p,label,div[data-testid="stMarkdownContainer"]{{text-align:{a}}}
div[data-testid="stMetric"]{{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);border-radius:10px;padding:.75rem .95rem}}
div[data-testid="stMetricLabel"] p{{color:#90aba0}}
div[data-testid="stMetricValue"]{{color:#f4fff9}}
section[data-testid="stSidebar"]{{background:#070e0d}}
.stButton button{{border-radius:8px;border:1px solid rgba(64,224,171,.32);background:#0e302a;color:#edfff6;min-height:2.3rem}}
div[data-testid="stDataFrame"]{{border:1px solid rgba(255,255,255,.07);border-radius:8px;overflow:hidden}}
/* live metric grid */
.lm-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;direction:{d};margin:.6rem 0 .85rem}}
.lm-card{{position:relative;min-height:96px;border-radius:13px;border:1px solid rgba(255,255,255,.09);
          background:linear-gradient(180deg,rgba(255,255,255,.065),rgba(255,255,255,.028)),rgba(7,14,16,.93);
          box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 12px 26px rgba(0,0,0,.20);
          padding:.72rem .86rem;overflow:hidden;text-align:{a};transition:border-color .22s,box-shadow .22s}}
.lm-card.up{{border-color:rgba(45,212,128,.68);animation:pulseG 800ms ease-out 1,glowG 1050ms ease-out 1}}
.lm-card.down{{border-color:rgba(248,113,113,.68);animation:pulseR 800ms ease-out 1,glowR 1050ms ease-out 1}}
.lm-label{{color:#90aba0;font-size:.72rem;margin-bottom:.36rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.lm-value{{color:#f5fff9;font-size:clamp(1.18rem,2vw,1.82rem);font-weight:800;line-height:1.05;font-variant-numeric:tabular-nums;transition:color .20s,transform .20s}}
.lm-card.up .lm-value{{color:#86efac;transform:translateY(-1px)}}
.lm-card.down .lm-value{{color:#fca5a5;transform:translateY(1px)}}
.lm-delta{{display:inline-flex;align-items:center;gap:.2rem;margin-top:.44rem;border-radius:999px;padding:.14rem .42rem;
           font-size:.70rem;font-weight:700;font-variant-numeric:tabular-nums;
           border:1px solid rgba(255,255,255,.08);color:#7d9a90;background:rgba(255,255,255,.038)}}
.lm-delta.up{{color:#86efac;background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.24)}}
.lm-delta.down{{color:#fca5a5;background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.24)}}
@keyframes pulseG{{0%{{background-color:rgba(34,197,94,.20)}}100%{{background-color:transparent}}}}
@keyframes pulseR{{0%{{background-color:rgba(239,68,68,.20)}}100%{{background-color:transparent}}}}
@keyframes glowG{{35%{{box-shadow:0 0 22px rgba(45,212,128,.15),0 14px 28px rgba(0,0,0,.24)}}}}
@keyframes glowR{{35%{{box-shadow:0 0 22px rgba(248,113,113,.17),0 14px 28px rgba(0,0,0,.24)}}}}
/* badges */
.badge{{display:inline-block;padding:.16rem .48rem;border-radius:6px;font-size:.70rem;font-weight:700;letter-spacing:.02em}}
.bg{{background:rgba(34,197,94,.16);color:#86efac;border:1px solid rgba(34,197,94,.28)}}
.bb{{background:rgba(59,130,246,.16);color:#93c5fd;border:1px solid rgba(59,130,246,.28)}}
.bo{{background:rgba(249,115,22,.16);color:#fdba74;border:1px solid rgba(249,115,22,.28)}}
.br{{background:rgba(239,68,68,.16);color:#fca5a5;border:1px solid rgba(239,68,68,.28)}}
.by{{background:rgba(234,179,8,.16);color:#fde047;border:1px solid rgba(234,179,8,.28)}}
.bgr{{background:rgba(107,114,128,.16);color:#9ca3af;border:1px solid rgba(107,114,128,.28)}}
@media(max-width:980px){{.lm-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:560px){{.lm-grid{{grid-template-columns:1fr}}.lm-card{{min-height:84px}}}}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LIVE METRICS
# ══════════════════════════════════════════════════════════════

def _init_lm() -> None:
    if "lm_prev" not in st.session_state:
        st.session_state.lm_prev = {}
    st.session_state.lm_pending = {}


def _commit_lm() -> None:
    st.session_state.lm_prev = {
        **st.session_state.get("lm_prev", {}),
        **st.session_state.get("lm_pending", {}),
    }


def _lm_direction(key: str, value: Any) -> tuple[str, float]:
    cur = safe_float(value, float("nan"))
    st.session_state.lm_pending[key] = cur
    prev = safe_float(st.session_state.get("lm_prev", {}).get(key), float("nan"))
    if math.isnan(cur) or math.isnan(prev):
        return "flat", 0.0
    tol   = max(abs(prev), 1.0) * 0.00001
    delta = cur - prev
    if delta > tol:
        return "up", delta
    if delta < -tol:
        return "down", delta
    return "flat", 0.0


def render_lm_grid(metrics: list[dict[str, Any]], lang: str) -> None:
    flat_txt = "بدون تغيير" if is_arabic(lang) else "No change"
    cards = ""
    for m in metrics:
        key       = str(m["key"])
        value     = m["value"]
        label     = str(m.get("label", ""))
        prefix    = str(m.get("prefix", ""))
        suffix    = str(m.get("suffix", ""))
        direction, delta = _lm_direction(key, value)
        arrow = {"up": "▲", "down": "▼", "flat": ""}[direction]
        dt    = flat_txt if direction == "flat" else f"{arrow} {abs(delta):.3g}"

        n = safe_float(value, float("nan"))
        if math.isnan(n):
            vs = "N/A"
        elif isinstance(value, str):
            vs = value
        elif abs(n) >= 1000:
            vs = f"{prefix}{n:,.0f}{suffix}"
        elif abs(n) < 1:
            vs = f"{prefix}{n:.4f}{suffix}"
        else:
            vs = f"{prefix}{n:,.2f}{suffix}"

        cards += (
            f'<div class="lm-card {direction}">'
            f'<div class="lm-label">{html.escape(label)}</div>'
            f'<div class="lm-value">{html.escape(vs)}</div>'
            f'<div class="lm-delta {direction}">{html.escape(dt)}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="lm-grid">{cards}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  BADGE
# ══════════════════════════════════════════════════════════════

_BADGE_CLS: dict[str, str] = {
    "ENTRY READY":        "bg",
    "EARLY FIRE":         "bg",
    "DIP BUY ZONE":       "bb",
    "NHOD MOMENTUM":      "bb",
    "BUILDING MOMENTUM":  "bb",
    "BREAKOUT WATCH":     "by",
    "EXPLOSIVE RUNNER":   "bo",
    "HOT RUNNER WAIT":    "bo",
    "PREMARKET RUNNER":   "by",
    "AFTER-HOURS RUNNER": "by",
    "SCALP ONLY":         "bgr",
    "WAIT FOR PULLBACK":  "bgr",
    "WAIT FOR REVERSAL":  "bgr",
    "TOO LATE / AVOID":   "br",
    "INVALID DATA":       "br",
}


def _badge_html(setup: str, lang: str) -> str:
    cls   = _BADGE_CLS.get(setup, "bgr")
    label = setup_label(setup, lang)
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


# ══════════════════════════════════════════════════════════════
#  TRADE CARD
# ══════════════════════════════════════════════════════════════

def render_trade_card(row: dict[str, Any], lang: str) -> None:
    notes  = playbook_notes(row, lang)
    setup  = row.get("setup", "INVALID DATA")
    ticker = row.get("ticker", "?")

    with st.container(border=True):
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            st.subheader(ticker)
            st.markdown(_badge_html(setup, lang), unsafe_allow_html=True)
        with col_b:
            st.caption(f"{row.get('session_type','?')} · {tr('data_q', lang)}: {row.get('data_quality','?')}")
            if row.get("hot"):
                st.caption("🔥 Hot Pool")
            dq = row.get("data_quality", "")
            if dq in ("estimated", "stale", "invalid"):
                st.warning(f"⚠️ Data quality: {dq}", icon=None)

        t = ticker
        render_lm_grid([
            {"label": tr("current", lang),    "value": row.get("current"),        "key": f"{t}:px",   "prefix": "$"},
            {"label": tr("gain", lang),        "value": row.get("gain_pct"),       "key": f"{t}:g",    "suffix": "%"},
            {"label": tr("rvol", lang),        "value": row.get("rvol"),           "key": f"{t}:rv",   "suffix": "x"},
            {"label": tr("vol_acc", lang),     "value": row.get("vacc"),           "key": f"{t}:va",   "suffix": "x"},
            {"label": tr("break_level", lang), "value": row.get("breakout_level"), "key": f"{t}:brk",  "prefix": "$"},
            {"label": tr("stop", lang),        "value": row.get("stop"),           "key": f"{t}:stp",  "prefix": "$"},
            {"label": tr("t1", lang),          "value": row.get("target_1"),       "key": f"{t}:t1",   "prefix": "$"},
            {"label": tr("t2", lang),          "value": row.get("target_2"),       "key": f"{t}:t2",   "prefix": "$"},
        ], lang)

        dip_str = f"{fmt_money(row.get('dip_low'))} – {fmt_money(row.get('dip_high'))}"
        st.markdown(
            f"**{tr('dip_zone_label', lang)}:** {dip_str} &nbsp;·&nbsp; "
            f"**R/R:** {fmt_rr(row.get('rr'))} &nbsp;·&nbsp; "
            f"**VWAP:** {fmt_money(row.get('vwap'))}",
        )

        st.markdown(f"**{tr('confirmation', lang)}:** {notes.get('confirmation', '')}")
        st.markdown(f"**{tr('invalidation', lang)}:** {notes.get('invalidation', '')}")
        st.markdown(f"**{tr('playbook', lang)}:** {notes.get('playbook', '')}")
        st.caption(f"⚠️ {tr('avoid', lang)}: {notes.get('avoid', '')}")

        with st.expander(tr("details", lang)):
            detail = {
                "Open":     fmt_money(row.get("open")),
                "High":     fmt_money(row.get("day_high")),
                "Low":      fmt_money(row.get("day_low")),
                "Gap":      fmt_pct(row.get("gap_pct")),
                "PM Gain":  fmt_pct(row.get("pm_gain")),
                "AH Gain":  fmt_pct(row.get("ah_gain")),
                "Ext VWAP": fmt_pct(row.get("ext_vwap")),
                "Ext Base": fmt_pct(row.get("ext_base")),
                "Near Hi":  fmt_pct(row.get("near_high_pct")),
                "L2 Move":  fmt_pct(row.get("l2_move")),
                "Volume":   f"{safe_int(row.get('volume')):,}",
                "Age min":  f"{safe_float(row.get('stale_secs', 0)) / 60:.1f}",
                "Early Sc": str(row.get("early_score", "N/A")),
                "Expl Sc":  str(row.get("explosion_score", "N/A")),
                "H.Lows":   "✓" if row.get("higher_lows") else "✗",
                "Consol":   "✓" if row.get("tight_consol") else "✗",
                "VWAP Rcl": "✓" if row.get("vwap_reclaim") else "✗",
                "Source":   str(row.get("source", "N/A")),
            }
            st.dataframe(pd.DataFrame([detail]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  COMPACT TABLE
# ══════════════════════════════════════════════════════════════

def compact_table(rows: list[dict[str, Any]], lang: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        out.append({
            tr("ticker", lang):      r.get("ticker", ""),
            tr("setup", lang):       setup_label(r.get("setup", ""), lang),
            tr("current", lang):     fmt_money(r.get("current")),
            tr("gain", lang):        fmt_pct(r.get("gain_pct")),
            tr("rvol", lang):        fmt_x(r.get("rvol")),
            tr("vol_acc", lang):     fmt_x(r.get("vacc")),
            tr("break_level", lang): fmt_money(r.get("breakout_level")),
            tr("stop", lang):        fmt_money(r.get("stop")),
            tr("t1", lang):          fmt_money(r.get("target_1")),
            tr("rr", lang):          fmt_rr(r.get("rr")),
            tr("session", lang):     r.get("session_type", "?"),
        })
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════
#  SECTIONS
# ══════════════════════════════════════════════════════════════

SECTION_DEFS: list[tuple[str, list[str]]] = [
    ("early_fire",        ["EARLY FIRE", "NHOD MOMENTUM"]),
    ("entry_ready",       ["ENTRY READY"]),
    ("dip_zone",          ["DIP BUY ZONE"]),
    ("breakout_watch",    ["BREAKOUT WATCH", "BUILDING MOMENTUM"]),
    ("explosive",         ["EXPLOSIVE RUNNER", "HOT RUNNER WAIT"]),
    ("ah_pm",             ["AFTER-HOURS RUNNER", "PREMARKET RUNNER"]),
    ("hot_pool_section",  ["SCALP ONLY", "WAIT FOR PULLBACK", "WAIT FOR REVERSAL"]),
]


def _filter_section(rows: list[dict[str, Any]], setups: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("setup") in setups]


def render_section(
    section_key: str,
    setups: list[str],
    rows: list[dict[str, Any]],
    lang: str,
    max_cards: int,
) -> None:
    st.header(tr(section_key, lang))
    subset = _filter_section(rows, set(setups))
    if not subset:
        st.info(tr("no_results", lang))
        return
    limited = subset[:max_cards]
    st.dataframe(compact_table(limited, lang), use_container_width=True, hide_index=True)
    for row in limited:
        render_trade_card(row, lang)


# ══════════════════════════════════════════════════════════════
#  STATE + ORCHESTRATION
# ══════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "scan_rows":           [],
        "last_scan_ts":        None,
        "source_counts":       {},
        "discovery_message":   "",
        "optional_watchlist":  "",
        "scan_stats":          ScanStats(),
        "hot_pool":            {},
        "lm_prev":             {},
        "lm_pending":          {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _should_scan(interval: int) -> bool:
    last = st.session_state.get("last_scan_ts")
    return (not last) or (now_ts() - float(last) >= interval)


def _run_scan(config: ScanConfig, lang: str) -> None:
    # Pre-seed hot pool from fallback list
    for sym in FALLBACK_RUNNERS:
        hot_pool_add(sym, float("nan"), float("nan"), float("nan"),
                     float("nan"), float("nan"), float("nan"), "fallback", True)
    hot_pool_expire()

    watchlist  = st.session_state.get("optional_watchlist", "")
    discovery  = discover_tickers(watchlist, config.max_tickers, config.request_timeout)

    label    = "جار الفحص..." if is_arabic(lang) else "Scanning..."
    progress = st.progress(0, text=label)
    rows, stats = scan_universe(discovery.tickers, config, progress=progress)
    progress.empty()

    st.session_state.scan_rows           = rows
    st.session_state.last_scan_ts        = now_ts()
    st.session_state.source_counts       = discovery.sources
    st.session_state.discovery_message   = discovery.message
    st.session_state.scan_stats          = stats


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Momentum Playbook Scanner",
        page_icon="🔥",
        layout="wide",
    )
    _init_state()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        lang = st.radio(
            "Language / اللغة", ["English", "Arabic"],
            horizontal=True, key="language",
        )
        apply_theme(lang)
        st.caption(APP_VERSION)
        st.divider()

        st.subheader(tr("scanner_controls", lang))
        cloud_fast = st.toggle(tr("cloud_fast", lang), value=True)
        config = ScanConfig() if cloud_fast else ScanConfig(
            max_tickers=200, workers=10, scan_timeout=28,
        )

        with st.expander(tr("advanced_watchlist", lang)):
            wl = st.text_area(
                tr("advanced_watchlist", lang),
                value=st.session_state.optional_watchlist,
                help=tr("advanced_help", lang),
                height=76,
                label_visibility="collapsed",
            )
            st.session_state.optional_watchlist = wl

        scan_clicked = st.button(tr("refresh", lang), use_container_width=True)
        st.caption(f"{tr('next_refresh', lang)}: {config.scan_interval}s")

        # Hot pool mini-widget
        pool: dict[str, HotEntry] = st.session_state.get("hot_pool", {})
        if pool:
            st.divider()
            st.caption(f"🟡 {tr('hot_pool_lbl', lang)}: {len(pool)}")
            top = sorted(pool.values(), key=lambda e: e.score, reverse=True)[:10]
            for e in top:
                st.caption(f"**{e.symbol}** {fmt_pct(e.gain_pct)} · {e.reason[:22]}")

    # ── Page header ───────────────────────────────────────────
    apply_theme(lang)
    _init_lm()

    st.title(tr("page_title", lang))
    st.caption(tr("subtitle", lang))

    # ── Trigger scan ──────────────────────────────────────────
    if scan_clicked or _should_scan(config.scan_interval):
        _run_scan(config, lang)

    rows: list[dict[str, Any]] = st.session_state.scan_rows
    stats: ScanStats            = st.session_state.scan_stats
    pool_size = len(st.session_state.get("hot_pool", {}))
    valid_cnt = sum(1 for r in rows if r.get("valid_trade"))

    # ── Top KPI row ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr("last_scan", lang), utc_clock(st.session_state.last_scan_ts))
    m2.metric(tr("universe", lang),  f"{len(rows):,}")
    m3.metric(tr("valid", lang),     f"{valid_cnt:,}")
    m4.metric(tr("hot_pool", lang),  f"{pool_size:,}")

    # ── Discovery sources ─────────────────────────────────────
    with st.expander(tr("sources", lang)):
        st.write(st.session_state.discovery_message)
        src_df = pd.DataFrame(
            [{"Source": k, "Tickers": v} for k, v in st.session_state.source_counts.items()]
        )
        if not src_df.empty:
            st.dataframe(src_df, use_container_width=True, hide_index=True)

    # ── Debug diagnostics ─────────────────────────────────────
    with st.expander(tr("debug", lang)):
        s = stats
        dbg = {
            "Scanned":       s.scanned,
            "Passed":        s.passed,
            "Missing data":  s.missing_data,
            "Stale":         s.stale_data,
            "Price filtered":s.price_filter,
            "No momentum":   s.no_momentum,
            "Invalid":       s.invalid_data,
            "Hot pool":      pool_size,
        }
        st.dataframe(pd.DataFrame([dbg]), use_container_width=True, hide_index=True)

    # ── Sections ──────────────────────────────────────────────
    if not rows:
        st.warning(tr("no_results", lang))
    else:
        for section_key, setups in SECTION_DEFS:
            render_section(section_key, setups, rows, lang, config.max_cards)

        with st.expander(tr("table", lang)):
            st.dataframe(compact_table(rows[:60], lang), use_container_width=True, hide_index=True)

    _commit_lm()
    st.caption(tr("footer", lang))

    # ── Auto-rerun ────────────────────────────────────────────
    last = st.session_state.get("last_scan_ts")
    if last and now_ts() - float(last) >= config.scan_interval:
        st.rerun()


if __name__ == "__main__":
    main()
