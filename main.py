# main.py
from __future__ import annotations
import os
import math
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ==============================
# Env / Config
# ==============================
PORT = int(os.getenv("PORT", "10000"))

USE_OPLAB = os.getenv("USE_OPLAB", "1") == "1"
OPLAB_BASE_URL = os.getenv("OPLAB_BASE_URL", "https://api.oplab.com.br")
OPLAB_CHAIN_PATH = os.getenv("OPLAB_CHAIN_PATH", "/v3/options/chain")
# Ex.: "symbol={ticker}&maturity={vencimento}"
OPLAB_QUERY_TEMPLATE = os.getenv("OPLAB_QUERY_TEMPLATE", "symbol={ticker}&maturity={vencimento}")
OPLAB_VENC_FMT = os.getenv("OPLAB_VENC_FMT", "YYYY-MM-DD")  # ou "YYYYMMDD"
OPLAB_AUTH_HEADER = os.getenv("OPLAB_AUTH_HEADER", "Access-Token")
OPLAB_BEARER = os.getenv("OPLAB_BEARER", "0") == "1"
OPLAB_API_KEY = os.getenv("OPLAB_API_KEY", "")

# Candles (técnicos)
# Ex.: /v3/stocks/candles ? symbol={ticker}&timeframe=1d&limit=260
OPLAB_CANDLES_PATH = os.getenv("OPLAB_CANDLES_PATH", "/v3/stocks/candles")
OPLAB_CANDLES_QUERY_TEMPLATE = os.getenv("OPLAB_CANDLES_QUERY_TEMPLATE", "symbol={ticker}&timeframe=1d&limit=260")

# Lista default de tickers para screener/MEI
OPLAB_TICKERS_DEFAULT = os.getenv(
    "OPLAB_TICKERS_DEFAULT",
    "PETR4,VALE3,ITUB4,PRIO3,BBDC4,ABEV3,BBAS3,B3SA3,WEGE3,SUZB3,ELET3,ELET6,GGBR4,USIM5,CSNA3,LREN3,AZUL4,GOLL4,MGLU3,JBSS3"
)

# Parâmetros de modelo (mercado BR)
TRADING_DAYS = 252
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.12"))   # 12% a.a. (aprox CDI)
DIVIDEND_YIELD = float(os.getenv("DIVIDEND_YIELD", "0.00"))   # 0% se desconhecido

# ==============================
# App
# ==============================
app = FastAPI(title="robo-venda-coberta", version="2.0.0")

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

# ==============================
# Datas (3ª sexta)
# ==============================
def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_friday_offset = (4 - d.weekday()) % 7  # 0=Mon ... 4=Fri
    first_friday = d + timedelta(days=first_friday_offset)
    return first_friday + timedelta(days=14)

def adjust_for_holiday(dt: date, holidays: Optional[List[date]] = None) -> date:
    if holidays is None:
        holidays = []
    d = dt
    if d.weekday() >= 5 or d in holidays:
        while d.weekday() >= 5 or d in holidays:
            d = d - timedelta(days=1)
    return d

def parse_vencimento_auto(today: date = date.today()) -> date:
    y, m = today.year, today.month
    tf = third_friday(y, m)
    if today > tf:
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        tf = third_friday(y, m)
    return adjust_for_holiday(tf)

def dt_to_fmt(d: date, fmt: str) -> str:
    if fmt == "YYYYMMDD":
        return d.strftime("%Y%m%d")
    return d.strftime("%Y-%m-%d")

def yearfrac(d1: date, d2: date) -> float:
    return max((d2 - d1).days, 1) / TRADING_DAYS

# ==============================
# Black-Scholes / IV / Gregas
# ==============================
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1d2(S, K, r, q, T, sigma) -> Tuple[float, float]:
    sqrtT = math.sqrt(max(T, 1e-9))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def bs_call_price(S, K, r, q, T, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    return S * math.exp(-q*T) * _norm_cdf(d1) - K * math.exp(-r*T) * _norm_cdf(d2)

def bs_put_price(S, K, r, q, T, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    return K * math.exp(-r*T) * _norm_cdf(-d2) - S * math.exp(-q*T) * _norm_cdf(-d1)

def bs_call_greeks(S, K, r, q, T, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    sqrtT = math.sqrt(T)
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    pdf = _norm_pdf(d1)
    delta = math.exp(-q*T) * _norm_cdf(d1)
    gamma = (math.exp(-q*T) * pdf) / (S * sigma * sqrtT)
    theta = (-(S * math.exp(-q*T) * pdf * sigma) / (2 * sqrtT)
             - r * K * math.exp(-r*T) * _norm_cdf(d2)
             + q * S * math.exp(-q*T) * _norm_cdf(d1))
    vega = S * math.exp(-q*T) * pdf * sqrtT
    rho = K * T * math.exp(-r*T) * _norm_cdf(d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

def bs_put_greeks(S, K, r, q, T, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    sqrtT = math.sqrt(T)
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    pdf = _norm_pdf(d1)
    delta = -math.exp(-q*T) * _norm_cdf(-d1)
    gamma = (math.exp(-q*T) * pdf) / (S * sigma * sqrtT)
    theta = (-(S * math.exp(-q*T) * pdf * sigma) / (2 * sqrtT)
             + r * K * math.exp(-r*T) * _norm_cdf(-d2)
             - q * S * math.exp(-q*T) * _norm_cdf(-d1))
    vega = S * math.exp(-q*T) * pdf * sqrtT
    rho = -K * T * math.exp(-r*T) * _norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

def implied_vol_newton(target_price, S, K, r, q, T, is_call=True, initial_vol=0.30, tol=1e-6, max_iter=100):
    sigma = max(initial_vol, 1e-4)
    for _ in range(max_iter):
        price = bs_call_price(S, K, r, q, T, sigma) if is_call else bs_put_price(S, K, r, q, T, sigma)
        diff = price - target_price
        if abs(diff) < tol:
            return max(sigma, 0.0001)
        eps = 1e-5
        p_up = bs_call_price(S, K, r, q, T, sigma + eps) if is_call else bs_put_price(S, K, r, q, T, sigma + eps)
        vega = (p_up - price) / eps
        if vega == 0 or math.isnan(vega):
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5:
            return None
    return sigma

# ==============================
# OpLab Helpers
# ==============================
def oplab_headers() -> Dict[str, str]:
    if not OPLAB_API_KEY:
        raise HTTPException(status_code=500, detail="OPLAB_API_KEY ausente no Environment.")
    return {OPLAB_AUTH_HEADER: (f"Bearer {OPLAB_API_KEY}" if OPLAB_BEARER else OPLAB_API_KEY)}

def build_oplab_url(ticker: str, venc: date) -> str:
    venc_str = dt_to_fmt(venc, OPLAB_VENC_FMT)
    q = OPLAB_QUERY_TEMPLATE.format(ticker=ticker.upper(), vencimento=venc_str)
    return f"{OPLAB_BASE_URL.rstrip('/')}{OPLAB_CHAIN_PATH}?{q}"

def fetch_oplab_chain(ticker: str, venc: date, retries: int = 1) -> Dict[str, Any]:
    url = build_oplab_url(ticker, venc)
    last_err = None
    for _ in range(max(retries, 1)):
        try:
            r = requests.get(url, headers=oplab_headers(), timeout=30)
            if r.status_code == 200:
                return r.json()
            last_err = f"{r.status_code}: {r.text[:400]}"
        except requests.RequestException as e:
            last_err = str(e)
    raise HTTPException(status_code=502, detail=f"Erro OpLab: {last_err}")

def build_candles_url(ticker: str) -> str:
    q = OPLAB_CANDLES_QUERY_TEMPLATE.format(ticker=ticker.upper())
    return f"{OPLAB_BASE_URL.rstrip('/')}{OPLAB_CANDLES_PATH}?{q}"

def fetch_candles(ticker: str) -> pd.DataFrame:
    url = build_candles_url(ticker)
    try:
        r = requests.get(url, headers=oplab_headers(), timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erro rede candles OpLab: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Erro candles OpLab {r.status_code}: {r.text[:200]}")
    try:
        j = r.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Resposta candles não-JSON.")
    # Tentativa de normalização de estrutura
    data = j.get("data") if isinstance(j, dict) else j
    if not data:
        raise HTTPException(status_code=404, detail="Candles vazios.")
    df = pd.DataFrame(data)
    # Esperado: time/open/high/low/close/volume (ajuste nomes se necessário)
    # Tenta alternativas de chaves comuns
    for cpair in [("time","time"), ("t","time")]:
        if cpair[0] in df.columns:
            df.rename(columns={cpair[0]: cpair[1]}, inplace=True)
            break
    for cpair in [("o","open"),("h","high"),("l","low"),("c","close"),("v","volume")]:
        if cpair[0] in df.columns:
            df.rename(columns={cpair[0]: cpair[1]}, inplace=True)
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass
    cols_needed = {"open","high","low","close"}
    if not cols_needed.issubset(set(df.columns)):
        raise HTTPException(status_code=500, detail="Formato de candles inesperado.")
    return df

def extract_underlying_and_series(oplab_json: Dict[str, Any]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    spot = None
    for key in ("price","last","underlying_price","previous_close"):
        if isinstance(oplab_json.get(key), (int,float)):
            spot = float(oplab_json[key]); break
    meta = oplab_json.get("meta") or {}
    for key in ("price","last","underlying_price","previous_close"):
        if spot is None and isinstance(meta.get(key),(int,float)):
            spot = float(meta[key]); break
    series = oplab_json.get("data") or []
    return spot, series

def choose_due_date(series: List[Dict[str, Any]], target: date) -> Optional[Dict[str, Any]]:
    if not series: return None
    items = []
    for s in series:
        dd = s.get("due_date")
        try:
            d = datetime.fromisoformat(dd).date()
        except Exception:
            continue
        items.append((d,s))
    if not items: return None
    for d,s in items:
        if d == target: return s
    items.sort(key=lambda x: x[0])
    for d,s in items:
        if d >= target: return s
    return items[-1][1]

# ==============================
# Pydantic models (I/O)
# ==============================
class GreeksFilter(BaseModel):
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None
    vega_max: Optional[float] = None
    theta_min: Optional[float] = None

class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str  # "YYYY-MM-DD" ou "auto"
    criterio: str = ">=PM"
    min_liquidez: int = 10
    min_bid: float = 0.01
    max_spread: float = 0.50
    rel_floor_ratio: float = 0.05
    greeks: Optional[GreeksFilter] = None

class ScreenerIn(BaseModel):
    tickers: Optional[List[str]] = None
    vencimento: str = "auto"
    min_liquidez: int = 50
    min_bid: float = 0.05
    max_spread: float = 0.50
    rel_floor_ratio: float = 0.10
    top_n: int = 20

class SignalIn(BaseModel):
    ticker: str
    vencimento: str = "auto"
    preferencia: Optional[str] = Field(None, description="CALL/PUT se quiser forçar")
    min_liquidez: int = 50
    min_bid: float = 0.05
    max_spread: float = 0.50
    rel_floor_ratio: float = 0.10
    delta_call_range: Tuple[float,float] = (0.30, 0.45)
    delta_put_abs_range: Tuple[float,float] = (0.30, 0.45)  # usa |delta|
    news_bias: Optional[str] = Field(None, description="bullish/bearish/neutral (manual)")

# ==============================
# Helpers de opções comuns
# ==============================
def premium_ref(bid: float, ask: float, close: float) -> float:
    if bid>0 and ask>0 and ask>=bid: return 0.5*(bid+ask)
    if close>0: return close
    if bid>0: return bid
    return 0.0

def dataframe_calls_from_series(series: Dict[str,Any]) -> pd.DataFrame:
    rows, max_finvol = [], 0.0
    for r in series.get("strikes", []):
        K = float(r.get("strike") or 0)
        call = r.get("call") or {}
        if not K or not call: continue
        bid = float(call.get("bid") or 0)
        ask = float(call.get("ask") or 0)
        close = float(call.get("close") or 0)
        vol = int(call.get("volume") or 0)
        finvol = float(call.get("financial_volume") or 0)
        sym = str(call.get("symbol") or "")
        rows.append({"symbol":sym,"strike":K,"bid":bid,"ask":ask,"close":close,"volume":vol,"financial_volume":finvol})
        max_finvol = max(max_finvol, finvol)
    df = pd.DataFrame(rows)
    return df, max_finvol

def dataframe_puts_from_series(series: Dict[str,Any]) -> pd.DataFrame:
    rows, max_finvol = [], 0.0
    for r in series.get("strikes", []):
        K = float(r.get("strike") or 0)
        put = r.get("put") or {}
        if not K or not put: continue
        bid = float(put.get("bid") or 0)
        ask = float(put.get("ask") or 0)
        close = float(put.get("close") or 0)
        vol = int(put.get("volume") or 0)
        finvol = float(put.get("financial_volume") or 0)
        sym = str(put.get("symbol") or "")
        rows.append({"symbol":sym,"strike":K,"bid":bid,"ask":ask,"close":close,"volume":vol,"financial_volume":finvol})
        max_finvol = max(max_finvol, finvol)
    df = pd.DataFrame(rows)
    return df, max_finvol

def apply_liquidity_filters(df: pd.DataFrame, min_liq:int, min_bid:float, max_spread:float, rel_floor_ratio:float, max_finvol:float) -> pd.DataFrame:
    if df.empty: return df
    df = df[df["volume"] >= min_liq]
    df = df[df["bid"] >= min_bid]
    df["spread"] = (df["ask"] - df["bid"]).fillna(0.0)
    df = df[(df["spread"] <= max_spread) | (df["ask"] <= 0)]
    if max_finvol > 0 and rel_floor_ratio > 0:
        df = df[(df["financial_volume"] >= max_finvol*rel_floor_ratio) | (df["financial_volume"] == 0)]
    return df

# ==============================
# /covered-call/suggest (preservado)
# ==============================
@app.post("/covered-call/suggest")
def covered_call_suggest(data: SuggestIn):
    if not USE_OPLAB:
        raise HTTPException(status_code=400, detail="Configure USE_OPLAB=1 e variáveis da OpLab.")

    ticker = data.ticker_subjacente.upper()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)

    venc_target = parse_vencimento_auto() if data.vencimento.lower()=="auto" else datetime.strptime(data.vencimento,"%Y-%m-%d").date()
    chain_json = fetch_oplab_chain(ticker, venc_target, retries=2)
    spot, series_all = extract_underlying_and_series(chain_json)
    if spot is None: spot = pm
    series = choose_due_date(series_all, venc_target)
    if not series: raise HTTPException(status_code=404, detail="Vencimento indisponível.")
    dd = series.get("due_date"); venc_real = datetime.fromisoformat(dd).date()
    T = yearfrac(date.today(), venc_real)

    df, max_finvol = dataframe_calls_from_series(series)
    if df.empty: raise HTTPException(status_code=404, detail="Sem calls para vencimento.")

    # Liquidez
    df = apply_liquidity_filters(df, data.min_liquidez, data.min_bid, data.max_spread, data.rel_floor_ratio, max_finvol)
    if df.empty: raise HTTPException(status_code=404, detail="Nenhuma call passou nos filtros de liquidez.")

    # IV + gregas (CALL)
    r, q = RISK_FREE_RATE, DIVIDEND_YIELD
    rows = []
    for _, rr in df.iterrows():
        K = float(rr["strike"]); bid=float(rr["bid"]); ask=float(rr["ask"]); close=float(rr["close"])
        pref = premium_ref(bid,ask,close)
        if pref<=0: continue
        iv = implied_vol_newton(pref, spot, K, r, q, T, is_call=True, initial_vol=0.30)
        g = bs_call_greeks(spot,K,r,q,T,iv) if iv else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
        rows.append({**rr, "premium_ref":pref, "iv":iv, **g})
    df = pd.DataFrame(rows)
    if df.empty: raise HTTPException(status_code=404, detail="Nenhuma opção com prêmio válido.")

    # Critério strike vs PM
    if data.criterio.upper()==">=PM":
        df = df[df["strike"]>=pm]
    elif data.criterio.upper()=="PM":
        df["diff_pm"]=(df["strike"]-pm).abs()
        df=df.sort_values("diff_pm")

    if data.greeks:
        if data.greeks.delta_min is not None:
            df = df[(df["delta"].isna()) | (df["delta"]>=data.greeks.delta_min)]
        if data.greeks.delta_max is not None:
            df = df[(df["delta"].isna()) | (df["delta"]<=data.greeks.delta_max)]
        if data.greeks.vega_max is not None:
            df = df[(df["vega"].isna()) | (df["vega"]<=data.greeks.vega_max)]
        if data.greeks.theta_min is not None:
            df = df[(df["theta"].isna()) | (df["theta"]>=data.greeks.theta_min)]

    if df.empty: raise HTTPException(status_code=404, detail="Nenhuma opção atende aos filtros.")

    days = max((venc_real - date.today()).days, 1)
    df["retorno_premio_pct"] = (df["premium_ref"]/pm)*100.0
    df["retorno_anualizado_pct"] = df["retorno_premio_pct"] * (TRADING_DAYS/days)
    df = df.sort_values(["retorno_anualizado_pct","premium_ref"], ascending=[False,False])
    best = df.iloc[0].to_dict()

    strike = float(best["strike"]); premio=float(best["premium_ref"])
    bid=float(best["bid"]); ask=float(best["ask"]); volume=int(best["volume"])
    symbol=str(best["symbol"]); contratos=qty//100

    def resultado(preco_venc):
        if preco_venc>=strike:
            venda=strike*qty; custo=pm*qty; premio_total=premio*qty
            return venda + premio_total - custo
        else:
            return premio*qty

    cenarios=[]
    for p in [pm-10, pm, strike-1, strike, strike+3]:
        cenarios.append({"preco_acao_no_vencimento":round(float(p),2),
                         "resultado_total":round(float(resultado(p)),2),
                         "observacao":"exercido" if p>=strike else "não exercido"})

    preview_cols=["symbol","strike","bid","ask","close","volume","financial_volume","iv","delta","theta","vega","retorno_premio_pct","retorno_anualizado_pct"]
    top5=df[preview_cols].head(5).reset_index(drop=True).to_dict(orient="records")

    return {
        "subjacente":{"ticker":ticker,"spot_usado":spot,"preco_medio":pm},
        "vencimento":{"solicitado":data.vencimento,"usado":venc_real.isoformat(),"dias_uteis":days},
        "recomendacao":{
            "symbol":symbol,"strike":strike,"premio_ref":premio,"bid":bid,"ask":ask,"volume":volume,
            "iv":best.get("iv"),"delta":best.get("delta"),"theta":best.get("theta"),"vega":best.get("vega"),
            "retorno_premio_pct":round(float(best["retorno_premio_pct"]),2),
            "retorno_anualizado_pct":round(float(best["retorno_anualizado_pct"]),2),
            "contratos_sugeridos":contratos
        },
        "cenarios":cenarios,
        "top5":top5,
        "fonte_dados":"oplab_v3",
        "observacoes":[
            "Gregas via Black-Scholes com IV implícita pelo mid/close.",
            "Liquidez filtrada por volume/bid/spread/volume financeiro relativo."
        ]
    }

# ==============================
# Técnicos (SMA/RSI/MACD/Pivôs/Fibo)
# ==============================
def sma(series: pd.Series, n:int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(series: pd.Series, n:int=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0); dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn.replace(0, 1e-12))
    return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal

def atr(df: pd.DataFrame, n:int=14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def classic_pivots(last_h:float,last_l:float,last_c:float) -> Dict[str,float]:
    P=(last_h+last_l+last_c)/3.0
    R1=2*P-last_l; S1=2*P-last_h
    R2=P+(last_h-last_l); S2=P-(last_h-last_l)
    return {"P":P,"R1":R1,"S1":S1,"R2":R2,"S2":S2}

def fibo_levels(swing_low:float, swing_high:float) -> Dict[str,float]:
    diff = swing_high - swing_low
    return {
        "0.236": swing_high - 0.236*diff,
        "0.382": swing_high - 0.382*diff,
        "0.500": swing_high - 0.500*diff,
        "0.618": swing_high - 0.618*diff,
        "0.786": swing_high - 0.786*diff,
    }

def ta_summary(df: pd.DataFrame) -> Dict[str,Any]:
    df = df.copy()
    df["sma9"]=sma(df["close"],9)
    df["sma21"]=sma(df["close"],21)
    df["sma50"]=sma(df["close"],50)
    df["sma200"]=sma(df["close"],200)
    df["rsi14"]=rsi(df["close"],14)
    macd_line, macd_sig = macd(df["close"],12,26,9)
    df["macd"]=macd_line; df["macd_signal"]=macd_sig; df["macd_hist"]=df["macd"]-df["macd_signal"]
    df["atr14"]=atr(df,14)

    last = df.dropna().iloc[-1]
    piv = classic_pivots(df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1])

    # swing dos últimos ~60 pregões
    sw = df.tail(60)
    swing_low = float(sw["low"].min()); swing_high=float(sw["high"].max())
    fib = fibo_levels(swing_low, swing_high)

    trend_up = (last["close"]>last["sma50"]>last["sma200"])
    trend_down = (last["close"]<last["sma50"]<last["sma200"])
    macd_pos = last["macd"]>0 and last["macd_hist"]>0
    macd_neg = last["macd"]<0 and last["macd_hist"]<0

    bias=None
    if trend_up and macd_pos and 45<=last["rsi14"]<=70:
        bias="bullish"
    elif trend_down and macd_neg and 30<=last["rsi14"]<=55:
        bias="bearish"
    else:
        # neutro se não bateu os dois conjuntos
        bias="neutral"

    return {
        "last_close": float(last["close"]),
        "sma9": float(last["sma9"]), "sma21": float(last["sma21"]),
        "sma50": float(last["sma50"]), "sma200": float(last["sma200"]),
        "rsi14": float(last["rsi14"]),
        "macd": float(last["macd"]), "macd_signal": float(last["macd_signal"]), "macd_hist": float(last["macd_hist"]),
        "atr14": float(last["atr14"]),
        "pivots": {k:float(v) for k,v in piv.items()},
        "fibo": fib,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "bias": bias
    }

# ==============================
# MEI: tickers base / screener / signal
# ==============================
@app.post("/mei/tickers")
def mei_tickers(body: Dict[str,Any] = None):
    base = [t.strip().upper() for t in OPLAB_TICKERS_DEFAULT.split(",") if t.strip()]
    return {"tickers_default": base}

@app.post("/mei/screener")
def mei_screener(inp: ScreenerIn):
    if not USE_OPLAB:
        raise HTTPException(status_code=400, detail="USE_OPLAB=1 requerido.")
    tickers = [t.strip().upper() for t in (inp.tickers or OPLAB_TICKERS_DEFAULT.split(",")) if t.strip()]
    venc_target = parse_vencimento_auto() if inp.vencimento.lower()=="auto" else datetime.strptime(inp.vencimento,"%Y-%m-%d").date()
    out_rows = []
    for tk in tickers:
        try:
            chain_json = fetch_oplab_chain(tk, venc_target, retries=2)
            spot, all_series = extract_underlying_and_series(chain_json)
            series = choose_due_date(all_series, venc_target)
            if not series: continue
            calls, max_finvol = dataframe_calls_from_series(series)
            if calls.empty: continue
            calls = apply_liquidity_filters(calls, inp.min_liquidez, inp.min_bid, inp.max_spread, inp.rel_floor_ratio, max_finvol)
            if calls.empty: continue
            # score simples: soma de volume + volume financeiro + nº de linhas válidas
            tot_vol = int(calls["volume"].sum())
            tot_fin = float(calls["financial_volume"].sum())
            nlines = int(calls.shape[0])
            out_rows.append({"ticker": tk, "tot_volume": tot_vol, "tot_finvol": tot_fin, "n_calls": nlines})
        except Exception:
            # ignora falha desse ticker
            continue
    if not out_rows:
        raise HTTPException(status_code=404, detail="Nenhum ticker com liquidez suficiente.")
    df = pd.DataFrame(out_rows)
    df["score"] = (df["tot_volume"].rank(pct=True) + df["tot_finvol"].rank(pct=True) + df["n_calls"].rank(pct=True))
    df = df.sort_values(["score","tot_finvol","tot_volume"], ascending=[False,False,False]).head(inp.top_n)
    return {"vencimento": inp.vencimento, "ranked": df.to_dict(orient="records")}

@app.post("/mei/signal")
def mei_signal(inp: SignalIn):
    if not USE_OPLAB:
        raise HTTPException(status_code=400, detail="USE_OPLAB=1 requerido.")
    tk = inp.ticker.upper()
    # Técnicos
    candles = fetch_candles(tk)
    ta = ta_summary(candles)
    bias = inp.news_bias or ta["bias"]  # permite ajustar manualmente por notícia

    # Vencimento e chain
    venc_target = parse_vencimento_auto() if inp.vencimento.lower()=="auto" else datetime.strptime(inp.vencimento,"%Y-%m-%d").date()
    chain_json = fetch_oplab_chain(tk, venc_target, retries=2)
    spot, all_series = extract_underlying_and_series(chain_json)
    if spot is None: spot = float(ta["last_close"])
    series = choose_due_date(all_series, venc_target)
    if not series: raise HTTPException(status_code=404, detail="Vencimento indisponível.")
    dd = series.get("due_date"); venc_real = datetime.fromisoformat(dd).date()
    T = yearfrac(date.today(), venc_real)

    # Calls/Puts filtradas
    calls, max_finvol_c = dataframe_calls_from_series(series)
    puts,  max_finvol_p = dataframe_puts_from_series(series)
    calls = apply_liquidity_filters(calls, inp.min_liquidez, inp.min_bid, inp.max_spread, inp.rel_floor_ratio, max_finvol_c)
    puts  = apply_liquidity_filters(puts,  inp.min_liquidez, inp.min_bid, inp.max_spread, inp.rel_floor_ratio, max_finvol_p)

    if calls.empty and puts.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez mínima em calls/puts.")

    # Funções auxiliares de score por delta alvo
    r, q = RISK_FREE_RATE, DIVIDEND_YIELD
    def enrich_with_iv_greeks(df: pd.DataFrame, is_call: bool) -> pd.DataFrame:
        rows=[]
        for _, rr in df.iterrows():
            K=float(rr["strike"]); bid=float(rr["bid"]); ask=float(rr["ask"]); close=float(rr["close"])
            pref = premium_ref(bid,ask,close)
            if pref<=0: continue
            iv = implied_vol_newton(pref, spot, K, r, q, T, is_call=is_call, initial_vol=0.30)
            g  = (bs_call_greeks if is_call else bs_put_greeks)(spot,K,r,q,T,iv) if iv else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
            rows.append({**rr, "premium_ref":pref, "iv":iv, **g})
        return pd.DataFrame(rows)

    calls = enrich_with_iv_greeks(calls, True)
    puts  = enrich_with_iv_greeks(puts,  False)

    # Heurística de lado (a seco)
    lado_forcado = (inp.preferencia or "").upper() if inp.preferencia else None
    lado = lado_forcado or ("CALL" if bias=="bullish" else "PUT" if bias=="bearish" else "CALL")

    # Escolha por delta-range (CALL: 0.30~0.45 / PUT: |delta| 0.30~0.45), maior retorno esperado
    def pick_by_delta(df: pd.DataFrame, rng: Tuple[float,float], call: bool) -> Optional[Dict[str,Any]]:
        if df.empty: return None
        dmin, dmax = rng
        if call:
            cand = df[df["delta"].between(dmin,dmax, inclusive="both")]
        else:
            cand = df[df["delta"].abs().between(dmin,dmax, inclusive="both")]
        if cand.empty:
            cand = df.copy()
        # score: premium_ref * volume / (1+spread) priorizando menor spread
        cand["spread"] = (cand["ask"] - cand["bid"]).clip(lower=0)
        cand["score"] = cand["premium_ref"] * (cand["volume"].clip(lower=1)) / (1.0 + cand["spread"])
        cand = cand.sort_values(["score","premium_ref","volume"], ascending=[False,False,False])
        return cand.iloc[0].to_dict()

    pick = None
    if lado=="CALL" and not calls.empty:
        pick = pick_by_delta(calls, inp.delta_call_range, True)
    elif lado=="PUT" and not puts.empty:
        pick = pick_by_delta(puts, inp.delta_put_abs_range, False)
    else:
        # fallback: escolhe o outro se o preferido não existir
        if lado=="CALL" and not puts.empty: 
            lado="PUT"; pick = pick_by_delta(puts, inp.delta_put_abs_range, False)
        elif lado=="PUT" and not calls.empty:
            lado="CALL"; pick = pick_by_delta(calls, inp.delta_call_range, True)

    if not pick:
        raise HTTPException(status_code=404, detail="Nenhuma opção elegível após filtros/greeks.")

    # IV “percentil no vencimento” (relative IV)
    df_ref = calls if lado=="CALL" else puts
    iv_valid = df_ref["iv"].dropna()
    iv_pct = None
    if not iv_valid.empty and pick.get("iv"):
        iv_pct = float((iv_valid < pick["iv"]).mean())  # 0..1

    # Justificativa técnica resumida
    tech_notes=[]
    if bias=="bullish": tech_notes.append("Tendência/viés altista por SMA(50)>SMA(200) e MACD positivo.")
    elif bias=="bearish": tech_notes.append("Tendência/viés baixista por SMA(50)<SMA(200) e MACD negativo.")
    else: tech_notes.append("Viés neutro; favorece CALL por padrão (pode ajustar por notícia).")

    last_close = ta["last_close"]
    piv = ta["pivots"]
    if lado=="CALL" and last_close>piv["P"]: tech_notes.append("Preço acima do pivô (suporte intradiário).")
    if lado=="PUT"  and last_close<piv["P"]: tech_notes.append("Preço abaixo do pivô (resistência intradiária).")

    return {
        "ticker": tk,
        "lado": lado,  # CALL ou PUT
        "vencimento": {"solicitado": inp.vencimento, "usado": venc_real.isoformat(), "T_anos": round(yearfrac(date.today(), venc_real),4)},
        "spot_usado": spot,
        "tecnicos": ta,
        "recomendacao": {
            "symbol": pick["symbol"],
            "strike": float(pick["strike"]),
            "premium_ref": float(pick["premium_ref"]),
            "bid": float(pick["bid"]),
            "ask": float(pick["ask"]),
            "volume": int(pick["volume"]),
            "iv": (None if pd.isna(pick.get("iv")) else float(pick["iv"])),
            "iv_percentil_no_venc": (None if iv_pct is None else round(iv_pct,3)),
            "delta": (None if pd.isna(pick.get("delta")) else float(pick["delta"])),
            "theta": (None if pd.isna(pick.get("theta")) else float(pick["theta"])),
            "vega":  (None if pd.isna(pick.get("vega")) else float(pick["vega"]))
        },
        "justificativa": tech_notes,
        "observacoes": [
            "Sinal combina liquidez, spread, delta alvo e viés técnico (SMA/RSI/MACD/pivôs/Fibo).",
            "IV rank relativo é calculado dentro do vencimento (percentil vs outras strikes).",
            "Ajuste 'news_bias' para bullish/bearish se houver fato relevante."
        ]
    }
# ====== News (env) ======
USE_NEWS = os.getenv("USE_NEWS", "0") == "1"
NEWS_BASE_URL = os.getenv("NEWS_BASE_URL", "")
NEWS_QUERY_TEMPLATE = os.getenv("NEWS_QUERY_TEMPLATE", "q={query}&language=pt&sortBy=publishedAt&pageSize=30")
NEWS_AUTH_HEADER = os.getenv("NEWS_AUTH_HEADER", "Authorization")
NEWS_BEARER = os.getenv("NEWS_BEARER", "0") == "1"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

def news_headers() -> Dict[str, str]:
    if not NEWS_API_KEY:
        return {}
    token = f"Bearer {NEWS_API_KEY}" if NEWS_BEARER else NEWS_API_KEY
    return {NEWS_AUTH_HEADER: token}

def build_news_url(query: str) -> str:
    q = NEWS_QUERY_TEMPLATE.format(query=query)
    return f"{NEWS_BASE_URL.rstrip('/')}?{q}"

BULL_WORDS = {"alta","sobe","subiu","otimista","positivo","forte","recorde","aprova","aprovação","ganho"}
BEAR_WORDS = {"queda","cai","caiu","pessimista","negativo","fraco","despenca","reprova","risco","perde"}

def simple_sentiment(text: str) -> int:
    t = (text or "").lower()
    score = 0
    score += sum(w in t for w in BULL_WORDS)
    score -= sum(w in t for w in BEAR_WORDS)
    return score

def fetch_news_bias_for_ticker(ticker: str, extra_terms: Optional[List[str]] = None, lookback_days: int = 3) -> Optional[str]:
    if not USE_NEWS or not NEWS_BASE_URL:
        return None
    q_terms = [ticker, "ações"]
    if extra_terms:
        q_terms += extra_terms
    query = " ".join(q_terms)
    url = build_news_url(query)
    try:
        r = requests.get(url, headers=news_headers(), timeout=15)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        j = r.json()
    except Exception:
        return None
    # NewsAPI: artigos em j["articles"]; tente também j["data"] se outro provedor
    arts = j.get("articles") or j.get("data") or []
    if not isinstance(arts, list):
        return None

    since = datetime.utcnow() - timedelta(days=lookback_days)
    score = 0
    n = 0
    for a in arts:
        title = a.get("title") or ""
        desc  = a.get("description") or a.get("summary") or ""
        published = a.get("publishedAt") or a.get("published_at") or ""
        try:
            when = datetime.fromisoformat(published.replace("Z",""))
        except Exception:
            when = None
        if when and when < since:
            continue
        score += simple_sentiment(title) + simple_sentiment(desc)
        n += 1

    if n == 0:
        return None
    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"
class NewsIn(BaseModel):
    ticker: str
    lookback_days: int = 3

@app.post("/mei/news-bias")
def mei_news_bias(inp: NewsIn):
    bias = fetch_news_bias_for_ticker(inp.ticker.upper(), lookback_days=inp.lookback_days)
    return {"ticker": inp.ticker.upper(), "bias": bias}
# ==============================
# Fim
# ==============================