import os
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator


# =============================================================================
# Config via ENV (você já setou no Render)
# =============================================================================
PORT = int(os.getenv("PORT", "10000"))

USE_OPLAB = os.getenv("USE_OPLAB", "1") == "1"
OPLAB_BASE_URL = os.getenv("OPLAB_BASE_URL", "https://api.oplab.com.br/v3").rstrip("/")
# Endpoints que usamos (não precisa filtrar por vencimento no servidor; filtramos client-side)
OPLAB_COVERED_PATH = os.getenv("OPLAB_CHAIN_PATH", "/market/options/strategies/covered")
OPLAB_SERIES_PATH = os.getenv("OPLAB_SERIES_PATH", "/market/instruments/series")

# Autenticação
OPLAB_AUTH_HEADER = os.getenv("OPLAB_AUTH_HEADER", "Access-Token")
OPLAB_BEARER = os.getenv("OPLAB_BEARER", "0") == "1"  # para APIs que exigem Bearer; a OpLab usa token direto
OPLAB_API_KEY = os.getenv("OPLAB_API_KEY")  # NÃO versione

# =============================================================================
# App
# =============================================================================
app = FastAPI(title="robo-venda-coberta", version="1.0.0")


# =============================================================================
# Utilitários de tempo/preço (pregão vs after-hours)
# =============================================================================
def _parse_ts_to_brt(ts_val: Any) -> Optional[datetime]:
    """
    Aceita:
      - ISO string: "2025-08-15T20:30:00.000Z"
      - epoch segundos: 1755294000
      - epoch milissegundos: 1755294000000
    Retorna datetime em BRT (UTC-3) ou None.
    """
    try:
        if ts_val is None:
            return None
        if isinstance(ts_val, (int, float)):
            # Heurística: > 1e12 => milissegundos
            if ts_val > 1e12:
                ts_val = ts_val / 1000.0
            dt = datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
        elif isinstance(ts_val, str):
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        else:
            return None
        return dt.astimezone(timezone(timedelta(hours=-3)))
    except Exception:
        return None


def is_after_hours(ts_val: Any) -> bool:
    """
    Considera after-hours se:
      - sem timestamp válido,
      - não é 'hoje' em BRT,
      - hora fora da janela 09:30–18:30 BRT (ampla).
    """
    ts_brt = _parse_ts_to_brt(ts_val)
    if ts_brt is None:
        return True
    now_brt = datetime.now(timezone(timedelta(hours=-3)))
    if ts_brt.date() != now_brt.date():
        return True
    market_open = now_brt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_brt.replace(hour=18, minute=30, second=0, microsecond=0)
    return not (market_open.time() <= now_brt.time() <= market_close.time())


def price_for_calc(
    call_obj: Dict[str, Any],
    min_bid: float = 0.05,
    max_spread: float = 0.50,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Retorna (preco_escolhido, meta) com fallback:
      - Pregão: mid se spread OK; senão bid>=min_bid; senão close*0.9; senão None
      - After-hours: bid>=min_bid; senão close*0.9; senão None
    """
    bid = float(call_obj.get("bid") or 0)
    ask = float(call_obj.get("ask") or 0)
    close = float(call_obj.get("close") or 0)
    vol = int(call_obj.get("volume") or 0)

    ts_iso_or_epoch = call_obj.get("time")
    after = is_after_hours(ts_iso_or_epoch)
    spread = (ask - bid) if (ask > 0 and bid > 0) else math.inf

    meta = {
        "bid": bid,
        "ask": ask,
        "close": close,
        "volume": vol,
        "after_hours": after,
        "spread": spread if math.isfinite(spread) else None,
        "ts": ts_iso_or_epoch,
        "preco_fonte": None,  # mid|bid|close*0.9
    }

    # Pregão
    if not after:
        if ask > 0 and bid > 0 and ask >= bid:
            if (ask - bid) <= max_spread:
                meta["preco_fonte"] = "mid"
                return (bid + ask) / 2.0, meta
            if bid >= min_bid:
                meta["preco_fonte"] = "bid"
                return bid, meta
            return None, meta
        if bid >= min_bid:
            meta["preco_fonte"] = "bid"
            return bid, meta
        if close > 0:
            meta["preco_fonte"] = "close*0.9"
            return close * 0.9, meta
        return None, meta

    # After-hours
    if bid >= min_bid:
        meta["preco_fonte"] = "bid"
        return bid, meta
    if close > 0:
        meta["preco_fonte"] = "close*0.9"
        return close * 0.9, meta
    return None, meta


# =============================================================================
# Filtros de liquidez e (opcional) gregas
# =============================================================================
def extract_greeks(call_obj: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    OpLab pode (ou não) expor gregas por contrato. Tentamos achar em:
      - call_obj["greeks"] = {"delta":..., "vega":..., ...}
      - ou campos planos: call_obj["delta"], etc.
    """
    g = call_obj.get("greeks") or {}
    out = {
        "delta": _safe_float(call_obj.get("delta", g.get("delta"))),
        "gamma": _safe_float(call_obj.get("gamma", g.get("gamma"))),
        "vega": _safe_float(call_obj.get("vega", g.get("vega"))),
        "theta": _safe_float(call_obj.get("theta", g.get("theta"))),
        "rho": _safe_float(call_obj.get("rho", g.get("rho"))),
    }
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def passes_liquidity_filters(
    call_obj: Dict[str, Any],
    min_liquidez: int,
    rel_floor_ratio: float,
    peers_volume_max: int,
    min_bid: float,
    max_spread: float,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Liquidez absoluta: volume >= min_liquidez e bid >= min_bid
    Liquidez relativa: volume >= rel_floor_ratio * (máximo volume daquele vencimento)
    Spread (somente em pregão): (ask - bid) <= max_spread (se houver ambos)
    """
    vol = int(call_obj.get("volume") or 0)
    bid = float(call_obj.get("bid") or 0)
    ask = float(call_obj.get("ask") or 0)
    ts = call_obj.get("time")
    after = is_after_hours(ts)

    spread_ok = True
    if not after and (bid > 0 and ask > 0 and ask >= bid):
        spread_ok = (ask - bid) <= max_spread

    rel_ok = True
    if peers_volume_max > 0:
        rel_ok = vol >= rel_floor_ratio * peers_volume_max

    passed = (
        vol >= min_liquidez
        and bid >= min_bid
        and rel_ok
        and spread_ok
    )
    debug = {
        "vol": vol,
        "bid": bid,
        "ask": ask,
        "after_hours": after,
        "rel_volume_floor": rel_floor_ratio * peers_volume_max if peers_volume_max > 0 else None,
        "peers_volume_max": peers_volume_max,
        "spread_ok": spread_ok,
    }
    return passed, debug


def passes_greeks_filters(
    call_obj: Dict[str, Any],
    min_delta: Optional[float],
    max_delta: Optional[float],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Filtro simples por delta (se valores e greeks existirem).
    """
    g = extract_greeks(call_obj)
    d = g.get("delta")
    if d is None:
        # Se não há grega, não barramos por grega
        return True, {"delta": None, "note": "sem grega no payload"}
    if min_delta is not None and d < min_delta:
        return False, {"delta": d}
    if max_delta is not None and d > max_delta:
        return False, {"delta": d}
    return True, {"delta": d}


# =============================================================================
# Entrada/saída do endpoint
# =============================================================================
class SuggestIn(BaseModel):
    ticker_subjacente: str = Field(..., examples=["PETR4", "VALE3", "PRIO3"])
    preco_medio: float = Field(..., gt=0)
    quantidade_acoes: int = Field(..., ge=100, description="Múltiplos de 100 são recomendados")
    vencimento: str = Field(..., description='"YYYY-MM-DD" ou "auto"')
    criterio: str = Field(">=PM", description='">=PM" (OTM) ou "PM" (strike mais próximo do PM)')
    min_liquidez: int = Field(50, ge=0, description="Volume mínimo de negócios")
    # filtros extra:
    min_bid: float = Field(0.05, ge=0)
    max_spread: float = Field(0.50, ge=0)
    # liquidez relativa (ex.: 0.1 => pelo menos 10% do maior volume daquele vencimento)
    rel_floor_ratio: float = Field(0.10, ge=0, le=1.0)
    # gregas (opcionais)
    min_delta: Optional[float] = Field(None)
    max_delta: Optional[float] = Field(None)

    @validator("criterio")
    def _crit(cls, v: str) -> str:
        v = (v or "").upper().strip()
        if v not in {">=PM", "PM"}:
            raise ValueError('criterio deve ser ">=PM" ou "PM"')
        return v

    @validator("vencimento")
    def _venc(cls, v: str) -> str:
        v = v.strip().lower()
        if v == "auto":
            return v
        try:
            # valida formato
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except Exception:
            raise ValueError('vencimento deve ser "auto" ou data "YYYY-MM-DD"')


class Scenario(BaseModel):
    preco_acao_no_vencimento: float
    resultado_total: float
    observacao: str


class SuggestOut(BaseModel):
    recomendacao: Dict[str, Any]
    cenarios: List[Scenario]
    fonte_dados: str
    debug: Dict[str, Any]


# =============================================================================
# OpLab fetchers
# =============================================================================
def oplab_headers() -> Dict[str, str]:
    if not OPLAB_API_KEY:
        raise HTTPException(status_code=500, detail="OPLAB_API_KEY não configurada")
    if OPLAB_BEARER:
        return {"Authorization": f"Bearer {OPLAB_API_KEY}"}
    return {OPLAB_AUTH_HEADER: OPLAB_API_KEY}


def fetch_oplab_chain(ticker: str) -> Dict[str, Any]:
    """
    Busca estratégia 'covered' para o ticker (todas maturidades),
    depois filtramos client-side.
    """
    url = f"{OPLAB_BASE_URL}{OPLAB_COVERED_PATH}"
    params = {"symbol": ticker.upper()}
    try:
        r = requests.get(url, params=params, headers=oplab_headers(), timeout=20)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha na OpLab (covered): {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpLab covered HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not isinstance(data, dict) or "data" not in data:
        raise HTTPException(status_code=502, detail="Resposta inesperada da OpLab (covered)")
    return data


def fetch_oplab_series(ticker: str) -> Dict[str, Any]:
    """
    Metadados / últimos preços do subjacente (usamos close anterior, etc).
    """
    url = f"{OPLAB_BASE_URL}{OPLAB_SERIES_PATH}"
    params = {"symbols": ticker.upper()}
    try:
        r = requests.get(url, params=params, headers=oplab_headers(), timeout=20)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha na OpLab (series): {e}")
    if r.status_code != 200:
        # Não é fatal para a recomendação; só reportamos no debug
        return {"error": f"HTTP {r.status_code}", "text": r.text[:200]}
    try:
        return r.json()
    except Exception:
        return {"error": "invalid_json", "text": r.text[:200]}


def pick_maturity(nodes: List[Dict[str, Any]], venc: str, pm: float, criterio: str) -> Tuple[Dict[str, Any], str]:
    """
    Se venc = 'auto', escolhe a melhor maturidade com base em:
      - existir call com strike >= PM (se criterio '>=PM') ou
      - proximidade de PM (se criterio 'PM'),
      priorizando maior volume agregado do vencimento.
    Se venc for data, tenta matching exato no campo 'due_date'.
    """
    if not nodes:
        raise HTTPException(status_code=404, detail="Nenhuma maturidade retornada")

    # Normaliza para {due_date -> node}
    by_date: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        dd = node.get("due_date")  # "YYYY-MM-DD"
        if dd:
            by_date[dd] = node

    if venc != "auto":
        # Data específica
        node = by_date.get(venc)
        if not node:
            raise HTTPException(status_code=404, detail=f"Maturidade {venc} não encontrada")
        return node, "fixed"

    # AUTO: escolher o melhor vencimento
    best = None
    best_score = (-1, -1.0)  # (total_volume, -abs(strike-pm)) para desempate
    for node in nodes:
        strikes = node.get("strikes") or []
        # total volume / melhor proximidade de strike
        total_vol = 0
        best_prox = None
        for s in strikes:
            call = (s or {}).get("call") or {}
            vol = int(call.get("volume") or 0)
            total_vol += vol
            strike = _safe_float(s.get("strike"))
            if strike is None:
                continue
            if criterio == ">=PM" and strike < pm:
                continue
            diff = abs(strike - pm)
            if best_prox is None or diff < best_prox:
                best_prox = diff
        score = (total_vol, -best_prox if best_prox is not None else -1e9)
        if best is None or score > best_score:
            best = node
            best_score = score

    if best is None:
        # fallback: maior volume bruto, independentemente do critério
        best = max(nodes, key=lambda n: sum(int((s.get("call") or {}).get("volume") or 0)
                                            for s in (n.get("strikes") or [])))
        mode = "fallback-volume"
    else:
        mode = "auto"

    return best, mode


def calc_resultado_total(qty: int, pm: float, strike: float, premio: float, preco_venc: float) -> float:
    """
    Resultado bruto da estrutura (sem taxas): prêmio + (exercício se >= strike) - custo.
    """
    if preco_venc >= strike:
        venda = strike * qty
        custo = pm * qty
        premio_total = premio * qty
        return venda + premio_total - custo
    else:
        return premio * qty


# =============================================================================
# Rotas utilitárias
# =============================================================================
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}


@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})


# =============================================================================
# Endpoint principal
# =============================================================================
@app.post("/covered-call/suggest", response_model=SuggestOut)
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    criterio = data.criterio
    min_liq = int(data.min_liquidez)
    min_bid = float(data.min_bid)
    max_spread = float(data.max_spread)
    rel_floor = float(data.rel_floor_ratio)
    min_delta = data.min_delta
    max_delta = data.max_delta

    if qty < 100:
        raise HTTPException(status_code=400, detail="quantidade_acoes deve ser pelo menos 100")
    contratos = qty // 100

    if not USE_OPLAB:
        raise HTTPException(status_code=501, detail="Somente modo OpLab está habilitado neste deploy")

    # --- chama OpLab ---
    chain = fetch_oplab_chain(ticker)
    nodes = (chain.get("data") or {}).get("nodes") or []
    if not nodes:
        raise HTTPException(status_code=404, detail="OpLab sem nós de maturidade para este ticker")

    # série/último close do subjacente (não é obrigatório)
    series = fetch_oplab_series(ticker)

    # --- escolhe maturidade ---
    node, pick_mode = pick_maturity(nodes, data.vencimento, pm, criterio)
    due_date = node.get("due_date")
    days_to_maturity = int(node.get("days_to_maturity") or 1)
    strikes = node.get("strikes") or []

    # --- liquidez relativa: max volume no vencimento ---
    peers_vol_max = 0
    for s in strikes:
        vol = int(((s or {}).get("call") or {}).get("volume") or 0)
        if vol > peers_vol_max:
            peers_vol_max = vol

    # --- filtra por critérios (liquidez + gregas + strike) ---
    candidatos: List[Dict[str, Any]] = []
    for s in strikes:
        strike = _safe_float(s.get("strike"))
        if strike is None:
            continue
        if criterio == ">=PM" and strike < pm:
            continue

        call = (s or {}).get("call") or {}

        # Liquidez (absoluta + relativa + spread) e preço/fallback
        passed_liq, liq_dbg = passes_liquidity_filters(
            call_obj=call,
            min_liquidez=min_liq,
            rel_floor_ratio=rel_floor,
            peers_volume_max=peers_vol_max,
            min_bid=min_bid,
            max_spread=max_spread,
        )
        if not passed_liq:
            continue

        # Gregas (se houver)
        passed_greeks, gk_dbg = passes_greeks_filters(call, min_delta, max_delta)
        if not passed_greeks:
            continue

        preco, meta = price_for_calc(call, min_bid=min_bid, max_spread=max_spread)
        if preco is None or preco <= 0:
            continue

        ret_premio_pct = (preco / pm) * 100.0
        ret_anualizado_pct = ret_premio_pct * (252 / max(days_to_maturity, 1))

        candidatos.append({
            "strike": float(strike),
            "call": call,
            "preco_usado": float(preco),
            "preco_meta": meta,
            "retorno_premio_pct": ret_premio_pct,
            "retorno_anualizado_pct": ret_anualizado_pct,
            "liq_debug": liq_dbg,
            "greeks_debug": gk_dbg,
        })

    if not candidatos:
        raise HTTPException(status_code=404, detail="Nenhuma call passou nos filtros de liquidez/gregas")

    # Se criterio == "PM", ordena proximidade do PM; senão, pelo maior retorno anualizado
    if criterio == "PM":
        candidatos.sort(key=lambda x: abs(x["strike"] - pm))
    else:
        candidatos.sort(key=lambda x: x["retorno_anualizado_pct"], reverse=True)

    melhor = candidatos[0]
    strike = melhor["strike"]
    premio = melhor["preco_usado"]

    # --- cenários de resultado ---
    cenarios_vals = [pm - 10, pm, strike - 1, strike, strike + 3]
    cenarios = []
    for p in cenarios_vals:
        cenarios.append(Scenario(
            preco_acao_no_vencimento=round(p, 2),
            resultado_total=round(calc_resultado_total(qty, pm, strike, premio, p), 2),
            observacao="exercido" if p >= strike else "não exercido"
        ))

    # --- saída ---
    out = SuggestOut(
        recomendacao={
            "ticker": ticker,
            "due_date": due_date,
            "dias_ate_vencimento": days_to_maturity,
            "strike": strike,
            "ticker_opcao": (melhor["call"].get("symbol") or ""),
            "premio_usado": round(premio, 4),
            "preco_fonte": melhor["preco_meta"]["preco_fonte"],
            "volume": int((melhor["call"].get("volume") or 0)),
            "bid": float(melhor["preco_meta"]["bid"] or 0),
            "ask": float(melhor["preco_meta"]["ask"] or 0),
            "retorno_premio_pct": round(melhor["retorno_premio_pct"], 3),
            "retorno_anualizado_pct": round(melhor["retorno_anualizado_pct"], 3),
            "contratos_sugeridos": contratos,
            "greeks": extract_greeks(melhor["call"]),
        },
        cenarios=cenarios,
        fonte_dados="OpLab v3",
        debug={
            "pick_mode": pick_mode,
            "peers_volume_max": peers_vol_max,
            "total_candidatos": len(candidatos),
            "liquidez_params": {
                "min_liquidez": min_liq,
                "rel_floor_ratio": rel_floor,
                "min_bid": min_bid,
                "max_spread": max_spread,
            },
            "criterio": criterio,
            "series_peek": (series.get("data") or None) if isinstance(series, dict) else None,
        }
    )
    return out
