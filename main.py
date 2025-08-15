import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="robo-venda-coberta", version="1.0.0")

# utilitárias
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

# -----------------------------------------------------------------------------
# Modelo de entrada (mantido)
# -----------------------------------------------------------------------------
class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str                # ex: "2025-09-20" (formato que a OpLab retorna)
    criterio: str = ">=PM"         # ">=PM" ou "PM"
    min_liquidez: int = 10         # mínimo de negócios

# -----------------------------------------------------------------------------
# Integração OpLab
# -----------------------------------------------------------------------------
def _oplab_headers():
    api_key = os.getenv("OPLAB_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPLab API key ausente. Defina OPLAB_API_KEY no Render.")
    auth_header = os.getenv("OPLAB_AUTH_HEADER", "Access-Token")
    bearer = os.getenv("OPLAB_BEARER", "0") == "1"
    return {auth_header: f"Bearer {api_key}" if bearer else api_key}

def _oplab_base():
    base = os.getenv("OPLAB_BASE_URL", "https://api.oplab.com.br/v3").rstrip("/")
    return base

def fetch_oplab_series(ticker: str):
    """
    Lista os vencimentos/séries disponíveis para um subjacente.
    GET /v3/market/instruments/series/{symbol}
    """
    base = _oplab_base()
    path = os.getenv("OPLAB_SERIES_PATH", "/market/instruments/series").rstrip("/")
    url = f"{base}{path}/{ticker.upper()}"
    try:
        r = requests.get(url, headers=_oplab_headers(), timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao contactar OpLab (series): {e}")
    if r.status_code == 401:
        raise HTTPException(status_code=502, detail="Não autorizado na OpLab (token inválido/expirado).")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Erro OpLab {r.status_code}: {r.text[:200]}")
    return r.json()

def fetch_oplab_covered_chain(ticker: str, vencimento: str, min_liq: int):
    """
    Busca as opções elegíveis para venda coberta.
    GET /v3/market/options/strategies/covered?underlying={TICKER}
    Filtra por due_date == vencimento (YYYY-MM-DD) e trades >= min_liq.
    """
    base = _oplab_base()
    path = os.getenv("OPLAB_COVERED_PATH", "/market/options/strategies/covered")
    url = f"{base}{path}"
    params = {"underlying": ticker.upper()}

    try:
        r = requests.get(url, headers=_oplab_headers(), params=params, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao contactar OpLab (covered): {e}")

    if r.status_code == 401:
        raise HTTPException(status_code=502, detail="Não autorizado na OpLab (token inválido/expirado).")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Erro OpLab {r.status_code}: {r.text[:200]}")

    data = r.json()
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Resposta inesperada da OpLab (covered).")

    alvo_venc = vencimento.strip()
    chain = []
    for it in data:
        try:
            due = it.get("due_date")  # "YYYY-MM-DD"
            if alvo_venc and due != alvo_venc:
                continue
            trades = int(it.get("trades", 0) or 0)
            if trades < min_liq:
                continue
            strike = float(it.get("strike"))
            # prêmio: prioriza bid, depois ask, senão close
            raw_premio = it.get("bid") or it.get("ask") or it.get("close")
            if raw_premio in (None, 0):
                continue
            premio = float(raw_premio)

            chain.append({
                "ticker_opcao": it.get("symbol", ""),
                "strike": strike,
                "premio": premio,
                "negocios": trades,
                "vencimento": due,
            })
        except Exception:
            # ignora registros quebrados
            continue

    return chain

# Endpoint extra de apoio: lista séries/vencimentos pela própria API
@app.get("/oplab/series/{ticker}")
def api_series(ticker: str):
    series = fetch_oplab_series(ticker)
    return series

# -----------------------------------------------------------------------------
# Regra de sugestão da covered call (mantida)
# -----------------------------------------------------------------------------
@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    # parâmetros
    ticker = data.ticker_subjacente.upper()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    venc_str = data.vencimento.strip()       # OpLab usa "YYYY-MM-DD"
    criterio = data.criterio.upper()
    min_liq = int(data.min_liquidez)

    # garante que estamos usando OpLab
    if os.getenv("USE_OPLAB", "0") != "1":
        raise HTTPException(status_code=503, detail="USE_OPLAB=1 requerido no Render.")

    # 1) busca chain via OpLab (covered)
    chain = fetch_oplab_covered_chain(ticker, venc_str, min_liq)
    if not chain:
        # ajuda: sugere vencimentos existentes
        try:
            s = fetch_oplab_series(ticker)
        except Exception:
            s = []
        hint = {"vencimentos_disponiveis": s} if s else {}
        raise HTTPException(status_code=404, detail={"msg": "Sem opções após filtros (vencimento/liquidez).", **hint})

    # 2) calcula retornos
    hoje = datetime.now()
    try:
        venc = datetime.strptime(venc_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de vencimento inválido. Use YYYY-MM-DD.")
    dias = max((venc - hoje).days, 1)

    df = pd.DataFrame(chain)
    df["retorno_premio_pct"] = df["premio"] / pm * 100.0
    df["retorno_anualizado_pct"] = df["retorno_premio_pct"] * (252 / dias)

    # 3) filtra pelo critério
    if criterio == ">=PM":
        df = df[df["strike"] >= pm]
    elif criterio == "PM":
        df["diff"] = (df["strike"] - pm).abs()
        df = df.sort_values(by="diff")

    if df.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério.")

    melhor = df.sort_values(by="retorno_anualizado_pct", ascending=False).iloc[0]
    strike = float(melhor["strike"])
    premio = float(melhor["premio"])
    contratos = qty // 100

    # 4) cenários
    def resultado(preco_venc):
        if preco_venc >= strike:
            venda = strike * qty
            custo = pm * qty
            premio_total = premio * qty
            return venda + premio_total - custo
        else:
            return premio * qty

    cenarios = []
    for p in [pm - 10, pm, strike - 1, strike, strike + 3]:
        cenarios.append({
            "preco_acao_no_vencimento": round(p, 2),
            "resultado_total": round(resultado(p), 2),
            "observacao": "exercido" if p >= strike else "não exercido"
        })

    return {
        "recomendacao": {
            "strike": strike,
            "ticker_opcao": melhor.get("ticker_opcao", ""),
            "premio": premio,
            "negocios_hoje": int(melhor["negocios"]),
            "dias_ate_vencimento": dias,
            "retorno_premio_pct": round(float(melhor["retorno_premio_pct"]), 2),
            "retorno_anualizado_pct": round(float(melhor["retorno_anualizado_pct"]), 2),
            "contratos_sugeridos": contratos
        },
        "cenarios": cenarios,
        "fonte_dados": "OpLab API"
    }
