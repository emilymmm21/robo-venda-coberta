# main.py
from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import httpx
from bs4 import BeautifulSoup
from unidecode import unidecode
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_TITLE = "robo-venda-coberta"
APP_VERSION = "1.0.0"

# -------- Config por ENV --------
USE_OPLAB = os.getenv("USE_OPLAB", "0") == "1"
OPLAB_BASE_URL = os.getenv("OPLAB_BASE_URL", "").rstrip("/")  # ex: https://api.oplab.com.br
OPLAB_CHAIN_PATH = os.getenv("OPLAB_CHAIN_PATH", "")          # ex: /v1/options/chain
OPLAB_API_KEY = os.getenv("OPLAB_API_KEY", "")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# -------- FastAPI --------
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url=None,
)


# ------------ MODELOS ------------
class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str           # "YYYY-MM-DD"
    criterio: str = ">=PM"    # ou "PM"
    min_liquidez: int = 10    # negociações mínimas


# ------------ HELPERS ------------
def _parse_date(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(s.split("T")[0], fmt) if "T" in s else datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _annualize(premio_pct: float, dias: int) -> float:
    dias = max(dias, 1)
    return premio_pct * (252 / dias)


def _normalize_key(k: Any) -> str:
    return unidecode(str(k)).strip().lower().replace(" ", "_")


def _to_num(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    # tenta formato BR
    s1 = s.replace(".", "").replace(",", ".")
    try:
        return float(s1)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None


# ------------ PROVIDER: OPLAB ------------
def fetch_oplab_chain(ticker: str, venc_dt: Optional[datetime]) -> pd.DataFrame:
    """
    Cliente genérico pra OpLab.
    **IMPORTANTE**: como não tenho acesso à doc pública aqui, deixei o cliente
    configurável via ENV:
      - OPLAB_BASE_URL      (ex: https://api.oplab.com.br)
      - OPLAB_CHAIN_PATH    (ex: /v1/options/chain)
      - OPLAB_API_KEY       (Bearer)
    O endpoint deve retornar uma lista de contratos de CALL com campos
    que permitam identificar pelo menos: strike, preço (último/bid/ask),
    negócios/volume, ticker/símbolo e vencimento.
    """
    if not (OPLAB_BASE_URL and OPLAB_CHAIN_PATH and OPLAB_API_KEY):
        raise HTTPException(
            status_code=500,
            detail="OPLab não configurado: defina OPLAB_BASE_URL, OPLAB_CHAIN_PATH e OPLAB_API_KEY."
        )

    url = f"{OPLAB_BASE_URL}{OPLAB_CHAIN_PATH}"
    params = {"ticker": ticker.upper(), "type": "CALL"}
    if venc_dt:
        # se seu endpoint aceitar, mande a data também:
        params["maturity"] = venc_dt.strftime("%Y-%m-%d")

    headers = {"Authorization": f"Bearer {OPLAB_API_KEY}"}

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"OPLab HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar OPLab: {e}")

    # Tenta ser tolerante com formatos diferentes:
    # Pode vir {"results":[...]}, {"contracts":[...]}, ou diretamente uma lista
    items = None
    if isinstance(data, dict):
        for key in ("results", "contracts", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
    if items is None and isinstance(data, list):
        items = data

    if not items:
        raise HTTPException(status_code=404, detail="OPLab: nenhum contrato retornado")

    rows: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # normaliza chaves
        norm = { _normalize_key(k): v for k, v in it.items() }

        # tenta mapear campos mais comuns
        strike = _to_num(norm.get("strike") or norm.get("preco_exercicio"))
        # preço do prêmio: tentar "last", "lastprice", "ultimo", "premium", "bid"/"ask" média
        premio = _to_num(norm.get("last") or norm.get("lastprice") or norm.get("ultimo") or norm.get("premium"))
        if premio is None:
            bid = _to_num(norm.get("bid"))
            ask = _to_num(norm.get("ask"))
            if bid and ask:
                premio = (bid + ask) / 2.0
            elif bid:
                premio = bid
            elif ask:
                premio = ask

        negocios = None
        for k in ("trades", "tradecount", "negocios", "business_quantity", "volume"):
            v = _to_num(norm.get(k))
            if v is not None:
                negocios = int(v)
                break

        ticker_opcao = (norm.get("symbol") or norm.get("ticker") or norm.get("codigo") or "").upper()
        venc = norm.get("maturity") or norm.get("vencimento") or norm.get("expirationdate")
        venc_parsed = _parse_date(venc) if isinstance(venc, str) else None

        # filtra só CALL se vier misturado
        tipo = str(norm.get("type") or norm.get("tipo") or "").upper()
        if tipo and tipo not in ("C", "CALL", "OPC_CALL"):
            continue

        if strike is None or premio is None:
            continue

        rows.append({
            "ticker_opcao": ticker_opcao,
            "strike": strike,
            "premio": premio,
            "negocios": negocios if negocios is not None else 0,
            "vencimento": venc_parsed.strftime("%Y-%m-%d") if venc_parsed else None,
        })

    if not rows:
        raise HTTPException(status_code=404, detail="OPLab: nenhum contrato CALL utilizável")

    df = pd.DataFrame(rows)
    return df


# ------------ PROVIDER: opcoes.net (scraping) ------------
def fetch_opcoes_net_chain(ticker: str) -> pd.DataFrame:
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"opcoes.net HTTP {r.status_code}")

    # 1) tenta HTML tables direto
    dfs = []
    try:
        dfs = pd.read_html(r.text, decimal=",", thousands=".")
    except Exception:
        dfs = []

    # 2) se falhar, tenta montar a partir do HTML via BS4
    if not dfs:
        soup = BeautifulSoup(r.text, "lxml")
        tables = soup.find_all("table")
        for tb in tables:
            try:
                df = pd.read_html(str(tb), decimal=",", thousands=".")[0]
                dfs.append(df)
            except Exception:
                pass

    # varre possíveis tabelas de calls
    chosen = None
    for df in dfs:
        cols = [_normalize_key(c) for c in df.columns.astype(str)]
        has_strike = any("strike" in c or "exercicio" in c for c in cols)
        has_premio = any("ult" in c or "premio" in c or "ultimo" in c for c in cols)
        if has_strike and has_premio:
            chosen = df
            break

    if chosen is None or chosen.empty:
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

    rename = {}
    for c in chosen.columns:
        cl = _normalize_key(c)
        if "strike" in cl or "exercicio" in cl:
            rename[c] = "strike"
        elif "neg" in cl or "business" in cl or "volume" in cl:
            rename[c] = "negocios"
        elif "ult" in cl or "premio" in cl or "ultimo" in cl:
            rename[c] = "premio"
        elif "venc" in cl or "maturity" in cl:
            rename[c] = "vencimento"
        elif "codigo" in cl or "ticker" in cl or "symbol" in cl:
            rename[c] = "ticker_opcao"

    df = chosen.rename(columns=rename)
    for col in ("strike", "premio"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "negocios" in df.columns:
        df["negocios"] = pd.to_numeric(df["negocios"], errors="coerce").fillna(0).astype(int)
    else:
        df["negocios"] = 0

    df = df.dropna(subset=["strike", "premio"])
    if df.empty:
        raise HTTPException(status_code=404, detail="Sem dados válidos de strike/premio")
    return df


def get_chain(ticker: str, venc_dt: Optional[datetime]) -> pd.DataFrame:
    if USE_OPLAB:
        return fetch_oplab_chain(ticker, venc_dt)
    # fallback
    return fetch_opcoes_net_chain(ticker)


# ------------ ENDPOINTS ------------
@app.get("/")
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    criterio = (data.criterio or ">=PM").upper().strip()
    venc_dt = _parse_date(data.vencimento)
    min_liq = int(data.min_liquidez)

    if venc_dt is None:
        raise HTTPException(status_code=422, detail="vencimento deve estar no formato YYYY-MM-DD")

    chain = get_chain(ticker, venc_dt)

    # filtra liquidez
    if "negocios" in chain.columns:
        chain = chain[chain["negocios"] >= min_liq]
    if chain.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente")

    # filtra por vencimento se disponível
    if "vencimento" in chain.columns:
        chain["venc_dt"] = chain["vencimento"].apply(lambda s: _parse_date(str(s)) if pd.notna(s) else None)
        chain = chain[chain["venc_dt"].apply(lambda d: (d == venc_dt) if d else True)]
    if chain.empty:
        # se não houver dado de vencimento, segue com a melhor disponível
        pass

    # cálculo de retornos
    dias = max((venc_dt - datetime.now()).days, 1)
    chain["retorno_premio_pct"] = (chain["premio"] / pm) * 100.0
    chain["retorno_anualizado_pct"] = chain["retorno_premio_pct"].apply(lambda x: _annualize(x, dias))

    # critério de strike
    if criterio == ">=PM":
        chain = chain[chain["strike"] >= pm]
    elif criterio == "PM":
        chain["diff_pm"] = (chain["strike"] - pm).abs()
        chain = chain.sort_values(by="diff_pm")
    if chain.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério")

    best = chain.sort_values(by="retorno_anualizado_pct", ascending=False).iloc[0]

    strike = float(best["strike"])
    premio = float(best["premio"])
    negocios = int(best.get("negocios", 0))
    venc_out = best.get("vencimento")
    contratos = qty // 100

    def resultado(preco_venc: float) -> float:
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
            "preco_acao_no_vencimento": round(float(p), 2),
            "resultado_total": round(float(resultado(p)), 2),
            "observacao": "exercido" if p >= strike else "nao_exercido"
        })

    return {
        "recomendacao": {
            "ticker_subjacente": ticker,
            "ticker_opcao": best.get("ticker_opcao", ""),
            "strike": strike,
            "premio": premio,
            "negocios_hoje": negocios,
            "dias_ate_vencimento": dias,
            "retorno_premio_pct": round(float(best["retorno_premio_pct"]), 2),
            "retorno_anualizado_pct": round(float(best["retorno_anualizado_pct"]), 2),
            "contratos_sugeridos": contratos,
        },
        "cenarios": cenarios,
        "fonte_dados": "OPLab" if USE_OPLAB else "opcoes.net (scraping)",
        "observacoes": "Para OPLab, configure OPLAB_BASE_URL, OPLAB_CHAIN_PATH e OPLAB_API_KEY.",
    }
