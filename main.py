from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, requests, pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from unidecode import unidecode
from urllib.parse import urlencode

app = FastAPI(title="robo-venda-coberta", version="1.0.0")

# ---------- Config ----------
USE_OPLAB = os.getenv("USE_OPLAB", "0") == "1"
OPLAB_BASE_URL = os.getenv("OPLAB_BASE_URL", "").rstrip("/")
OPLAB_CHAIN_PATH = os.getenv("OPLAB_CHAIN_PATH", "")
OPLAB_QUERY_TEMPLATE = os.getenv("OPLAB_QUERY_TEMPLATE", "ticker={ticker}&vencimento={vencimento}&type=CALL")
OPLAB_API_KEY = os.getenv("OPLAB_API_KEY", "")
OPLAB_AUTH_HEADER = os.getenv("OPLAB_AUTH_HEADER", "Authorization")  # ou "X-API-Key"

# ---------- Model ----------
class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str
    criterio: str = ">=PM"
    min_liquidez: int = 10

# ---------- Utils ----------
def _first_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return d[k]
        # também aceita variantes case-insensitive/underscore
        for dk in d.keys():
            if dk.replace("_", "").lower() == k.replace("_", "").lower():
                return d[dk]
    return None

def _extract_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ["data","results","items","content","options","chain","response"]:
            v = payload.get(k)
            if isinstance(v, list):
                return v
    # tenta pegar a 1ª lista encontrada
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []

def fetch_oplab_chain(ticker: str, vencimento: str) -> pd.DataFrame:
    if not (OPLAB_BASE_URL and OPLAB_CHAIN_PATH and OPLAB_API_KEY):
        raise HTTPException(status_code=500, detail="OpLab não configurado (OPLAB_* faltando)")

    # monta query a partir do template
    query_str = OPLAB_QUERY_TEMPLATE.format(ticker=ticker, vencimento=vencimento)
    url = f"{OPLAB_BASE_URL}{OPLAB_CHAIN_PATH}"
    if "?" in OPLAB_CHAIN_PATH:
        url = f"{OPLAB_BASE_URL}{OPLAB_CHAIN_PATH}&{query_str}"
    else:
        url = f"{OPLAB_BASE_URL}{OPLAB_CHAIN_PATH}?{query_str}"

    headers = {OPLAB_AUTH_HEADER: f"Bearer {OPLAB_API_KEY}" if OPLAB_AUTH_HEADER.lower()=="authorization" else OPLAB_API_KEY}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpLab retornou {r.status_code}")

    payload = r.json()
    items = _extract_items(payload)
    if not items:
        raise HTTPException(status_code=404, detail="OpLab: lista vazia")

    # normaliza em DataFrame
    df = pd.DataFrame(items)

    # mapeia campos comuns (várias alternativas)
    def pick_col(cands, default=None):
        for c in df.columns:
            norm = c.replace("_","").lower()
            for cand in cands:
                if norm == cand.replace("_","").lower():
                    return c
        return default

    col_symbol = pick_col(["symbol","ticker","code","option_symbol","optionsymbol"])
    col_strike = pick_col(["strike","strikeprice","exercise","exerciseprice"])
    col_last   = pick_col(["last","lastprice","price","optionprice","close"])
    col_bid    = pick_col(["bid","bestbid"])
    col_liq    = pick_col(["trades","numberoftrades","business","negocios","volume","liquidity"])
    col_exp    = pick_col(["expiration","expirationdate","maturity","duedate"])

    # cria colunas padronizadas
    out = pd.DataFrame()
    if col_symbol: out["ticker_opcao"] = df[col_symbol]
    if col_strike: out["strike"] = pd.to_numeric(df[col_strike], errors="coerce")
    # prêmio: prioriza last, depois bid
    premio_series = None
    if col_last: premio_series = pd.to_numeric(df[col_last], errors="coerce")
    if (premio_series is None or premio_series.isna().all()) and col_bid:
        premio_series = pd.to_numeric(df[col_bid], errors="coerce")
    out["premio"] = premio_series
    if col_liq:
        out["negocios"] = pd.to_numeric(df[col_liq], errors="coerce")
    else:
        out["negocios"] = 0
    if col_exp: out["vencimento"] = df[col_exp]

    out = out.dropna(subset=["strike","premio"])
    return out

def fetch_opcoes_net_chain(ticker: str) -> pd.DataFrame:
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Falha ao acessar opcoes.net")

    # tenta encontrar a maior tabela com 'strike' e 'últ/premio'
    dfs = pd.read_html(r.text, decimal=",", thousands=".")
    chain = None
    best_len = 0
    for df in dfs:
        cols = [unidecode(str(c)).lower() for c in df.columns]
        if any("strike" in c for c in cols) and any(("ult" in c) or ("premio" in c) for c in cols):
            if len(df) > best_len:
                chain, best_len = df, len(df)

    if chain is None:
        # plano B: usar BS4 pra extrair tabela manualmente
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if not table:
            raise HTTPException(status_code=404, detail="Grade de opções não localizada")
        chain = pd.read_html(str(table), decimal=",", thousands=".")[0]

    rename_map = {}
    for c in chain.columns:
        cl = unidecode(str(c)).strip().lower()
        if "strike" in cl: rename_map[c] = "strike"
        elif "neg" in cl: rename_map[c] = "negocios"
        elif ("ult" in cl) or ("premio" in cl): rename_map[c] = "premio"
        elif "venc" in cl: rename_map[c] = "vencimento"
        elif "codigo" in cl or "ticker" in cl or "simbolo" in cl: rename_map[c] = "ticker_opcao"
    chain = chain.rename(columns=rename_map)

    for col in ["strike","premio"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    if "negocios" in chain.columns:
        chain["negocios"] = pd.to_numeric(chain["negocios"], errors="coerce").fillna(0)
    else:
        chain["negocios"] = 0
    chain = chain.dropna(subset=["strike","premio"])
    return chain

# ---------- Rotas utilitárias ----------
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

# ---------- Endpoint principal ----------
@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper().strip()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    criterio = data.criterio.upper().strip()
    min_liq = int(data.min_liquidez)

    # pega chain
    if USE_OPLAB:
        df = fetch_oplab_chain(ticker, data.vencimento)
    else:
        df = fetch_opcoes_net_chain(ticker)

    # filtra liquidez
    df = df[df["negocios"] >= min_liq]
    if df.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente")

    # datas
    dias = None
    if data.vencimento:
        try:
            venc = datetime.strptime(data.vencimento, "%Y-%m-%d")
            dias = max((venc - datetime.now()).days, 1)
        except Exception:
            dias = None

    # métricas
    df["retorno_premio_pct"] = (df["premio"] / pm) * 100.0
    if dias:
        df["retorno_anualizado_pct"] = df["retorno_premio_pct"] * (252 / dias)
    else:
        df["retorno_anualizado_pct"] = df["retorno_premio_pct"]

    # critério de strike
    if criterio == ">=PM":
        df = df[df["strike"] >= pm]
    elif criterio == "PM":
        df["diff"] = (df["strike"] - pm).abs()
        df = df.sort_values("diff")

    if df.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério")

    best = df.sort_values(by="retorno_anualizado_pct", ascending=False).iloc[0]
    strike = float(best["strike"]); premio = float(best["premio"])
    negocios = int(best.get("negocios", 0))
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
    for p in [pm-10, pm, strike-1, strike, strike+3]:
        cenarios.append({
            "preco_acao_no_vencimento": round(p, 2),
            "resultado_total": round(resultado(p), 2),
            "observacao": "exercido" if p >= strike else "nao exercido"
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
            "contratos_sugeridos": contratos
        },
        "cenarios": cenarios,
        "fonte_dados": "OpLab" if USE_OPLAB else "opcoes.net"
    }
