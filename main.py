from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- Playwright (para renderizar o Opcoes.net quando carrega via JS) ---
USE_PLAYWRIGHT = os.getenv("SCRAPER_PROVIDER", "playwright").lower() == "playwright"
if USE_PLAYWRIGHT:
    # Importa só quando precisar (evita custo em ambientes que não usam)
    from playwright.sync_api import sync_playwright

app = FastAPI(title="robo-venda-coberta", version="1.0.0")


# --------------------------- MODELOS ---------------------------
class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: Optional[str] = None  # "YYYY-MM-DD" (opcional)
    criterio: str = ">=PM"            # ">=PM" ou "PM"
    min_liquidez: int = 10            # filtro por número de negócios


# --------------------------- ROTAS BÁSICAS ---------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})


# ----------------------- FUNÇÕES AUXILIARES -----------------------
def _read_html_tables(html: str) -> List[pd.DataFrame]:
    """Tenta ler todas as tabelas com pandas.read_html e retorna uma lista."""
    try:
        return pd.read_html(html, decimal=",", thousands=".")
    except ValueError:
        return []

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas comuns da grade de opções para nomes padronizados."""
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "strike" in cl or "exerc" in cl:
            rename_map[c] = "strike"
        elif "negó" in cl or "negoc" in cl or "neg" in cl:
            rename_map[c] = "negocios"
        elif "últ" in cl or "preço" in cl or "prem" in cl or "ult" in cl:
            rename_map[c] = "premio"
        elif "venc" in cl or "expira" in cl:
            rename_map[c] = "vencimento"
        elif "código" in cl or "ticker" in cl or "símbolo" in cl or "codigo" in cl:
            rename_map[c] = "ticker_opcao"

    df = df.rename(columns=rename_map)

    # Tipos
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "premio" in df.columns:
        df["premio"] = pd.to_numeric(df["premio"], errors="coerce")
    if "negocios" in df.columns:
        df["negocios"] = pd.to_numeric(df["negocios"], errors="coerce").fillna(0).astype(int)
    else:
        df["negocios"] = 0

    return df

def _pick_options_table(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Escolhe a tabela que parece ser a grade de opções de CALLs."""
    best = None
    for df in dfs:
        dfn = _normalize_columns(df.copy())
        cols = set([c.lower() for c in dfn.columns.astype(str)])
        # Heurística: precisa ter strike e premio; ticker_opcao ajuda muito.
        if {"strike", "premio"}.issubset(cols):
            best = dfn
            if "ticker_opcao" in cols:
                return dfn
    return best

def fetch_chain_with_requests(ticker: str) -> pd.DataFrame:
    """Coleta HTML estático (sem JS). Útil se a página servir a tabela de primeira."""
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Falha ao acessar Opcoes.net (requests)")

    dfs = _read_html_tables(r.text)
    if not dfs:
        return pd.DataFrame()

    chain = _pick_options_table(dfs)
    return chain if chain is not None else pd.DataFrame()

def fetch_chain_with_playwright(ticker: str) -> pd.DataFrame:
    """Renderiza a página com Chromium headless e extrai a grade de opções."""
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=60_000, wait_until="domcontentloaded")

        # Aguarda alguma tabela aparecer
        try:
            page.wait_for_selector("table", timeout=60_000)
        except Exception:
            # Tenta um pequeno delay extra e segue assim mesmo
            time.sleep(2)

        html = page.content()
        context.close()
        browser.close()

    # Agora parseia o HTML renderizado
    dfs = _read_html_tables(html)
    if not dfs:
        return pd.DataFrame()

    chain = _pick_options_table(dfs)
    return chain if chain is not None else pd.DataFrame()

def fetch_option_chain(ticker: str) -> pd.DataFrame:
    """Tenta playwright primeiro (padrão). Cai para requests se necessário."""
    if USE_PLAYWRIGHT:
        chain = fetch_chain_with_playwright(ticker)
        if not chain.empty:
            return chain
        # fallback
        chain = fetch_chain_with_requests(ticker)
        return chain
    else:
        chain = fetch_chain_with_requests(ticker)
        return chain


# ----------------------- LÓGICA DE SUGESTÃO -----------------------
@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper().strip()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    criterio = (data.criterio or ">=PM").upper().strip()
    min_liq = int(data.min_liquidez)

    # (vencimento é opcional; se vier, filtra depois)
    venc_dt: Optional[datetime] = None
    if data.vencimento:
        try:
            venc_dt = datetime.strptime(data.vencimento, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="vencimento deve estar em YYYY-MM-DD")

    # 1) baixa a grade
    chain = fetch_option_chain(ticker)
    if chain.empty:
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

    # 2) normaliza e filtra liquidez
    chain = chain.dropna(subset=["strike", "premio"])
    chain = chain[chain["premio"] > 0]
    if "negocios" in chain.columns:
        chain = chain[chain["negocios"] >= min_liq]
    if chain.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente após filtros")

    # 3) se houver coluna vencimento e o usuário enviou a data, filtra
    if venc_dt is not None and "vencimento" in chain.columns:
        def parse_venc(x: Any) -> Optional[datetime]:
            s = str(x).strip()
            # Aceita "20/09/2025" ou "2025-09-20" etc.
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    pass
            return None

        chain["__venc"] = chain["vencimento"].apply(parse_venc)
        chain = chain[chain["__venc"].notna()]
        chain = chain[chain["__venc"] == venc_dt]
        if chain.empty:
            raise HTTPException(status_code=404, detail="Não há opções para o vencimento informado")

    # 4) métricas de retorno
    hoje = datetime.now()
    dias = 30  # default (se não souber o vencimento)
    if venc_dt is not None:
        dd = (venc_dt - hoje).days
        dias = max(dd, 1)

    chain["retorno_premio_pct"] = (chain["premio"] / pm) * 100.0
    chain["retorno_anualizado_pct"] = chain["retorno_premio_pct"] * (252 / dias)

    # 5) aplica o critério de strike
    if criterio == ">=PM":
        chain = chain[chain["strike"] >= pm]
        if chain.empty:
            raise HTTPException(status_code=404, detail="Nenhum strike >= PM encontrado")
    elif criterio == "PM":
        chain["__diff_pm"] = (chain["strike"] - pm).abs()
        chain = chain.sort_values(by="__diff_pm")
    else:
        raise HTTPException(status_code=400, detail="criterio deve ser '>=PM' ou 'PM'")

    # 6) escolhe a melhor (pelo retorno anualizado)
    chain = chain.sort_values(by="retorno_anualizado_pct", ascending=False)
    best = chain.iloc[0]

    strike = float(best["strike"])
    premio = float(best["premio"])
    negocios = int(best.get("negocios", 0))
    ticker_opcao = str(best.get("ticker_opcao", "")).strip()
    contratos = qty // 100

    # 7) cenários simples
    def resultado(preco_venc: float) -> float:
        # ganho por prêmio sempre entra
        premio_total = premio * qty
        if preco_venc >= strike:
            venda = strike * qty
            custo = pm * qty
            return (venda + premio_total - custo)
        else:
            return premio_total

    cenarios = []
    for p in [pm - 10, pm, strike - 1, strike, strike + 3]:
        cenarios.append({
            "preco_acao_no_vencimento": round(float(p), 2),
            "resultado_total": round(resultado(float(p)), 2),
            "observacao": "exercido" if p >= strike else "não exercido"
        })

    return {
        "recomendacao": {
            "ticker_subjacente": ticker,
            "ticker_opcao": ticker_opcao,
            "strike": strike,
            "premio": premio,
            "negocios_hoje": negocios,
            "dias_ate_vencimento": dias if venc_dt else None,
            "retorno_premio_pct": round(float(best["retorno_premio_pct"]), 2),
            "retorno_anualizado_pct": round(float(best["retorno_anualizado_pct"]), 2),
            "contratos_sugeridos": contratos
        },
        "cenarios": cenarios,
        "fonte_dados": "opcoes.net (renderizado com Playwright)" if USE_PLAYWRIGHT else "opcoes.net (requests)"
    }
