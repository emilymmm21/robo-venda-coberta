# app.py
from datetime import datetime
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from unidecode import unidecode


# -----------------------------------------------------------------------------
# FastAPI (uma única instância)
# -----------------------------------------------------------------------------
app = FastAPI(title="robo-venda-coberta", version="1.0.0")


# -----------------------------------------------------------------------------
# Rotas utilitárias
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})


# -----------------------------------------------------------------------------
# Modelo de entrada
# -----------------------------------------------------------------------------
class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str          # "YYYY-MM-DD"
    criterio: str = ">=PM"   # "> =PM" (fora do dinheiro) ou "PM" (próximo ao PM)
    min_liquidez: int = 10   # mínimo de negócios


# -----------------------------------------------------------------------------
# Funções de parsing
# -----------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Remove acentos e normaliza minúsculas/espaços."""
    return unidecode(str(s)).lower().strip()

def _is_strike_col(name: str) -> bool:
    c = _normalize(name)
    # Exemplos possíveis no site: "Strike", "Preço de exercício", "Preço exercício"
    return ("strike" in c) or ("exercicio" in c) or ("preco" in c and "exercicio" in c)

def _is_premio_col(name: str) -> bool:
    c = _normalize(name)
    # Pode aparecer como "Últ.", "Ultimo", "Prêmio"
    return ("ult" in c) or ("ultimo" in c) or ("premio" in c) or ("preco" in c and "ult" in c)

def _is_negocios_col(name: str) -> bool:
    c = _normalize(name)
    # "Negócios", "Negociacoes", "Neg"
    return "neg" in c

def _is_venc_col(name: str) -> bool:
    c = _normalize(name)
    return "venc" in c

def _is_ticker_opcao_col(name: str) -> bool:
    c = _normalize(name)
    return ("codigo" in c) or ("ticker" in c)

def _try_map_chain(df: pd.DataFrame) -> pd.DataFrame | None:
    """Tenta mapear colunas essenciais (strike/premio). Retorna None se não servir."""
    norm_cols = [_normalize(c) for c in df.columns]
    has_strike = any(_is_strike_col(c) for c in df.columns)
    has_premio = any(_is_premio_col(c) for c in df.columns)
    if not (has_strike and has_premio):
        return None

    rename_map = {}
    for orig in df.columns:
        if _is_strike_col(orig):
            rename_map[orig] = "strike"
        elif _is_premio_col(orig):
            rename_map[orig] = "premio"
        elif _is_negocios_col(orig):
            rename_map[orig] = "negocios"
        elif _is_venc_col(orig):
            rename_map[orig] = "vencimento"
        elif _is_ticker_opcao_col(orig):
            rename_map[orig] = "ticker_opcao"

    out = df.rename(columns=rename_map)

    # precisa no mínimo strike e premio
    if not {"strike", "premio"}.issubset(set(out.columns)):
        return None
    return out


def _fetch_option_chain_html(ticker: str) -> str:
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException:
        raise HTTPException(status_code=500, detail="Erro de rede ao acessar Opcoes.net")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Falha ao acessar Opcoes.net")
    return r.text


def _extract_tables(html: str) -> List[pd.DataFrame]:
    """Primeiro tenta ler todas as tabelas direto do HTML. Se vier vazio, usa BS4 por table."""
    dfs: List[pd.DataFrame] = []
    try:
        dfs = pd.read_html(html, decimal=",", thousands=".")
    except ValueError:
        # sem tabelas diretas
        dfs = []

    if not dfs:
        soup = BeautifulSoup(html, "lxml")
        for t in soup.select("table"):
            try:
                piece = pd.read_html(str(t), decimal=",", thousands=".")
                if piece:
                    dfs.extend(piece)
            except Exception:
                pass
    return dfs


# -----------------------------------------------------------------------------
# Endpoint principal
# -----------------------------------------------------------------------------
@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper().strip()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    venc_str = data.vencimento.strip()
    criterio = data.criterio.upper().strip()
    min_liq = int(data.min_liquidez)

    if qty < 100:
        raise HTTPException(status_code=400, detail="Quantidade de ações deve ser >= 100.")
    try:
        venc = datetime.strptime(venc_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de vencimento inválido. Use YYYY-MM-DD.")

    html = _fetch_option_chain_html(ticker)
    dfs = _extract_tables(html)
    if not dfs:
        raise HTTPException(status_code=404, detail="Nenhuma tabela encontrada")

    chain = None
    for df in dfs:
        df2 = _try_map_chain(df)
        if df2 is not None:
            chain = df2
            break
    if chain is None:
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

    # ------- tratamento dos dados -------
    for col in ["strike", "premio"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    if "negocios" in chain.columns:
        chain["negocios"] = pd.to_numeric(chain["negocios"], errors="coerce").fillna(0).astype(int)
    else:
        chain["negocios"] = 0

    chain = chain.dropna(subset=["strike", "premio"])
    chain = chain[chain["negocios"] >= min_liq]
    if chain.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente")

    hoje = datetime.now()
    dias = (venc - hoje).days
    dias = max(dias, 1)

    chain["retorno_premio_pct"] = (chain["premio"] / pm) * 100.0
    chain["retorno_anualizado_pct"] = chain["retorno_premio_pct"] * (252 / dias)

    # ------- filtro pelo critério -------
    if criterio == ">=PM":
        chain = chain[chain["strike"] >= pm]
    elif criterio == "<=PM":
        chain = chain[chain["strike"] <= pm]
    elif criterio == "PM":
        chain["diff_pm"] = (chain["strike"] - pm).abs()
        chain = chain.sort_values(by="diff_pm")
    else:
        # fallback: mantém tudo se critério desconhecido
        pass

    if chain.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério")

    # Escolhe melhor por retorno anualizado
    chain = chain.sort_values(by="retorno_anualizado_pct", ascending=False)
    melhor = chain.iloc[0]

    strike = float(melhor["strike"])
    premio = float(melhor["premio"])
    negocios = int(melhor.get("negocios", 0))
    dias_venc = dias
    contratos = qty // 100

    def resultado(preco_venc: float) -> float:
        # Resultado total considerando 100 ações por contrato (qty é o total de ações)
        # Se exercer, vende a ação a strike e ainda fica com o prêmio.
        if preco_venc >= strike:
            venda = strike * qty
            custo = pm * qty
            premio_total = premio * qty
            return venda + premio_total - custo
        else:
            # não exercido: fica com as ações e com o prêmio
            return premio * qty

    cenarios = []
    for p in [pm - 10, pm, max(strike - 1, 0), strike, strike + 3]:
        cenarios.append({
            "preco_acao_no_vencimento": round(float(p), 2),
            "resultado_total": round(float(resultado(p)), 2),
            "observacao": "exercido" if p >= strike else "nao_exercido"
        })

    return {
        "recomendacao": {
            "ticker_subjacente": ticker,
            "ticker_opcao": str(melhor.get("ticker_opcao", "")),
            "strike": round(strike, 2),
            "premio": round(premio, 2),
            "negocios_hoje": negocios,
            "dias_ate_vencimento": dias_venc,
            "retorno_premio_pct": round(float(melhor["retorno_premio_pct"]), 2),
            "retorno_anualizado_pct": round(float(melhor["retorno_anualizado_pct"]), 2),
            "contratos_sugeridos": contratos,
        },
        "cenarios": cenarios,
        "fonte_dados": "opcoes.net.br (scraping)"
    }
