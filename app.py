from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

app = FastAPI()

class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str
    criterio: str = ">=PM"
    min_liquidez: int = 10

@app.post("/covered-call/suggest")
def suggest(data: SuggestIn):
    ticker = data.ticker_subjacente.upper()
    pm = float(data.preco_medio)
    qty = int(data.quantidade_acoes)
    venc_str = data.vencimento
    criterio = data.criterio.upper()
    min_liq = data.min_liquidez

    # Coleta a página do Opcoes.net
    url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Falha ao acessar Opcoes.net")

    dfs = pd.read_html(r.text, decimal=",", thousands=".")
    if not dfs:
        raise HTTPException(status_code=404, detail="Nenhuma tabela encontrada")

    # Tenta achar a tabela certa
    chain = None
    for df in dfs:
        cols = [c.lower() for c in df.columns.astype(str)]
        if "strike" in str(cols) and ("últ" in str(cols) or "premio" in str(cols)):
            chain = df
            break

    if chain is None:
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

    # Padroniza colunas
    rename_map = {}
    for c in chain.columns:
        cl = str(c).strip().lower()
        if "strike" in cl: rename_map[c] = "strike"
        elif "negó" in cl or "negoc" in cl: rename_map[c] = "negocios"
        elif "últ" in cl or "premio" in cl: rename_map[c] = "premio"
        elif "venc" in cl: rename_map[c] = "vencimento"
        elif "código" in cl or "ticker" in cl: rename_map[c] = "ticker_opcao"
    chain = chain.rename(columns=rename_map)

    # Trata os dados
    for col in ["strike", "premio"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    chain["negocios"] = pd.to_numeric(chain.get("negocios", 0), errors="coerce").fillna(0).astype(int)
    chain = chain.dropna(subset=["strike", "premio"])
    chain = chain[chain["negocios"] >= min_liq]
    if chain.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente")

    # Calcula retorno
    hoje = datetime.now()
    venc = datetime.strptime(venc_str, "%Y-%m-%d")
    dias = (venc - hoje).days
    dias = max(dias, 1)

    chain["retorno_premio_pct"] = chain["premio"] / pm * 100.0
    chain["retorno_anualizado_pct"] = chain["retorno_premio_pct"] * (252 / dias)

    # Filtra strike
    if criterio == ">=PM":
        chain = chain[chain["strike"] >= pm]
    elif criterio == "PM":
        chain["diff"] = (chain["strike"] - pm).abs()
        chain = chain.sort_values(by="diff")

    if chain.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério")

    melhor = chain.sort_values(by="retorno_anualizado_pct", ascending=False).iloc[0]
    strike = float(melhor["strike"])
    premio = float(melhor["premio"])
    contratos = qty // 100

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
            "retorno_premio_pct": round(melhor["retorno_premio_pct"], 2),
            "retorno_anualizado_pct": round(melhor["retorno_anualizado_pct"], 2),
            "contratos_sugeridos": contratos
        },
        "cenarios": cenarios,
        "fonte_dados": "opcoes.net (scraping ~delay)"
    }
from fastapi import FastAPI, Response

app = FastAPI(title="robo-venda-coberta")

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.head("/")
def head_root():
    # evita 404 em HEAD /
    return Response(status_code=200)

@app.get("/health")
def health():
    return {"ok": True}
