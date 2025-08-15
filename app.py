# app.py
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

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
# Rotas utilitárias (ficam fora do OpenAPI por include_in_schema=False)
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
    criterio: str = ">=PM"   # ">=PM" | "<=PM" | "PM"
    min_liquidez: int = 10
    debug: Optional[bool] = False


# -----------------------------------------------------------------------------
# Helpers de normalização
# -----------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Remove acentos e normaliza minúsculas/espaços."""
    return unidecode(str(s)).lower().strip()

def _is_strike_col(name: str) -> bool:
    c = _normalize(name)
    # Exemplos: "Strike", "Preco de exercicio", "Preco exercicio"
    return ("strike" in c) or ("exercicio" in c) or ("preco" in c and "exercicio" in c)

def _is_premio_col(name: str) -> bool:
    c = _normalize(name)
    # Pode aparecer como "Ult.", "Ultimo", "Premio", "Preco ultimo"
    return ("ult" in c) or ("ultimo" in c) or ("premio" in c) or ("preco" in c and "ult" in c)

def _is_negocios_col(name: str) -> bool:
    c = _normalize(name)
    # "Negocios", "Qtd negocios", "Neg", ou até "volume" como proxy de liquidez
    return ("neg" in c) or ("volume" in c) or ("vol" in c)

def _is_venc_col(name: str) -> bool:
    c = _normalize(name)
    return "venc" in c

def _is_ticker_opcao_col(name: str) -> bool:
    c = _normalize(name)
    return ("codigo" in c) or ("ticker" in c) or ("ativo" in c)

def _try_map_chain(df: pd.DataFrame) -> pd.DataFrame | None:
    """Tenta mapear colunas essenciais (strike/premio). Retorna None se não servir."""
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


# -----------------------------------------------------------------------------
# Download e extração de tabelas (com iframes)
# -----------------------------------------------------------------------------
def _fetch(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Falha ao acessar {url}")
        return r.text
    except requests.RequestException:
        raise HTTPException(status_code=500, detail=f"Erro de rede ao acessar {url}")

def _extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    # 1) tenta direto
    try:
        dfs = pd.read_html(html, decimal=",", thousands=".")
    except ValueError:
        dfs = []

    # 2) tenta por cada <table>
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

def _extract_tables_following_iframes(base_url: str, html: str) -> List[pd.DataFrame]:
    soup = BeautifulSoup(html, "lxml")
    dfs: List[pd.DataFrame] = []
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src")
        if not src:
            continue
        child_url = urljoin(base_url, src)
        child_html = _fetch(child_url)
        child_dfs = _extract_tables_from_html(child_html)
        dfs.extend(child_dfs)
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
    debug = bool(data.debug)

    if qty < 100:
        raise HTTPException(status_code=400, detail="Quantidade de ações deve ser >= 100.")
    try:
        venc = datetime.strptime(venc_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de vencimento inválido. Use YYYY-MM-DD.")

    base_url = f"https://opcoes.net.br/opcoes/bovespa/{ticker}"
    html = _fetch(base_url)

    # tenta várias formas de extrair tabelas
    all_dfs: List[pd.DataFrame] = []
    all_dfs.extend(_extract_tables_from_html(html))
    if not all_dfs:
        all_dfs.extend(_extract_tables_following_iframes(base_url, html))

    if not all_dfs:
        raise HTTPException(status_code=404, detail="Nenhuma tabela encontrada")

    # procura a primeira tabela que tenha strike/premio plausíveis
    chain = None
    examined = 0
    for df in all_dfs:
        examined += 1
        df2 = _try_map_chain(df)
        if df2 is not None:
            chain = df2
            break

    if chain is None:
        if debug:
            return {
                "erro": "Grade de opções não localizada",
                "debug": {
                    "tabelas_encontradas": len(all_dfs),
                    "examinadas": examined,
                    "amostras_colunas": [list(map(str, d.columns)) for d in all_dfs[:3]],
                },
            }
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

    # ------- tratamento dos dados -------
    # remove possíveis "R$" e espaços finos
    def _clean_money(x):
        if isinstance(x, str):
            x = x.replace("R$", "").replace("\xa0", " ").strip()
            x = x.replace(".", "").replace(",", ".") if x.count(",") == 1 else x
        return x

    chain["strike"] = pd.to_numeric(chain["strike"].map(_clean_money), errors="coerce")
    chain["premio"] = pd.to_numeric(chain["premio"].map(_clean_money), errors="coerce")

    if "negocios" in chain.columns:
        chain["negocios"] = pd.to_numeric(chain["negocios"], errors="coerce").fillna(0).astype(int)
    else:
        # como fallback, se existir "volume" mapeado como "negocios" acima, já cobre
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
    # senão, mantém tudo

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

    out = {
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
        "fonte_dados": "opcoes.net.br (scraping)",
    }

    if debug:
        out["debug"] = {
            "tabelas_totais": len(all_dfs),
            "colunas_grade": list(map(str, chain.columns)),
            "amostra_top5": chain.head(5).to_dict(orient="records"),
        }
    return out

