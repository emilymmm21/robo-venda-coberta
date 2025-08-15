# app.py
import os
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

USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "1") == "1"

app = FastAPI(title="robo-venda-coberta", version="1.0.0")

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "Robo Venda Coberta API"}

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

class SuggestIn(BaseModel):
    ticker_subjacente: str
    preco_medio: float
    quantidade_acoes: int
    vencimento: str
    criterio: str = ">=PM"
    min_liquidez: int = 10
    debug: Optional[bool] = False

# ----------------- helpers -----------------
def _normalize(s: str) -> str:
    return unidecode(str(s)).lower().strip()

def _is_strike_col(name: str) -> bool:
    c = _normalize(name)
    return ("strike" in c) or ("exercicio" in c) or ("preco" in c and "exercicio" in c)

def _is_premio_col(name: str) -> bool:
    c = _normalize(name)
    return ("ult" in c) or ("ultimo" in c) or ("premio" in c) or ("preco" in c and "ult" in c)

def _is_negocios_col(name: str) -> bool:
    c = _normalize(name)
    return ("neg" in c) or ("volume" in c) or ("vol" in c)

def _is_venc_col(name: str) -> bool:
    return "venc" in _normalize(name)

def _is_ticker_opcao_col(name: str) -> bool:
    c = _normalize(name)
    return ("codigo" in c) or ("ticker" in c) or ("ativo" in c)

def _try_map_chain(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
    if not {"strike", "premio"}.issubset(out.columns):
        return None
    return out

# ----------------- fetchers -----------------
def _fetch(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
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

def _fetch_dynamic(url: str, timeout_ms: int = 15000) -> Optional[str]:
    """
    Usa Playwright/Chromium para renderizar a página (resolve iframes/JS).
    Retorna o HTML renderizado ou None se Playwright não estiver disponível.
    """
    if not USE_PLAYWRIGHT:
        return None
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            ctx = browser.new_context(locale="pt-BR")
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # Tenta esperar por pelo menos uma tabela visível
            try:
                page.locator("table").first.wait_for(timeout=7000, state="visible")
            except Exception:
                pass
            html = page.content()
            ctx.close()
            browser.close()
            return html
    except Exception:
        return None

def _extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    try:
        dfs = pd.read_html(html, decimal=",", thousands=".")
    except ValueError:
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

# ----------------- endpoint -----------------
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

    # 1) tenta HTML estático
    html = _fetch(base_url)
    all_dfs: List[pd.DataFrame] = _extract_tables_from_html(html)

    # 2) tenta iframes
    if not all_dfs:
        all_dfs.extend(_extract_tables_following_iframes(base_url, html))

    # 3) tenta Playwright (dinâmico)
    if not all_dfs:
        dyn_html = _fetch_dynamic(base_url)
        if dyn_html:
            all_dfs.extend(_extract_tables_from_html(dyn_html))

    if not all_dfs:
        raise HTTPException(status_code=404, detail="Nenhuma tabela encontrada")

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
                    "amostras_colunas": [list(map(str, d.columns)) for d in all_dfs[:5]],
                    "use_playwright": USE_PLAYWRIGHT,
                },
            }
        raise HTTPException(status_code=404, detail="Grade de opções não localizada")

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
        chain["negocios"] = 0

    chain = chain.dropna(subset=["strike", "premio"])
    chain = chain[chain["negocios"] >= min_liq]
    if chain.empty:
        raise HTTPException(status_code=404, detail="Sem liquidez suficiente")

    hoje = datetime.now()
    dias = max((venc - hoje).days, 1)
    chain["retorno_premio_pct"] = (chain["premio"] / pm) * 100.0
    chain["retorno_anualizado_pct"] = chain["retorno_premio_pct"] * (252 / dias)

    if criterio == ">=PM":
        chain = chain[chain["strike"] >= pm]
    elif criterio == "<=PM":
        chain = chain[chain["strike"] <= pm]
    elif criterio == "PM":
        chain["diff_pm"] = (chain["strike"] - pm).abs()
        chain = chain.sort_values(by="diff_pm")

    if chain.empty:
        raise HTTPException(status_code=404, detail="Nenhum strike compatível com o critério")

    chain = chain.sort_values(by="retorno_anualizado_pct", ascending=False)
    melhor = chain.iloc[0]

    strike = float(melhor["strike"])
    premio = float(melhor["premio"])
    negocios = int(melhor.get("negocios", 0))
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
            "dias_ate_vencimento": dias,
            "retorno_premio_pct": round(float(melhor["retorno_premio_pct"]), 2),
            "retorno_anualizado_pct": round(float(melhor["retorno_anualizado_pct"]), 2),
            "contratos_sugeridos": contratos,
        },
        "cenarios": cenarios,
        "fonte_dados": "opcoes.net.br (scraping; fallback headless)",
    }
    if debug:
        out["debug"] = {
            "colunas_grade": list(map(str, chain.columns)),
            "amostra_top5": chain.head(5).to_dict(orient="records"),
            "use_playwright": USE_PLAYWRIGHT,
        }
    return out
