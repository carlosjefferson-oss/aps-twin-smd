import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import unicodedata
import re
import base64
import html as html_lib
from urllib.parse import urlparse, parse_qs

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# APS-Twin SMD v1.9.0.7.6.5.4.3.2
# Digital Twin Gerencial da Atenção Primária à Saúde
# Versão com gráficos refinados + Digital Twin ciberfísico + perfil população em linguagem cidadã
# =========================================================

st.set_page_config(
    page_title="APS-Twin SMD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CAMINHOS DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).parent

# Pastas com fallback para funcionar tanto no computador quanto no Streamlit Cloud.
# O projeto pode estar com nomes em português (dados/ativos) ou inglês (data/assets).
DATA_DIR = BASE_DIR / "dados"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"

ASSETS_DIR = BASE_DIR / "ativos"
if not ASSETS_DIR.exists():
    ASSETS_DIR = BASE_DIR / "assets"

# Bandeiras exibidas no cabeçalho institucional
FLAG_PE_PATH = ASSETS_DIR / "bandeira_pernambuco.png"
FLAG_SALGUEIRO_PATH = ASSETS_DIR / "bandeira_salgueiro.png"

UNIVASF_LOGO_CANDIDATES = [
    ASSETS_DIR / "logo_univasf.png",
    ASSETS_DIR / "univasf_logo.png",
    ASSETS_DIR / "univasf.png",
]
UFBA_LOGO_CANDIDATES = [
    ASSETS_DIR / "logo_ufba.png",
    ASSETS_DIR / "ufba_logo.png",
    ASSETS_DIR / "ufba.png",
]

UNIVASF_LOGO_PATH = next((p for p in UNIVASF_LOGO_CANDIDATES if p.exists()), None)
UFBA_LOGO_PATH = next((p for p in UFBA_LOGO_CANDIDATES if p.exists()), None)

LOGO_CANDIDATOS = [
    ASSETS_DIR / "aps_twin_login_card_900x420.png",
    ASSETS_DIR / "aps_twin_login_hero_1600x900.png",
    ASSETS_DIR / "logo_aps_twin.png",
    ASSETS_DIR / "aps_twin_logo_square_600x600.png",
]

LOGO_PATH = next((p for p in LOGO_CANDIDATOS if p.exists()), None)


# =========================================================
# USUÁRIOS DO PROTÓTIPO
# =========================================================

USUARIOS = {
    "admin": {
        "senha": "aps123",
        "nome": "Administrador do Sistema",
        "perfil": "adm",
        "ubs": "Todas",
        "descricao": "Acesso administrativo completo"
    },
    "aluno": {
        "senha": "aluno123",
        "nome": "Aluno/Pesquisador",
        "perfil": "aluno",
        "ubs": "Todas",
        "descricao": "Acesso acadêmico completo, equivalente ao ADM"
    },
    "secretario": {
        "senha": "sec123",
        "nome": "Secretaria Municipal de Saúde",
        "perfil": "secretario",
        "ubs": "Todas",
        "descricao": "Acesso executivo municipal sem API e sem filtros administrativos"
    },
    "prefeito": {
        "senha": "pref123",
        "nome": "Prefeito Municipal",
        "perfil": "prefeito",
        "ubs": "Todas",
        "descricao": "Acesso estratégico municipal para tomada de decisão"
    },
    "populacao": {
        "senha": "apspublica",
        "nome": "População Geral",
        "perfil": "populacao",
        "ubs": "APS Municipal",
        "descricao": "Acesso público aos indicadores municipais"
    },
    "gerente_cohab1": {
        "senha": "ubs123",
        "nome": "Gerente UBS Cohab 1",
        "perfil": "gerente",
        "ubs": "Cohab 1",
        "descricao": "Acesso restrito à UBS Cohab 1"
    },
    "gerente_cohab2": {
        "senha": "ubs123",
        "nome": "Gerente UBS Cohab 2",
        "perfil": "gerente",
        "ubs": "Cohab 2",
        "descricao": "Acesso restrito à UBS Cohab 2"
    },
    "gerente_fatima": {
        "senha": "ubs123",
        "nome": "Gerente UBS Fátima",
        "perfil": "gerente",
        "ubs": "Fátima",
        "descricao": "Acesso restrito à UBS Fátima"
    },
    "gerente_mariapanta": {
        "senha": "ubs123",
        "nome": "Gerente UBS Maria Panta",
        "perfil": "gerente",
        "ubs": "Maria Panta",
        "descricao": "Acesso restrito à UBS Maria Panta"
    },
    "gerente_pauferro": {
        "senha": "ubs123",
        "nome": "Gerente UBS Pau Ferro",
        "perfil": "gerente",
        "ubs": "Pau Ferro",
        "descricao": "Acesso restrito à UBS Pau Ferro"
    },
    "gerente_umas": {
        "senha": "ubs123",
        "nome": "Gerente UBS Umãs",
        "perfil": "gerente",
        "ubs": "Umãs",
        "descricao": "Acesso restrito à UBS Umãs"
    },
    "gerente_santamargarida": {
        "senha": "ubs123",
        "nome": "Gerente UBS Santa Margarida",
        "perfil": "gerente",
        "ubs": "Santa Margarida",
        "descricao": "Acesso restrito à UBS Santa Margarida"
    }
}


# =========================================================
# PALETA E ESTILO GLOBAL
# =========================================================

COLORS = {
    "navy": "#0B2545",
    "blue": "#1976D2",
    "blue2": "#42A5F5",
    "teal": "#0097A7",
    "cyan": "#00B8D4",
    "green": "#2E7D32",
    "green2": "#66BB6A",
    "amber": "#F9A825",
    "orange": "#FB8C00",
    "red": "#C62828",
    "red2": "#EF5350",
    "gray": "#667085",
    "light": "#F5F8FC",
    "card": "#FFFFFF",
    "line": "#DDE7F2"
}

STATUS_COLORS = {
    "Adequado": COLORS["green"],
    "Atenção": COLORS["amber"],
    "Crítico": COLORS["red"],
    "Sem dado": COLORS["gray"]
}

MODEL_COLORS = {
    "Regressão Linear": COLORS["blue"],
    "Random Forest": COLORS["green"],
    "Gradient Boosting": COLORS["amber"],
    "MLP Neural Network": COLORS["teal"]
}

st.markdown(
    """
    <style>
        :root {
            --navy: #0B2545;
            --blue: #1976D2;
            --teal: #0097A7;
            --green: #2E7D32;
            --amber: #F9A825;
            --red: #C62828;
            --bg: #F5F8FC;
            --card: #FFFFFF;
            --muted: #667085;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #F7FAFE 0%, #EEF5FC 100%);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }

        section[data-testid="stSidebar"] {
            background: #F0F5FA;
            border-right: 1px solid #DFE8F3;
        }

        .app-header {
            max-width: 100%;
            box-sizing: border-box;
            background: radial-gradient(circle at right top, rgba(0,184,212,.35), transparent 28%),
                        linear-gradient(135deg, #0B2545 0%, #1976D2 58%, #0097A7 100%);
            padding: 30px 32px;
            border-radius: 30px;
            color: white;
            box-shadow: 0 16px 36px rgba(11, 37, 69, 0.25);
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }

        .app-title {
            font-size: clamp(34px, 4vw, 58px);
            font-weight: 950;
            margin: 0;
            letter-spacing: -1px;
        }

        .app-subtitle {
            font-size: clamp(15px, 1.8vw, 21px);
            opacity: .96;
            margin-top: 6px;
            max-width: 980px;
        }

        .app-tags {
            margin-top: 15px;
            font-size: 14px;
            opacity: .95;
        }

        .login-shell {
            max-width: 1020px;
            margin: 34px auto 0 auto;
            display: grid;
            grid-template-columns: 1.25fr .9fr;
            gap: 24px;
            align-items: stretch;
        }

        .login-visual {
            background: #FFFFFF;
            border-radius: 30px;
            padding: 16px;
            box-shadow: 0 18px 46px rgba(11,37,69,0.14);
            border: 1px solid #E6EEF7;
            overflow: hidden;
        }

        .login-panel {
            background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFE 100%);
            border-radius: 30px;
            padding: 34px 32px;
            box-shadow: 0 18px 46px rgba(11,37,69,0.14);
            border-top: 8px solid #1976D2;
            border-left: 1px solid #E6EEF7;
            border-right: 1px solid #E6EEF7;
            border-bottom: 1px solid #E6EEF7;
        }

        .login-title {
            font-size: 34px;
            font-weight: 950;
            color: var(--navy);
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }

        .login-subtitle {
            font-size: 15px;
            color: var(--muted);
            margin-bottom: 18px;
            line-height: 1.45;
        }

        .login-mini {
            font-size: 12px;
            color: var(--muted);
            background: #EEF6FF;
            border-left: 4px solid #1976D2;
            padding: 10px 12px;
            border-radius: 12px;
            margin-top: 14px;
        }

        .ubs-card {
            border-radius: 20px;
            padding: 15px 16px;
            box-shadow: 0 12px 28px rgba(11,37,69,0.10);
            min-height: 140px;
            margin-bottom: 6px;
            border: 1px solid #E6EEF7;
            background: #FFFFFF;
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .ubs-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(11,37,69,0.14);
        }

        .ubs-ok {
            border-left: 8px solid #2E7D32;
            background: linear-gradient(135deg, #E8F5E9, #FFFFFF);
        }

        .ubs-alerta {
            border-left: 8px solid #F9A825;
            background: linear-gradient(135deg, #FFF8E1, #FFFFFF);
        }

        .ubs-critico {
            border-left: 8px solid #C62828;
            background: linear-gradient(135deg, #FFEBEE, #FFFFFF);
        }

        .ubs-icon {
            font-size: 16px;
            margin-bottom: 4px;
            line-height: 1;
        }

        
        .status-dot {
            width: 12px;
            height: 12px;
            min-width: 12px;
            border-radius: 50%;
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.55), 0 4px 10px rgba(11,37,69,0.16);
            margin-top: 3px;
        }

        .status-dot-ok {
            background: linear-gradient(135deg, #66BB6A, #2E7D32);
        }

        .status-dot-alerta {
            background: linear-gradient(135deg, #FFD166, #F9A825);
        }

        .status-dot-critico {
            background: linear-gradient(135deg, #EF5350, #C62828);
        }

        
        .ubs-card-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 4px;
        }

        .ubs-title-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }

        .ubs-mini-icon {
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 10px;
            background: rgba(25,118,210,0.10);
            color: #1976D2;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.65);
        }

        .ubs-title {
            font-size: 18px;
            font-weight: 900;
            color: #0B2545;
            line-height: 1.15;
        }

        .ubs-score {
            font-size: 30px;
            font-weight: 950;
            color: #0B2545;
            margin-top: 4px;
        }

        .ubs-status {
            font-size: 12px;
            font-weight: 850;
            color: #344054;
        }

        .ubs-small {
            font-size: 10.5px;
            color: #667085;
            margin-top: 6px;
        }

        .section-title {
            font-size: 28px;
            font-weight: 900;
            color: #0B2545;
            margin: 10px 0 16px 0;
        }

        .section-caption {
            color: #667085;
            font-size: 14px;
            margin-top: -8px;
            margin-bottom: 18px;
        }

        .dt-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFE 100%);
            border: 1px solid #E3ECF6;
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 24px rgba(11,37,69,0.08);
            min-height: 150px;
        }

        .dt-title {
            color: #0B2545;
            font-size: 16px;
            font-weight: 900;
            margin-bottom: 6px;
        }

        .dt-value {
            color: #1976D2;
            font-size: 34px;
            font-weight: 950;
            line-height: 1;
        }

        .dt-note {
            color: #667085;
            font-size: 12px;
            margin-top: 8px;
        }

        .kpi-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
            border: 1px solid #E5EDF6;
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(11,37,69,0.08);
            min-height: 126px;
            position: relative;
            overflow: hidden;
        }

        .kpi-card:before {
            content: "";
            position: absolute;
            right: -26px;
            top: -26px;
            width: 88px;
            height: 88px;
            background: rgba(25,118,210,0.08);
            border-radius: 50%;
        }

        .kpi-icon {
            font-size: 22px;
            margin-bottom: 5px;
        }

        .kpi-label {
            color: #667085;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .045em;
        }

        .kpi-value {
            color: #0B2545;
            font-size: 34px;
            font-weight: 950;
            line-height: 1.05;
            margin-top: 6px;
        }

        .kpi-note {
            color: #98A2B3;
            font-size: 11px;
            margin-top: 4px;
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5EDF6;
            border-radius: 18px;
            padding: 15px 16px;
            box-shadow: 0 5px 16px rgba(11,37,69,0.06);
        }

        .executive-strip {
            background: linear-gradient(135deg, rgba(11,37,69,0.98), rgba(25,118,210,0.95), rgba(0,151,167,0.96));
            color: white;
            padding: 22px 26px;
            border-radius: 28px;
            box-shadow: 0 18px 42px rgba(11,37,69,0.18);
            margin: 8px 0 18px 0;
            border: 1px solid rgba(255,255,255,0.18);
        }

        .executive-strip-title {
            font-size: 28px;
            font-weight: 950;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }

        .executive-strip-subtitle {
            font-size: 14px;
            line-height: 1.55;
            opacity: .94;
            max-width: 1000px;
        }

        .context-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .context-pill {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: 700;
            color: white;
        }

        .institutional-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
            border: 1px solid #E5EDF6;
            border-radius: 20px;
            padding: 15px 16px;
            box-shadow: 0 12px 30px rgba(11,37,69,0.08);
            margin-bottom: 16px;
        }

        .decision-card {
            background: #FFFFFF;
            border: 1px solid #E5EDF6;
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 24px rgba(11,37,69,0.07);
            min-height: 150px;
            border-top: 5px solid #1976D2;
        }

        .decision-title {
            font-size: 16px;
            font-weight: 900;
            color: #0B2545;
            margin-bottom: 8px;
        }

        .decision-text {
            color: #667085;
            font-size: 13px;
            line-height: 1.5;
        }

        .priority-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 800;
            color: white;
            background: #1976D2;
            margin-top: 8px;
        }

        .priority-critical { background: #C62828; }
        .priority-high { background: #FB8C00; }
        .priority-medium { background: #F9A825; color: #0B2545; }
        .priority-low { background: #2E7D32; }

        .ubs-panel-title {
            color:#0B2545;
            font-weight:950;
            font-size:24px;
            margin-bottom:4px;
        }

        .ubs-panel-subtitle {
            color:#667085;
            font-size:13px;
            margin-bottom:16px;
        }

        .sidebar-brand-box {
            background: linear-gradient(135deg, #0B2545, #1976D2, #0097A7);
            color: white;
            border-radius: 20px;
            padding: 16px;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(11,37,69,0.18);
        }

        .sidebar-brand-title {
            font-size: 21px;
            font-weight: 950;
            margin-bottom: 3px;
        }

        .sidebar-brand-subtitle {
            font-size: 12px;
            opacity: .92;
            line-height: 1.35;
        }

        .sidebar-brand-version {
            font-size: 11px;
            font-weight: 800;
            margin-top: 8px;
            background: rgba(255,255,255,0.14);
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
        }

        
        div[data-testid="stButton"] button {
            border-radius: 12px;
            min-height: 36px;
            font-size: 13px;
            font-weight: 750;
        }

        div[data-testid="stTabs"] {
            background: rgba(255,255,255,0.92);
            border-radius: 20px;
            padding: 10px 10px 0 10px;
            border: 1px solid #E5EDF6;
            box-shadow: 0 10px 24px rgba(11,37,69,0.06);
        }

        div[data-testid="stTabs"] button {
            font-weight: 850;
            border-radius: 16px 16px 0 0;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            margin-right: 4px;
            color: #475467;
            transition: all .15s ease-in-out;
        }

        div[data-testid="stTabs"] button:hover {
            background: #F5F9FF;
            color: #0B2545;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(180deg, #FFF 0%, #F8FBFF 100%);
            color: #1976D2 !important;
            border-bottom: 3px solid #1976D2 !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .login-shell {
                grid-template-columns: 1fr;
                margin-top: 10px;
            }
            .login-panel {
                padding: 26px 22px;
            }
            .app-header {
                padding: 22px;
                border-radius: 22px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOGIN E CONTROLE DE ACESSO
# =========================================================




def image_to_base64(path):
    """Converte imagem local para base64 para uso seguro em HTML no Streamlit."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None




def render_header_principal():
    """
    Cabeçalho institucional limpo.
    A navegação foi removida do card azul para evitar repetição visual.
    """
    flag_pe_b64 = image_to_base64(FLAG_PE_PATH)
    flag_salgueiro_b64 = image_to_base64(FLAG_SALGUEIRO_PATH)
    univasf_b64 = image_to_base64(UNIVASF_LOGO_PATH) if UNIVASF_LOGO_PATH else None
    ufba_b64 = image_to_base64(UFBA_LOGO_PATH) if UFBA_LOGO_PATH else None

    salgueiro_html = (
        f'<div class="identity-card"><img src="data:image/png;base64,{flag_salgueiro_b64}" class="identity-img" />'
        f'<div class="identity-label">Salgueiro-PE</div></div>'
        if flag_salgueiro_b64
        else '<div class="identity-card"><div class="identity-placeholder">Salgueiro-PE</div></div>'
    )

    pe_html = (
        f'<div class="identity-card"><img src="data:image/png;base64,{flag_pe_b64}" class="identity-img" />'
        f'<div class="identity-label">Pernambuco</div></div>'
        if flag_pe_b64
        else '<div class="identity-card"><div class="identity-placeholder">Pernambuco</div></div>'
    )

    univasf_html = (
        f'<div class="partner-card"><img src="data:image/png;base64,{univasf_b64}" class="partner-img" />'
        f'<div class="partner-label">UNIVASF</div></div>'
        if univasf_b64
        else '<div class="partner-card"><div class="partner-placeholder">UNIVASF</div><div class="partner-label">UNIVASF</div></div>'
    )

    ufba_html = (
        f'<div class="partner-card"><img src="data:image/png;base64,{ufba_b64}" class="partner-img" />'
        f'<div class="partner-label">UFBA</div></div>'
        if ufba_b64
        else '<div class="partner-card"><div class="partner-placeholder">UFBA</div><div class="partner-label">UFBA</div></div>'
    )

    html = f"""
<style>
.hero-main {{
    width: 100%;
    box-sizing: border-box;
    background:
        radial-gradient(circle at 92% 10%, rgba(255,255,255,0.22), transparent 26%),
        linear-gradient(135deg, #0B2545 0%, #1976D2 56%, #0097A7 100%);
    border-radius: 30px;
    padding: 34px 36px;
    color: #FFFFFF !important;
    box-shadow: 0 16px 38px rgba(11,37,69,0.22);
    margin: 8px 0 18px 0;
    border: 1px solid rgba(255,255,255,0.22);
    overflow: hidden;
}}
.hero-grid-html {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    gap: 34px;
    align-items: center;
}}
.hero-title-html {{
    color: #FFFFFF !important;
    font-size: clamp(40px, 4.5vw, 62px);
    font-weight: 950;
    line-height: 1.0;
    letter-spacing: -1.2px;
    margin: 0 0 14px 0;
}}
.hero-subtitle-html {{
    color: #FFFFFF !important;
    font-size: clamp(16px, 1.6vw, 21px);
    font-weight: 800;
    line-height: 1.45;
    margin: 0 0 14px 0;
}}
.hero-line-html {{
    color: rgba(255,255,255,0.95) !important;
    font-size: 14px;
    line-height: 1.6;
    font-weight: 650;
    margin: 0;
    max-width: 940px;
}}
.hero-mini-signature {{
    display: inline-flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}}
.hero-chip {{
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 7px 12px;
    color: #FFFFFF !important;
    font-size: 12px;
    font-weight: 850;
    letter-spacing: .01em;
}}
.identity-panel {{
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 22px;
    padding: 14px;
}}
.identity-section-title {{
    color: #FFFFFF !important;
    font-size: 12px;
    font-weight: 950;
    text-align: center;
    margin: 2px 0 8px 0;
}}
.identity-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
}}
.identity-card, .partner-card {{
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(255,255,255,0.70);
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 10px 22px rgba(0,0,0,0.12);
    text-align: center;
}}
.identity-img {{
    width: 100%;
    height: 68px;
    object-fit: contain;
    border-radius: 10px;
    background: #FFFFFF;
    display: block;
}}
.partner-img {{
    width: 100%;
    height: 34px;
    object-fit: contain;
    border-radius: 8px;
    background: #FFFFFF;
    display: block;
}}
.identity-label, .partner-label {{
    color: #0B2545 !important;
    font-size: 10px;
    font-weight: 950;
    margin-top: 5px;
    letter-spacing: .02em;
}}
.partner-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}}
.identity-placeholder, .partner-placeholder {{
    color: #0B2545 !important;
    font-size: 16px;
    font-weight: 900;
    min-height: 34px;
    display:flex;
    align-items:center;
    justify-content:center;
}}
@media (max-width: 980px) {{
    .hero-main {{
        padding: 26px 24px;
    }}
    .hero-grid-html {{
        grid-template-columns: 1fr;
        gap: 22px;
    }}
    .identity-panel {{
        max-width: 380px;
    }}
}}
</style>
<div class="hero-main">
  <div class="hero-grid-html">
    <div class="hero-text-block">
      <div class="hero-title-html">APS-Twin SMD</div>
      <div class="hero-subtitle-html">Digital Twin Gerencial da Atenção Primária à Saúde | Salgueiro-PE</div>
      <div class="hero-line-html">Solução desenvolvida por pesquisadores da UNIVASF e da UFBA, integrando Sistema de Medição de Desempenho, Machine Learning, API em tempo real e apoio à decisão para a Atenção Primária à Saúde.</div>
      <div class="hero-mini-signature">
        <span class="hero-chip">Atenção Primária</span>
        <span class="hero-chip">SMD</span>
        <span class="hero-chip">Machine Learning</span>
        <span class="hero-chip">Digital Twin</span>
      </div>
    </div>
    <div class="identity-panel">
      <div class="identity-section-title">Identidade territorial</div>
      <div class="identity-grid">{salgueiro_html}{pe_html}</div>
      <div class="identity-section-title">Instituições parceiras</div>
      <div class="partner-grid">{univasf_html}{ufba_html}</div>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def tela_login():
    """
    Tela de login institucional com termo LGPD obrigatório.
    """
    st.markdown(
        """
        <style>
            .login-page-note {
                text-align:center; color:#667085; font-size:15px;
                margin-top: 12px; margin-bottom: 18px;
            }
            .login-title-clean {
                text-align:center; color:#0B2545; font-size:44px;
                font-weight:950; line-height:1.05; margin-top:8px;
                margin-bottom:8px; letter-spacing:-1px;
            }
            .login-subtitle-clean {
                text-align:center; color:#667085; font-size:16px;
                line-height:1.60; margin-bottom:20px;
            }
            .login-helper-clean {
                background:#F7FAFE; border:1px solid #D9EAFB;
                border-left:4px solid #1976D2; color:#475467;
                border-radius:16px; padding:12px 14px; font-size:12px;
                line-height:1.55; margin-top:14px;
            }
            .lgpd-box {
                background:#FFFDF5; border:1px solid #FDE7A1;
                border-left:4px solid #F9A825; color:#344054;
                border-radius:16px; padding:12px 14px; font-size:11.5px;
                line-height:1.55; margin:12px 0 8px 0; text-align:left;
            }
            .login-footer-clean {
                text-align:center; color:#98A2B3; font-size:12px; margin-top:14px;
            }
            .login-institutional-badge {
                text-align:center; font-size:12px; font-weight:800;
                color:#1976D2; background:#EEF6FF; border:1px solid #D9EAFB;
                padding:6px 10px; border-radius:999px; width:fit-content;
                margin: 0 auto 12px auto;
            }
            .login-logo-wrap {
                display:flex; justify-content:center; align-items:center;
                margin: 8px 0 16px 0; width: 100%;
            }
            .login-logo-img {
                display:block; margin:0 auto; width: 190px; max-width: 58vw;
                border-radius: 18px; box-shadow: 0 10px 24px rgba(11,37,69,0.12);
            }
            .login-logo-emoji {text-align:center; font-size:88px; line-height:1; margin: 6px 0 10px 0;}
            div[data-testid="stTextInput"] input {
                border-radius: 14px !important; border: 1px solid #D0D5DD !important;
                background: #FFFFFF !important;
            }
            div[data-testid="stButton"] button[kind="primary"] {
                border-radius: 14px !important; height: 48px; font-weight: 800;
                background: linear-gradient(135deg, #1976D2, #0097A7) !important;
                border: none !important;
            }
            @media (max-width: 900px) {
                .login-title-clean {font-size: 34px;}
                .login-subtitle-clean {font-size: 14px;}
                .login-logo-img {width: 210px; max-width: 62vw;}
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-page-note">Plataforma analítica e gerencial da Atenção Primária à Saúde</div>',
        unsafe_allow_html=True
    )

    esq, centro, dir_ = st.columns([1.1, 1.2, 1.1])

    with centro:
        with st.container(border=True):
            st.markdown('<div class="login-institutional-badge">APS-Twin SMD • v1.9 institucional</div>', unsafe_allow_html=True)

            logo_pequena = ASSETS_DIR / "aps_twin_logo_square_600x600.png"
            caminho_logo = logo_pequena if logo_pequena.exists() else LOGO_PATH

            if caminho_logo is not None:
                logo_b64 = image_to_base64(caminho_logo)
                if logo_b64:
                    st.markdown(
                        f'<div class="login-logo-wrap"><img src="data:image/png;base64,{logo_b64}" class="login-logo-img"></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("<div class='login-logo-emoji'>🏥</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='login-logo-emoji'>🏥</div>", unsafe_allow_html=True)

            st.markdown('<div class="login-title-clean">APS-Twin SMD</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="login-subtitle-clean">
                    Digital Twin Gerencial da Atenção Primária à Saúde.<br>
                    Monitoramento, predição, simulação e decisão para a gestão da APS.
                </div>
                """,
                unsafe_allow_html=True
            )

            usuario = st.text_input("Usuário", placeholder="Ex.: admin", key="login_usuario")
            senha = st.text_input("Senha", type="password", placeholder="Digite sua senha", key="login_senha")

            st.markdown(
                """
                <div class="lgpd-box">
                    <b>Termo de ciência e responsabilidade — LGPD</b><br>
                    Declaro estar ciente de que as informações acessadas neste sistema possuem finalidade acadêmica,
                    gerencial e demonstrativa para apoio à decisão na Atenção Primária à Saúde. Comprometo-me a não
                    divulgar, copiar, compartilhar ou utilizar indevidamente dados, relatórios, indicadores ou informações
                    eventualmente sensíveis ou restritas. O uso do sistema deve observar a Lei Geral de Proteção de Dados
                    Pessoais — LGPD (Lei nº 13.709/2018), especialmente os princípios de confidencialidade, segurança,
                    finalidade, necessidade e responsabilidade no tratamento das informações.
                </div>
                """,
                unsafe_allow_html=True
            )

            aceite_lgpd = st.checkbox(
                "Li e estou ciente das condições de uso, sigilo e responsabilidade sobre as informações.",
                key="aceite_lgpd_login"
            )

            entrar = st.button("Entrar no sistema", type="primary", use_container_width=True)

            if entrar:
                usuario_limpo = usuario.strip().lower()
                if not aceite_lgpd:
                    st.warning("Para acessar, é necessário marcar o termo de ciência e responsabilidade.")
                elif usuario_limpo in USUARIOS and senha == USUARIOS[usuario_limpo]["senha"]:
                    st.session_state["logado"] = True
                    st.session_state["usuario"] = usuario_limpo
                    st.session_state["nome_usuario"] = USUARIOS[usuario_limpo]["nome"]
                    st.session_state["perfil_usuario"] = USUARIOS[usuario_limpo]["perfil"]
                    st.session_state["ubs_usuario"] = normalizar_nome_ubs(USUARIOS[usuario_limpo]["ubs"])
                    st.session_state["aceite_lgpd"] = True
                    st.success(f"Bem-vindo, {USUARIOS[usuario_limpo]['nome']}!")
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")

            st.markdown(
                """
                <div class="login-helper-clean">
                    <b>Acessos de teste</b><br>
                    ADM: <b>admin</b> | Senha: <b>aps123</b><br>
                    Aluno: <b>aluno</b> | Senha: <b>aluno123</b><br>
                    Secretaria: <b>secretario</b> | Senha: <b>sec123</b><br>
                    Prefeito: <b>prefeito</b> | Senha: <b>pref123</b><br>
                    Gerente UBS: <b>gerente_cohab1</b> | Senha: <b>ubs123</b><br>
                    População geral: <b>populacao</b> | Senha: <b>apspublica</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="login-footer-clean">APS-Twin SMD • Plataforma web para monitoramento, predição e apoio à decisão</div>', unsafe_allow_html=True)


def verificar_login():
    if "logado" not in st.session_state:
        st.session_state["logado"] = False

    if not st.session_state["logado"]:
        tela_login()
        st.stop()


def botao_logout():
    col1, col2 = st.columns([5, 1])

    with col1:
        st.caption(
            f"Usuário conectado: {st.session_state.get('nome_usuario', 'Não identificado')} | "
            f"Perfil: {st.session_state.get('perfil_usuario', '-')}"
        )

    with col2:
        if st.button("Sair", use_container_width=True):
            for chave in [
                "logado", "usuario", "nome_usuario", "perfil_usuario",
                "ubs_usuario", "ubs_selecionada_card"
            ]:
                if chave in st.session_state:
                    del st.session_state[chave]
            st.rerun()


# =========================================================
# FUNÇÕES DE APOIO
# =========================================================

def limpar_nome_coluna(coluna):
    return (
        str(coluna)
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        .strip()
    )


def chave_texto(texto):
    return (
        str(texto)
        .lower()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        .strip()
    )

def normalizar_nome_ubs(nome):
    """Padroniza nomes de UBS para evitar duplicidades entre bases locais e API."""
    if pd.isna(nome):
        return nome

    txt = str(nome).strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("_", " ")
    txt = txt.replace("-", " ")
    txt = re.sub(r"\s+", " ", txt).strip()

    canon = unicodedata.normalize("NFKD", txt)
    canon = canon.encode("ascii", "ignore").decode("utf-8").lower()
    canon = re.sub(r"[^a-z0-9 ]", "", canon)
    canon = re.sub(r"\s+", " ", canon).strip()

    mapa = {
        "pau ferro": "Pau Ferro",
        "cohab 1": "Cohab 1",
        "cohab1": "Cohab 1",
        "cohab 2": "Cohab 2",
        "cohab2": "Cohab 2",
        "maria panta": "Maria Panta",
        "fatima": "Fátima",
        "umas": "Umãs",
        "umas i": "Umãs",
        "umas ii": "Umãs",
        "aps municipal": "APS Municipal",
        "todas": "Todas",
    }

    if canon in mapa:
        return mapa[canon]

    if "aps municipal" in canon:
        return "APS Municipal"

    txt = " ".join([p.capitalize() if p else p for p in txt.split(" ")])
    txt = txt.replace(" Ubs ", " UBS ").replace(" Ubs", " UBS")
    txt = txt.replace("Psf", "PSF")
    return txt

def perfil_atual():
    return st.session_state.get("perfil_usuario", "")


def usuario_pode_admin():
    return perfil_atual() in ["adm", "aluno"]


def usuario_pode_api():
    return perfil_atual() in ["adm", "aluno"]


def usuario_pode_sidebar():
    return perfil_atual() in ["adm", "aluno"]


def usuario_e_gerente():
    return perfil_atual() == "gerente"


def usuario_e_publico():
    return perfil_atual() == "populacao"


def usuario_e_prefeito():
    return perfil_atual() == "prefeito"


def usuario_enxerga_tudo():
    return perfil_atual() in ["adm", "aluno", "secretario", "prefeito"]


def aplicar_restricao_de_perfil(df):
    if df is None or df.empty or "ubs" not in df.columns:
        return df

    df = df.copy()
    df["ubs"] = df["ubs"].apply(normalizar_nome_ubs)
    perfil = perfil_atual()
    ubs_usuario = normalizar_nome_ubs(st.session_state.get("ubs_usuario", "Todas"))

    if perfil == "gerente":
        return df[df["ubs"].apply(normalizar_nome_ubs) == ubs_usuario].copy()

    if perfil == "populacao":
        # Para o perfil público, mantemos os registros por UBS para permitir
        # tradução cidadã do tipo: UBS crítica, possível demora, procurar
        # agendamento etc. O sigilo é protegido na camada visual: não mostramos
        # bases técnicas, API, modelos, parâmetros internos ou dados sensíveis.
        return df.copy()

    return df




def detectar_arquivos():
    """
    Detecta automaticamente os arquivos Excel das bases municipal e por UBS.
    Aceita nomes flexíveis, desde que estejam na pasta DATA_DIR.
    """
    arquivos = list(DATA_DIR.glob("*.xlsx"))

    arquivo_municipal = None
    arquivo_ubs = None

    for arquivo in arquivos:
        nome = arquivo.name.lower()

        # Prioridade por nome do arquivo
        if "ubs" in nome:
            arquivo_ubs = arquivo.name
            continue

        if "aps" in nome or "municipal" in nome or "indicadores" in nome:
            if arquivo_municipal is None:
                arquivo_municipal = arquivo.name

    # Segunda tentativa por leitura das colunas
    for arquivo in arquivos:
        if arquivo_municipal is not None and arquivo_ubs is not None:
            break

        try:
            df_temp = pd.read_excel(arquivo, nrows=3)
            colunas = " ".join([str(c).lower() for c in df_temp.columns])

            if "ubs" in colunas:
                if arquivo_ubs is None:
                    arquivo_ubs = arquivo.name
            else:
                if arquivo_municipal is None:
                    arquivo_municipal = arquivo.name

        except Exception:
            pass

    return arquivo_municipal, arquivo_ubs

@st.cache_data
def carregar_excel(nome_arquivo):
    caminho = DATA_DIR / nome_arquivo
    df = pd.read_excel(caminho)
    df.columns = [limpar_nome_coluna(c) for c in df.columns]
    return df


@st.cache_data
def carregar_dicionario():
    caminho = DATA_DIR / "dictionary_indicators.csv"

    if not caminho.exists():
        return None

    dic = pd.read_csv(caminho)
    dic.columns = [limpar_nome_coluna(c) for c in dic.columns]

    dic["chave_indicador"] = dic["nome_original"].apply(chave_texto)
    dic["meta"] = pd.to_numeric(dic["meta"], errors="coerce")

    return dic


def transformar_base_municipal(df):
    df = df.copy()
    df.columns = [limpar_nome_coluna(c) for c in df.columns]

    col_data = df.columns[0]
    col_mes = df.columns[1]

    df = df.rename(columns={
        col_data: "data",
        col_mes: "mes"
    })

    colunas_id = ["data", "mes"]
    colunas_indicadores = [c for c in df.columns if c not in colunas_id]

    df_long = df.melt(
        id_vars=colunas_id,
        value_vars=colunas_indicadores,
        var_name="indicador_original",
        value_name="valor"
    )

    df_long["ubs"] = "APS Municipal"
    df_long["fonte"] = "Indicadores municipais"

    return df_long


def transformar_base_ubs(df):
    df = df.copy()
    df.columns = [limpar_nome_coluna(c) for c in df.columns]

    coluna_mes = None
    coluna_ubs = None

    for col in df.columns:
        col_lower = col.lower()

        if "mês" in col_lower or "mes" in col_lower:
            coluna_mes = col

        if "ubs" in col_lower:
            coluna_ubs = col

    if coluna_mes is None:
        coluna_mes = df.columns[0]

    if coluna_ubs is None:
        coluna_ubs = df.columns[1]

    df = df.rename(columns={
        coluna_mes: "mes",
        coluna_ubs: "ubs"
    })

    colunas_id = ["mes", "ubs"]

    colunas_excluir = []
    termos_excluir = [
        "carimbo", "timestamp", "data/hora", "data e hora", "hora da resposta",
        "endereço de e-mail", "endereco de e-mail", "e-mail", "email",
        "nome do respondente", "respondente"
    ]

    for c in df.columns:
        c_lower = str(c).lower()
        if c not in colunas_id and any(t in c_lower for t in termos_excluir):
            colunas_excluir.append(c)

    colunas_indicadores = [c for c in df.columns if c not in colunas_id + colunas_excluir]

    df_long = df.melt(
        id_vars=colunas_id,
        value_vars=colunas_indicadores,
        var_name="indicador_original",
        value_name="valor"
    )

    df_long["data"] = pd.NaT
    df_long["fonte"] = "Indicadores por UBS"

    return df_long


def calcular_score(valor, meta, polaridade):
    if pd.isna(valor) or pd.isna(meta) or meta == 0:
        return np.nan

    if polaridade == "menor_melhor":
        if valor == 0:
            return 100
        score = (meta / valor) * 100
    else:
        score = (valor / meta) * 100

    return max(0, min(score, 100))


def classificar_status(score):
    if pd.isna(score):
        return "Sem dado"

    if score >= 85:
        return "Adequado"
    elif score >= 60:
        return "Atenção"
    else:
        return "Crítico"


def criar_periodo_numero(mes):
    meses = {
        "janeiro": 1,
        "fevereiro": 2,
        "março": 3,
        "marco": 3,
        "abril": 4,
        "maio": 5,
        "junho": 6,
        "julho": 7,
        "agosto": 8,
        "setembro": 9,
        "outubro": 10,
        "novembro": 11,
        "dezembro": 12
    }

    texto = str(mes).lower().strip()

    ano = 2024
    for parte in texto.replace("-", "/").split("/"):
        if parte.isdigit() and len(parte) == 4:
            ano = int(parte)

    mes_num = None
    for nome, numero in meses.items():
        if nome in texto:
            mes_num = numero
            break

    if mes_num is None:
        try:
            return pd.to_datetime(mes).toordinal()
        except Exception:
            return np.nan

    return ano * 12 + mes_num


def ordenar_meses_lista(meses):
    """
    Ordena meses e períodos em ordem cronológica usando criar_periodo_numero().
    Evita sequências como Agosto, Julho, Outubro, Setembro nos gráficos e filtros.
    """
    meses_validos = [str(m).strip() for m in meses if str(m).strip() and str(m).lower() != "nan"]

    def chave(m):
        valor = criar_periodo_numero(m)
        if pd.isna(valor):
            return 999999
        return valor

    return sorted(set(meses_validos), key=lambda m: (chave(m), m))


def ordenar_dataframe_por_mes(df, coluna_mes="mes"):
    """
    Adiciona ordem temporal auxiliar e ordena DataFrame por mês/período.
    """
    if df is None or df.empty or coluna_mes not in df.columns:
        return df

    df = df.copy()
    df["_ordem_mes"] = df[coluna_mes].apply(criar_periodo_numero)
    df = df.sort_values(["_ordem_mes", coluna_mes])
    return df



@st.cache_data
def preparar_dados():
    arquivo_municipal, arquivo_ubs = detectar_arquivos()

    if arquivo_municipal is None or arquivo_ubs is None:
        return None, None, arquivo_municipal, arquivo_ubs

    df_municipal = carregar_excel(arquivo_municipal)
    df_ubs = carregar_excel(arquivo_ubs)
    dic = carregar_dicionario()

    if dic is None:
        return None, None, arquivo_municipal, arquivo_ubs

    base_municipal = transformar_base_municipal(df_municipal)
    base_ubs = transformar_base_ubs(df_ubs)

    dados = pd.concat([base_municipal, base_ubs], ignore_index=True)

    if "ubs" in dados.columns:
        dados["ubs"] = dados["ubs"].apply(normalizar_nome_ubs)

    dados["indicador_original"] = dados["indicador_original"].apply(limpar_nome_coluna)
    dados["chave_indicador"] = dados["indicador_original"].apply(chave_texto)
    dados["valor"] = pd.to_numeric(dados["valor"], errors="coerce")

    dados = dados.dropna(subset=["valor"])

    dados = dados.merge(
        dic[
            [
                "chave_indicador",
                "nome_curto",
                "dimensao",
                "nivel",
                "polaridade",
                "meta"
            ]
        ],
        on="chave_indicador",
        how="left"
    )

    dados["indicador"] = dados["nome_curto"].fillna(dados["indicador_original"])
    dados["dimensao"] = dados["dimensao"].fillna("Sem dimensão")
    dados["nivel"] = dados["nivel"].fillna("Não classificado")
    dados["polaridade"] = dados["polaridade"].fillna("maior_melhor")
    dados["meta"] = pd.to_numeric(dados["meta"], errors="coerce")

    dados["score"] = dados.apply(
        lambda linha: calcular_score(
            linha["valor"],
            linha["meta"],
            linha["polaridade"]
        ),
        axis=1
    )

    dados["status"] = dados["score"].apply(classificar_status)
    dados["periodo_num"] = dados["mes"].apply(criar_periodo_numero)

    dados = dados.dropna(subset=["score"])

    return dados, dic, arquivo_municipal, arquivo_ubs


def calcular_sistematicidade(dados):
    tabela = dados.pivot_table(
        index=["ubs", "mes"],
        columns="indicador",
        values="score",
        aggfunc="mean"
    )

    if tabela.shape[1] < 2:
        return np.nan, None, None

    corr = tabela.corr().abs()
    matriz_sem_diag = corr.where(~np.eye(corr.shape[0], dtype=bool))

    relevantes = matriz_sem_diag.stack()
    relevantes = relevantes[relevantes >= 0.30]

    if len(relevantes) == 0:
        sistematicidade = 0
    else:
        sistematicidade = relevantes.mean() * 100

    conectividade = matriz_sem_diag.mean().sort_values(ascending=False).reset_index()
    conectividade.columns = ["indicador", "conectividade_media"]

    return sistematicidade, corr, conectividade


def treinar_modelos_ml(dados_ml):
    dados_ml = dados_ml.copy()

    dados_ml = (
        dados_ml
        .groupby("periodo_num")["score"]
        .mean()
        .reset_index()
        .dropna()
        .sort_values("periodo_num")
    )

    if len(dados_ml) < 4:
        return None, "Dados insuficientes para treinar modelos. É necessário ter pelo menos 4 períodos."

    dados_ml["lag_1"] = dados_ml["score"].shift(1)
    dados_ml["lag_2"] = dados_ml["score"].shift(2)
    dados_ml = dados_ml.dropna()

    if len(dados_ml) < 4:
        return None, "Dados insuficientes após criação das defasagens temporais."

    X = dados_ml[["periodo_num", "lag_1", "lag_2"]]
    y = dados_ml["score"]

    if len(dados_ml) >= 8:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            shuffle=False
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    modelos = {
        "Regressão Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "MLP Neural Network": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            random_state=42
        )
    }

    resultados = []
    previsoes = {}

    for nome, modelo in modelos.items():
        try:
            if nome == "MLP Neural Network":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)

                ultimo = X.iloc[[-1]].copy()
                proximo = ultimo.copy()
                proximo["periodo_num"] = ultimo["periodo_num"].iloc[0] + 1
                proximo["lag_1"] = y.iloc[-1]
                proximo["lag_2"] = y.iloc[-2]

                proximo_scaled = scaler.transform(proximo)
                previsao_futura = modelo.predict(proximo_scaled)[0]
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                ultimo = X.iloc[[-1]].copy()
                proximo = ultimo.copy()
                proximo["periodo_num"] = ultimo["periodo_num"].iloc[0] + 1
                proximo["lag_1"] = y.iloc[-1]
                proximo["lag_2"] = y.iloc[-2]

                previsao_futura = modelo.predict(proximo)[0]

            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            resultados.append({
                "modelo": nome,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "previsao_proximo_periodo": max(0, min(previsao_futura, 100))
            })

            previsoes[nome] = pd.DataFrame({
                "observado": y_test.values,
                "previsto": y_pred
            })

        except Exception:
            resultados.append({
                "modelo": nome,
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "previsao_proximo_periodo": np.nan
            })

    resultados = pd.DataFrame(resultados)

    return {
        "serie": dados_ml,
        "resultados": resultados,
        "previsoes": previsoes
    }, None


def classificar_modelo_por_mae(mae):
    if pd.isna(mae):
        return "Indisponível"
    if mae <= 5:
        return "Excelente"
    elif mae <= 10:
        return "Bom"
    elif mae <= 20:
        return "Moderado"
    else:
        return "Frágil"


def preparar_resultados_ml_para_graficos(resultados):
    df = resultados.copy()

    for col in ["R2", "RMSE", "MAE", "previsao_proximo_periodo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qualidade"] = df["MAE"].apply(classificar_modelo_por_mae)

    df["R2_ajustado"] = df["R2"].fillna(0).clip(lower=0)

    rmse_fill = df["RMSE"].max() if df["RMSE"].notna().any() else 0
    mae_fill = df["MAE"].max() if df["MAE"].notna().any() else 0

    df["RMSE_inv"] = 1 / (1 + df["RMSE"].fillna(rmse_fill))
    df["MAE_inv"] = 1 / (1 + df["MAE"].fillna(mae_fill))

    df["score_modelo"] = (
        0.40 * df["R2_ajustado"] +
        0.30 * df["RMSE_inv"] +
        0.30 * df["MAE_inv"]
    ) * 100

    return df.sort_values("MAE")



def exibir_card_ubs(nome_ubs, score, criticos, atencao, adequados):
    """
    Card individual do Mapa executivo das UBS.
    Versão sem indentação HTML para evitar que o Streamlit mostre tags como texto.
    """
    score = 0 if pd.isna(score) else float(score)

    if score >= 85:
        classe = "ubs-card ubs-ok"
        status = "Adequado"
        dot_class = "status-dot-ok"
    elif score >= 60:
        classe = "ubs-card ubs-alerta"
        status = "Atenção"
        dot_class = "status-dot-alerta"
    else:
        classe = "ubs-card ubs-critico"
        status = "Crítico"
        dot_class = "status-dot-critico"

    html = (
        f'<div class="{classe}">'
        f'<div class="ubs-card-head">'
        f'<div class="ubs-title-wrap">'
        f'<div class="ubs-mini-icon">🏥</div>'
        f'<div class="ubs-title">{nome_ubs}</div>'
        f'</div>'
        f'<div class="status-dot {dot_class}"></div>'
        f'</div>'
        f'<div class="ubs-score">{score:.1f}</div>'
        f'<div class="ubs-status">{status}</div>'
        f'<div class="ubs-small">'
        f'Críticos: <b>{criticos}</b> &nbsp;|&nbsp; '
        f'Atenção: <b>{atencao}</b> &nbsp;|&nbsp; '
        f'Adequados: <b>{adequados}</b>'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def plotly_layout(fig, height=500, showlegend=True):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=65, b=35),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#344054"),
        title=dict(font=dict(size=18, color="#0B2545")),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E7EEF7",
        zeroline=False,
        linecolor="#DDE7F2"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E7EEF7",
        zeroline=False,
        linecolor="#DDE7F2"
    )
    return fig


def fig_gauge(valor, titulo, subtitulo="", maximo=100):
    valor = 0 if pd.isna(valor) else float(valor)

    if valor >= 85:
        cor = COLORS["green"]
    elif valor >= 60:
        cor = COLORS["amber"]
    else:
        cor = COLORS["red"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        number={"suffix": "", "font": {"size": 42, "color": COLORS["navy"]}},
        title={"text": f"<b>{titulo}</b><br><span style='font-size:13px;color:#667085'>{subtitulo}</span>"},
        gauge={
            "axis": {"range": [0, maximo], "tickwidth": 1, "tickcolor": "#667085"},
            "bar": {"color": cor, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#FDECEC"},
                {"range": [60, 85], "color": "#FFF8E1"},
                {"range": [85, 100], "color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": COLORS["navy"], "width": 3},
                "thickness": 0.75,
                "value": valor
            }
        }
    ))

    fig.update_layout(
        height=280,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=70, b=20)
    )

    return fig


def normalizar_serie_0_100(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.max() == s.min():
        return pd.Series([50] * len(s), index=s.index)
    return 100 * (s - s.min()) / (s.max() - s.min())


def calcular_camada_ciberfisica(dados_base, dados_simulado=None):
    atual = dados_base.copy()

    qualidade_dados = 100 * atual["valor"].notna().mean()
    atualizacao = 100
    integracao = min(100, atual["indicador"].nunique() / 18 * 100)
    sensoriamento = min(100, atual["ubs"].nunique() / max(1, atual["ubs"].nunique()) * 100)

    if dados_simulado is None:
        resposta = atual["score"].mean()
    else:
        ganho = dados_simulado["score"].mean() - atual["score"].mean()
        resposta = max(0, min(100, 50 + ganho * 5))

    maturidade = np.nanmean([qualidade_dados, atualizacao, integracao, sensoriamento, resposta])

    return pd.DataFrame({
        "camada": [
            "Coleta/Sensores",
            "Integração de Dados",
            "Qualidade do Dado",
            "Atualização",
            "Resposta Simulada",
            "Maturidade DT"
        ],
        "valor": [
            sensoriamento,
            integracao,
            qualidade_dados,
            atualizacao,
            resposta,
            maturidade
        ]
    })


def criar_fluxo_ciberfisico():
    etapas = pd.DataFrame({
        "etapa": [
            "UBS",
            "Formulários/APIs",
            "Base SMD",
            "Machine Learning",
            "Digital Twin",
            "Decisão"
        ],
        "x": [0, 1, 2, 3, 4, 5],
        "y": [1, 1, 1, 1, 1, 1],
        "descricao": [
            "Processos reais da APS",
            "Coleta e transmissão dos dados",
            "18 indicadores estruturados",
            "Predição e identificação de padrões",
            "Simulação de cenários",
            "Ação gerencial"
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=etapas["x"],
        y=etapas["y"],
        mode="lines+markers+text",
        line=dict(color=COLORS["cyan"], width=5),
        marker=dict(
            size=34,
            color=[COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["amber"], COLORS["cyan"], COLORS["navy"]],
            line=dict(color="white", width=3)
        ),
        text=["🏥", "🔌", "📊", "🤖", "🧠", "🎯"],
        textposition="middle center",
        hovertext=etapas["etapa"] + "<br>" + etapas["descricao"],
        hoverinfo="text",
        showlegend=False
    ))

    for _, row in etapas.iterrows():
        fig.add_annotation(
            x=row["x"],
            y=row["y"] - 0.18,
            text=f"<b>{row['etapa']}</b><br><span style='font-size:11px'>{row['descricao']}</span>",
            showarrow=False,
            font=dict(size=12, color=COLORS["navy"]),
            align="center"
        )

    fig.update_layout(
        height=340,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.4, 5.4]),
        yaxis=dict(visible=False, range=[0.45, 1.35]),
        margin=dict(l=20, r=20, t=40, b=50),
        title=dict(text="Arquitetura ciberfísica do APS-Twin SMD", font=dict(size=19, color=COLORS["navy"]))
    )

    return fig



def google_sheet_to_csv_url(url):
    """
    Converte links do Google Sheets para uma URL CSV.

    Aceita:
    - link normal de edição/visualização;
    - link compartilhado com ?usp=sharing;
    - link com gid;
    - link já em CSV;
    - link de publicação na web.
    """
    if not url or not str(url).strip():
        return None

    url = str(url).strip()

    if "output=csv" in url or "format=csv" in url or url.endswith(".csv"):
        return url

    if "docs.google.com/spreadsheets" not in url:
        return url

    try:
        sheet_id = url.split("/d/")[1].split("/")[0]

        gid = "0"
        if "gid=" in url:
            gid = url.split("gid=")[-1].split("&")[0].split("#")[0]

        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    except Exception:
        return url


@st.cache_data(ttl=60)
def carregar_google_sheet_csv(url):
    """
    Lê uma planilha do Google Sheets como CSV.
    Faz duas tentativas:
    1. endpoint /export?format=csv;
    2. endpoint /gviz/tq?tqx=out:csv.
    """
    csv_url = google_sheet_to_csv_url(url)

    if csv_url is None:
        return None, None

    tentativas = [csv_url]

    if "docs.google.com/spreadsheets/d/" in csv_url:
        try:
            sheet_id = csv_url.split("/d/")[1].split("/")[0]

            gid = "0"
            if "gid=" in csv_url:
                gid = csv_url.split("gid=")[-1].split("&")[0].split("#")[0]

            tentativas.append(
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
            )
        except Exception:
            pass

    ultimo_erro = None

    for tentativa in tentativas:
        try:
            df = pd.read_csv(tentativa)
            df.columns = [limpar_nome_coluna(c) for c in df.columns]

            if df.empty:
                ultimo_erro = "A planilha foi lida, mas está vazia."
                continue

            return df, tentativa

        except Exception as e:
            ultimo_erro = e

    raise Exception(
        f"Não foi possível ler a planilha. Último erro: {ultimo_erro}. "
        "Verifique se a planilha está pública, compartilhada como Leitor ou publicada na web."
    )



def finalizar_base_longa_com_dicionario(base_longa, dic, fonte_nome):
    """
    Recebe uma base em formato longo e aplica o dicionário dos 18 indicadores.
    Usada tanto para bases locais quanto para bases vindas do Google Sheets/Formulários.
    """
    dados_tmp = base_longa.copy()

    dados_tmp["indicador_original"] = dados_tmp["indicador_original"].apply(limpar_nome_coluna)
    dados_tmp["chave_indicador"] = dados_tmp["indicador_original"].apply(chave_texto)
    dados_tmp["valor"] = pd.to_numeric(dados_tmp["valor"], errors="coerce")
    dados_tmp = dados_tmp.dropna(subset=["valor"])

    dados_tmp = dados_tmp.merge(
        dic[
            [
                "chave_indicador",
                "nome_curto",
                "dimensao",
                "nivel",
                "polaridade",
                "meta"
            ]
        ],
        on="chave_indicador",
        how="left"
    )

    dados_tmp["indicador"] = dados_tmp["nome_curto"].fillna(dados_tmp["indicador_original"])
    dados_tmp["dimensao"] = dados_tmp["dimensao"].fillna("Sem dimensão")
    dados_tmp["nivel"] = dados_tmp["nivel"].fillna("Não classificado")
    dados_tmp["polaridade"] = dados_tmp["polaridade"].fillna("maior_melhor")
    dados_tmp["meta"] = pd.to_numeric(dados_tmp["meta"], errors="coerce")

    dados_tmp["score"] = dados_tmp.apply(
        lambda linha: calcular_score(
            linha["valor"],
            linha["meta"],
            linha["polaridade"]
        ),
        axis=1
    )

    dados_tmp["status"] = dados_tmp["score"].apply(classificar_status)
    dados_tmp["periodo_num"] = dados_tmp["mes"].apply(criar_periodo_numero)
    dados_tmp["fonte"] = fonte_nome

    if "ubs" in dados_tmp.columns:
        dados_tmp["ubs"] = dados_tmp["ubs"].apply(normalizar_nome_ubs)

    dados_tmp["mes"] = dados_tmp["mes"].astype(str).str.strip()

    dados_tmp = dados_tmp.dropna(subset=["score"])

    return dados_tmp


@st.cache_data(ttl=15)
def carregar_google_sheet_csv_realtime(url):
    """
    Lê o Google Sheets com cache curto para simular atualização em tempo real.
    TTL = 15 segundos.
    """
    return carregar_google_sheet_csv(url)


def preparar_google_sheet_para_smd(df_google, dic):
    """
    Transforma uma planilha de respostas do Google Forms/Sheets em dados do SMD.
    Espera uma estrutura parecida com:
    - coluna de mês;
    - coluna de UBS;
    - colunas dos indicadores.
    Campos como timestamp e e-mail são ignorados automaticamente.
    """
    if df_google is None or df_google.empty:
        return pd.DataFrame()

    df_google = df_google.copy()
    df_google.columns = [limpar_nome_coluna(c) for c in df_google.columns]

    colunas_txt = " ".join([str(c).lower() for c in df_google.columns])

    if "ubs" in colunas_txt:
        base_longa = transformar_base_ubs(df_google)
    else:
        base_longa = transformar_base_municipal(df_google)

    dados_google = finalizar_base_longa_com_dicionario(
        base_longa=base_longa,
        dic=dic,
        fonte_nome="API Google Forms/Sheets"
    )

    return dados_google



def exibir_decision_card(titulo, texto, prioridade="Moderado"):
    prioridade_norm = str(prioridade).lower()
    classe = "priority-medium"
    if prioridade_norm in ["crítico", "critico"]:
        classe = "priority-critical"
    elif prioridade_norm == "alto":
        classe = "priority-high"
    elif prioridade_norm == "baixo":
        classe = "priority-low"

    st.markdown(
        f"""
        <div class="decision-card">
            <div class="decision-title">{titulo}</div>
            <div class="decision-text">{texto}</div>
            <span class="priority-badge {classe}">{prioridade}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def exibir_kpi_card(label, value, icon="📊", note="", accent="#1976D2"):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 5px solid {accent};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )




def calcular_qualidade_dados(dados_base, dic_base):
    """
    Calcula métricas simples de qualidade dos dados para gestão do SMD.
    """
    if dados_base is None or dados_base.empty:
        return {
            "qualidade_geral": 0,
            "completude": 0,
            "cobertura_indicadores": 0,
            "cobertura_ubs": 0,
            "consistencia_metas": 0,
            "n_registros": 0,
            "n_indicadores": 0,
            "n_ubs": 0,
            "n_meses": 0
        }

    n_indicadores_esperados = 18
    n_indicadores = dados_base["indicador"].nunique()
    n_ubs = dados_base["ubs"].nunique()
    n_meses = dados_base["mes"].astype(str).nunique()

    completude = 100 * dados_base["valor"].notna().mean()
    cobertura_indicadores = min(100, 100 * n_indicadores / n_indicadores_esperados)
    cobertura_ubs = min(100, 100 * n_ubs / max(1, n_ubs))
    consistencia_metas = 100 * dados_base["meta"].notna().mean()

    qualidade_geral = np.nanmean([
        completude,
        cobertura_indicadores,
        cobertura_ubs,
        consistencia_metas
    ])

    return {
        "qualidade_geral": qualidade_geral,
        "completude": completude,
        "cobertura_indicadores": cobertura_indicadores,
        "cobertura_ubs": cobertura_ubs,
        "consistencia_metas": consistencia_metas,
        "n_registros": len(dados_base),
        "n_indicadores": n_indicadores,
        "n_ubs": n_ubs,
        "n_meses": n_meses
    }


def calcular_maturidade_dt(dados_base, sistematicidade_valor, api_ativa=False, ml_ativo=True, simulacao_ativa=True):
    """
    Índice experimental de maturidade do Digital Twin.
    Mede se o sistema tem dados, integração, modelo, simulação e alimentação digital.
    """
    q = calcular_qualidade_dados(dados_base, None)

    componente_dados = q["qualidade_geral"]
    componente_integracao = 100 if api_ativa else 55
    componente_ml = 100 if ml_ativo else 0
    componente_simulacao = 100 if simulacao_ativa else 0
    componente_sistematicidade = 0 if pd.isna(sistematicidade_valor) else min(100, sistematicidade_valor)

    maturidade = np.nanmean([
        componente_dados,
        componente_integracao,
        componente_ml,
        componente_simulacao,
        componente_sistematicidade
    ])

    return {
        "maturidade_dt": maturidade,
        "dados": componente_dados,
        "integracao": componente_integracao,
        "ml": componente_ml,
        "simulacao": componente_simulacao,
        "sistematicidade": componente_sistematicidade
    }


def calcular_risco_gerencial(dados_base, qualidade_geral, sistematicidade_valor):
    """
    Índice experimental de risco gerencial.
    Quanto maior, maior a necessidade de atenção da gestão.
    """
    if dados_base is None or dados_base.empty:
        return 100, "Crítico"

    total = len(dados_base)
    perc_critico = 100 * (dados_base["status"].eq("Crítico").sum() / max(1, total))
    score_medio = dados_base["score"].mean()

    risco_score = max(0, 100 - score_medio)
    risco_qualidade = max(0, 100 - qualidade_geral)
    risco_sistema = 100 - (0 if pd.isna(sistematicidade_valor) else min(100, sistematicidade_valor))

    risco = np.nanmean([
        perc_critico,
        risco_score,
        risco_qualidade,
        risco_sistema
    ])

    if risco >= 70:
        classe = "Crítico"
    elif risco >= 45:
        classe = "Alto"
    elif risco >= 25:
        classe = "Moderado"
    else:
        classe = "Baixo"

    return risco, classe


def gerar_relatorio_textual(dados_base, ubs_ref, indice_geral_ref, sistematicidade_ref, qualidade_ref, maturidade_ref, risco_ref, classe_risco_ref):
    """
    Gera relatório textual simples em Markdown para download.
    """
    periodo = ", ".join(sorted(dados_base["mes"].astype(str).unique())[:8])
    if dados_base["mes"].astype(str).nunique() > 8:
        periodo += " ..."

    resumo_ind = (
        dados_base
        .groupby(["indicador", "dimensao"])
        .agg(
            score_medio=("score", "mean"),
            valor_medio=("valor", "mean"),
            registros=("score", "count")
        )
        .reset_index()
        .sort_values("score_medio")
    )

    criticos = resumo_ind[resumo_ind["score_medio"] < 60].head(8)
    melhores = resumo_ind.sort_values("score_medio", ascending=False).head(5)

    linhas_criticas = "\n".join([
        f"- {r['indicador']} ({r['dimensao']}): score médio {r['score_medio']:.1f}"
        for _, r in criticos.iterrows()
    ]) or "- Nenhum indicador crítico no recorte atual."

    linhas_melhores = "\n".join([
        f"- {r['indicador']} ({r['dimensao']}): score médio {r['score_medio']:.1f}"
        for _, r in melhores.iterrows()
    ])

    sistematicidade_texto = 0 if pd.isna(sistematicidade_ref) else sistematicidade_ref

    relatorio = f"""# Relatório Executivo APS-Twin SMD

## 1. Identificação do recorte analisado

**Unidade/Nível:** {ubs_ref}  
**Período analisado:** {periodo}  
**Registros analisados:** {len(dados_base)}  
**Indicadores monitorados:** {dados_base['indicador'].nunique()}  
**Dimensões avaliadas:** {dados_base['dimensao'].nunique()}  

## 2. Indicadores executivos

- **Índice Geral do SMD:** {indice_geral_ref:.1f}
- **Índice de Sistematicidade:** {sistematicidade_texto:.1f}
- **Qualidade dos Dados:** {qualidade_ref:.1f}
- **Maturidade Digital Twin:** {maturidade_ref:.1f}
- **Risco Gerencial:** {risco_ref:.1f} ({classe_risco_ref})

## 3. Indicadores mais críticos

{linhas_criticas}

## 4. Indicadores com melhor desempenho

{linhas_melhores}

## 5. Leitura gerencial automática

O recorte analisado apresenta Índice Geral do SMD de {indice_geral_ref:.1f}. A maturidade do Digital Twin foi estimada em {maturidade_ref:.1f}, indicando o grau de integração entre dados, indicadores, modelos analíticos, simulação e apoio à decisão. O risco gerencial foi classificado como **{classe_risco_ref}**, sugerindo o nível de prioridade para acompanhamento da gestão.

## 6. Recomendações iniciais

1. Priorizar os indicadores com menor score médio.
2. Verificar a qualidade e periodicidade dos dados recebidos.
3. Utilizar o módulo de Machine Learning para antecipar tendências.
4. Simular intervenções na aba Digital Twin Ciberfísico.
5. Atualizar o formulário/API de entrada para garantir alimentação contínua do SMD.

---

Relatório gerado automaticamente pelo APS-Twin SMD v1.9.0.7.7.6.6.5.5.4.4.3.3.2.2.1.
"""
    return relatorio


def grafico_qualidade_dados(qualidade_dict):
    dfq = pd.DataFrame({
        "métrica": [
            "Completude",
            "Cobertura dos indicadores",
            "Cobertura das UBS",
            "Consistência das metas",
            "Qualidade geral"
        ],
        "valor": [
            qualidade_dict["completude"],
            qualidade_dict["cobertura_indicadores"],
            qualidade_dict["cobertura_ubs"],
            qualidade_dict["consistencia_metas"],
            qualidade_dict["qualidade_geral"]
        ]
    })

    fig = px.bar(
        dfq.sort_values("valor"),
        x="valor",
        y="métrica",
        orientation="h",
        text="valor",
        color="valor",
        color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
        title="Painel de qualidade dos dados"
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    return plotly_layout(fig, height=430, showlegend=False)


def grafico_maturidade_dt(maturidade_dict):
    dfm = pd.DataFrame({
        "camada": [
            "Dados",
            "Integração/API",
            "Machine Learning",
            "Simulação",
            "Sistematicidade"
        ],
        "valor": [
            maturidade_dict["dados"],
            maturidade_dict["integracao"],
            maturidade_dict["ml"],
            maturidade_dict["simulacao"],
            maturidade_dict["sistematicidade"]
        ]
    })

    fig = px.line_polar(
        dfm,
        r="valor",
        theta="camada",
        line_close=True,
        title="Radar de maturidade do Digital Twin"
    )
    fig.update_traces(fill="toself", line=dict(color=COLORS["teal"], width=4))
    fig.update_layout(
        height=450,
        paper_bgcolor="white",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title_font=dict(size=18, color=COLORS["navy"])
    )
    return fig


# =========================================================
# RESPONSIVIDADE, ROTEAMENTO POR LOGIN E LAYOUTS ADAPTATIVOS
# =========================================================

def inject_responsive_css():
    """CSS final para deixar cards, icones, navegacao e graficos proporcionais em desktop e celular."""
    st.markdown(
        """
        <style>
            :root {
                --resp-card-radius: 20px;
                --resp-gap: clamp(10px, 1.4vw, 18px);
            }

            .block-container {
                width: min(100%, 1480px);
                padding-left: clamp(0.85rem, 2vw, 2.5rem) !important;
                padding-right: clamp(0.85rem, 2vw, 2.5rem) !important;
            }

            .responsive-shell {
                width: 100%;
                box-sizing: border-box;
            }

            .responsive-card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr));
                gap: var(--resp-gap);
                align-items: stretch;
                margin: 12px 0 18px 0;
            }

            .responsive-card-grid.wide {
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr));
            }

            .responsive-kpi-card {
                height: 100%;
                min-height: clamp(116px, 13vw, 150px);
                box-sizing: border-box;
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
                border: 1px solid #E5EDF6;
                border-left: 7px solid var(--accent, #1976D2);
                border-radius: var(--resp-card-radius);
                padding: clamp(13px, 1.35vw, 18px);
                box-shadow: 0 10px 24px rgba(11,37,69,0.075);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }

            .responsive-kpi-top {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 10px;
            }

            .responsive-kpi-icon {
                width: clamp(34px, 3.6vw, 44px);
                height: clamp(34px, 3.6vw, 44px);
                min-width: clamp(34px, 3.6vw, 44px);
                border-radius: 14px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: clamp(18px, 2.2vw, 25px);
                line-height: 1;
                background: rgba(25,118,210,0.09);
                box-shadow: inset 0 1px 2px rgba(255,255,255,0.75);
            }

            .responsive-kpi-label {
                color: #667085;
                font-size: clamp(10px, 1.05vw, 12px);
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: .04em;
                line-height: 1.25;
                overflow-wrap: anywhere;
            }

            .responsive-kpi-value {
                color: #0B2545;
                font-size: clamp(24px, 4.1vw, 38px);
                font-weight: 950;
                line-height: 1.02;
                letter-spacing: -0.6px;
                overflow-wrap: anywhere;
            }

            .responsive-kpi-note {
                color: #667085;
                font-size: clamp(10.5px, 1.05vw, 12px);
                line-height: 1.35;
                overflow-wrap: anywhere;
            }

            .responsive-section-card {
                background: rgba(255,255,255,0.96);
                border: 1px solid #E5EDF6;
                border-radius: 24px;
                padding: clamp(14px, 1.8vw, 22px);
                box-shadow: 0 10px 24px rgba(11,37,69,0.06);
                margin: 12px 0 18px 0;
            }

            .responsive-nav-title {
                color: #0B2545;
                font-size: clamp(16px, 1.5vw, 20px);
                font-weight: 950;
                margin-bottom: 4px;
            }

            .responsive-nav-caption {
                color: #667085;
                font-size: clamp(11.5px, 1.1vw, 13px);
                margin-bottom: 10px;
            }

            .profile-badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }

            .profile-badge {
                background: #EEF6FF;
                color: #0B2545;
                border: 1px solid #D9EAFB;
                border-radius: 999px;
                padding: 7px 11px;
                font-size: clamp(10.5px, 1vw, 12px);
                font-weight: 850;
                white-space: nowrap;
            }

            .ubs-responsive-card {
                min-height: 140px;
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
                border: 1px solid #E5EDF6;
                border-left: 7px solid var(--accent, #1976D2);
                border-radius: 20px;
                padding: 15px 16px;
                box-shadow: 0 10px 24px rgba(11,37,69,0.075);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }

            .ubs-responsive-title {
                color: #0B2545;
                font-size: clamp(16px, 1.7vw, 20px);
                font-weight: 950;
                line-height: 1.15;
                overflow-wrap: anywhere;
            }

            .ubs-responsive-score {
                color: #0B2545;
                font-size: clamp(26px, 4vw, 36px);
                font-weight: 950;
                line-height: 1;
            }

            .ubs-responsive-meta {
                color: #667085;
                font-size: clamp(10.5px, 1vw, 12px);
                line-height: 1.35;
            }

            .public-hero {
                background: radial-gradient(circle at 88% 8%, rgba(255,255,255,0.22), transparent 28%),
                            linear-gradient(135deg, #0B2545 0%, #1976D2 58%, #0097A7 100%);
                color: #FFFFFF !important;
                border-radius: clamp(20px, 2.4vw, 30px);
                padding: clamp(22px, 3.2vw, 36px);
                box-shadow: 0 16px 36px rgba(11,37,69,0.20);
                margin: 10px 0 18px 0;
                overflow: hidden;
            }

            .public-hero-title {
                color: #FFFFFF !important;
                font-size: clamp(28px, 4.6vw, 56px);
                font-weight: 950;
                line-height: 1.02;
                letter-spacing: -1px;
                margin-bottom: 8px;
            }

            .public-hero-text {
                color: rgba(255,255,255,0.94) !important;
                font-size: clamp(14px, 1.5vw, 18px);
                line-height: 1.55;
                max-width: 980px;
            }

            div[data-testid="stMetric"] {
                min-height: 110px;
            }

            div[data-testid="stButton"] button {
                white-space: normal !important;
                word-break: normal !important;
                line-height: 1.15 !important;
            }

            div[data-testid="stSelectbox"] label,
            div[data-testid="stMultiSelect"] label,
            div[data-testid="stRadio"] label {
                font-weight: 800;
                color: #0B2545;
            }

            @media (max-width: 900px) {
                .hero-main, .app-header {
                    border-radius: 20px !important;
                    padding: 22px !important;
                }

                .hero-grid-html {
                    grid-template-columns: 1fr !important;
                }

                .identity-panel {
                    width: 100% !important;
                    max-width: 100% !important;
                }

                .identity-grid, .partner-grid {
                    grid-template-columns: 1fr 1fr !important;
                }

                .section-title {
                    font-size: 22px !important;
                    line-height: 1.18 !important;
                }

                .responsive-card-grid {
                    grid-template-columns: 1fr 1fr;
                }
            }

            @media (max-width: 560px) {
                .block-container {
                    padding-top: 0.75rem !important;
                    padding-left: 0.75rem !important;
                    padding-right: 0.75rem !important;
                }

                .responsive-card-grid,
                .responsive-card-grid.wide {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }

                .identity-grid, .partner-grid {
                    grid-template-columns: 1fr !important;
                }

                .profile-badge {
                    width: 100%;
                    box-sizing: border-box;
                    text-align: center;
                }

                .responsive-section-card {
                    border-radius: 18px;
                    padding: 13px;
                }

                .public-hero {
                    border-radius: 18px;
                    padding: 20px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def _safe_html(valor):
    return html_lib.escape(str(valor))


def responsive_metric_card(label, value, icon="📊", note="", accent="#1976D2"):
    """Retorna uma estrutura simples para renderizacao nativa no Streamlit.

    A versao anterior retornava HTML. Em alguns ambientes, especialmente
    exportando/visualizando em PDF ou com tema/versao diferente do Streamlit,
    esse HTML pode aparecer como texto na tela. Por isso, os cards principais
    agora sao desenhados com componentes nativos: st.container, st.columns,
    st.metric, st.caption e st.markdown.
    """
    return {
        "label": str(label),
        "value": str(value),
        "icon": str(icon),
        "note": str(note),
        "accent": str(accent),
    }


def _chunks(lista, tamanho):
    for i in range(0, len(lista), tamanho):
        yield lista[i:i + tamanho]


def _render_native_metric_card(card):
    label = card.get("label", "Indicador") if isinstance(card, dict) else str(card)
    value = card.get("value", "-") if isinstance(card, dict) else "-"
    icon = card.get("icon", "📊") if isinstance(card, dict) else "📊"
    note = card.get("note", "") if isinstance(card, dict) else ""

    with st.container(border=True):
        st.markdown(f"**{icon} {label}**")
        st.metric(label="", value=value)
        if note:
            st.caption(note)


def render_metric_grid(cards, wide=False):
    """Renderiza cards proporcionais sem HTML bruto.

    No desktop, os cards aparecem em linhas de 4 ou 3 colunas.
    No celular, o Streamlit empilha automaticamente as colunas, evitando
    que icones e textos fiquem sobrepostos.
    """
    if not cards:
        return

    colunas_por_linha = 3 if wide else 4
    for linha in _chunks(cards, colunas_por_linha):
        cols = st.columns(len(linha))
        for col, card in zip(cols, linha):
            with col:
                _render_native_metric_card(card)
        st.write("")

def modulos_por_perfil():
    perfil = perfil_atual()
    if perfil == "adm":
        return [
            "📊 Painel Executivo",
            "🧪 Qualidade dos Dados",
            "🎯 18 Indicadores",
            "🏥 UBS e Território",
            "📈 Séries Temporais",
            "🤖 Machine Learning",
            "🧠 Digital Twin Ciberfísico",
            "🕸️ Sistematicidade",
            "📄 Relatórios",
            "ℹ️ Sobre o SMD",
        ]
    if perfil == "aluno":
        return [
            "📊 Painel Executivo",
            "🎯 18 Indicadores",
            "📈 Séries Temporais",
            "🤖 Machine Learning",
            "🧠 Digital Twin Ciberfísico",
            "🕸️ Sistematicidade",
            "📄 Relatórios",
            "ℹ️ Sobre o SMD",
        ]
    if perfil == "secretario":
        return [
            "📊 Painel Executivo",
            "🏥 UBS e Território",
            "📈 Séries Temporais",
            "🧠 Digital Twin Ciberfísico",
            "📄 Relatórios",
            "ℹ️ Sobre o SMD",
        ]
    if perfil == "prefeito":
        return [
            "🏛️ Visão do Prefeito",
            "🏥 Prioridade das UBS",
            "👥 Risco para a população",
            "🧭 Cenários e decisões",
            "📄 Resumo de governo",
            "ℹ️ Sobre o SMD",
        ]
    if perfil == "gerente":
        return [
            "🏥 UBS e Território",
            "📊 Painel Executivo",
            "🎯 18 Indicadores",
            "📈 Séries Temporais",
            "📄 Relatórios",
            "ℹ️ Sobre o SMD",
        ]
    if perfil == "populacao":
        return [
            "🏠 Orientações para o cidadão",
            "🏥 Situação das UBS",
            "⏱️ Previsão de atendimento",
            "ℹ️ Sobre o SMD",
        ]
    return ["📊 Painel Executivo", "ℹ️ Sobre o SMD"]


def modulo_inicial_por_perfil():
    perfil = perfil_atual()
    if perfil == "aluno":
        return "🤖 Machine Learning"
    if perfil == "gerente":
        return "🏥 UBS e Território"
    if perfil == "prefeito":
        return "🏛️ Visão do Prefeito"
    if perfil == "populacao":
        return "🏠 Orientações para o cidadão"
    return "📊 Painel Executivo"


def render_nav_responsiva(modulos_disponiveis):
    """Navegacao nativa e proporcional por perfil.

    Evita abrir/fechar divs HTML ao redor de widgets Streamlit, pois isso pode
    provocar renderizacao quebrada em algumas telas.
    """
    if not modulos_disponiveis:
        return

    atual = st.session_state.get("aba_ativa", modulo_inicial_por_perfil())
    if atual not in modulos_disponiveis:
        atual = modulos_disponiveis[0]
        st.session_state["aba_ativa"] = atual

    with st.container(border=True):
        st.markdown("#### Navegação proporcional por perfil")
        st.caption("A seleção abaixo evita quebra visual de ícones no celular e mantém o layout limpo no desktop.")

        escolhido = st.selectbox(
            "Módulo do painel",
            options=modulos_disponiveis,
            index=modulos_disponiveis.index(atual),
            key="nav_responsiva_select"
        )

        if escolhido != atual:
            st.session_state["aba_ativa"] = escolhido
            st.rerun()

        perfil = st.session_state.get("perfil_usuario", "-").upper()
        escopo = st.session_state.get("ubs_usuario", "Todas")
        c1, c2, c3 = st.columns(3)
        c1.caption(f"Perfil: {perfil}")
        c2.caption(f"Escopo: {escopo}")
        c3.caption(f"Módulo ativo: {st.session_state.get('aba_ativa', '')}")

def status_accent(score):
    if pd.isna(score):
        return COLORS["gray"]
    if score >= 85:
        return COLORS["green"]
    if score >= 60:
        return COLORS["amber"]
    return COLORS["red"]


def render_ubs_grid_responsivo(dados_base, titulo="🏥 Mapa executivo das UBS", permitir_abertura=True):
    """Renderiza o painel de UBS com componentes nativos do Streamlit.

    A versao anterior usava HTML bruto para montar a grade. Esta versao usa
    st.columns + st.container(border=True), o que deixa o layout proporcional
    e mais estavel em desktop, notebook, tablet e celular.
    """
    if dados_base is None or dados_base.empty or "ubs" not in dados_base.columns:
        st.info("Nenhuma UBS disponível para o recorte atual.")
        return

    dados_cards_base = aplicar_restricao_de_perfil(dados_base.copy())
    if dados_cards_base.empty:
        st.info("Nenhuma UBS disponível para o perfil atual.")
        return

    dados_cards_base["ubs"] = dados_cards_base["ubs"].apply(normalizar_nome_ubs)
    ubs_validas = [
        u for u in sorted(set(dados_cards_base["ubs"].dropna().astype(str).tolist()))
        if str(u).strip() != ""
        and str(u).lower() not in ["aps municipal", "todas", "nan", "none", "municipal"]
        and "2024" not in str(u)
        and "/" not in str(u)
    ]

    if not ubs_validas:
        return

    resumo = (
        dados_cards_base[dados_cards_base["ubs"].isin(ubs_validas)]
        .groupby("ubs")
        .agg(
            score_ubs=("score", "mean"),
            crit_ubs=("status", lambda s: (s == "Crítico").sum()),
            aten_ubs=("status", lambda s: (s == "Atenção").sum()),
            adeq_ubs=("status", lambda s: (s == "Adequado").sum()),
        )
        .reset_index()
        .sort_values(["score_ubs", "ubs"], ascending=[False, True])
    )

    st.markdown(f"### {titulo}")

    for linha in _chunks(list(resumo.itertuples(index=False)), 3):
        cols = st.columns(len(linha))
        for col, row in zip(cols, linha):
            with col:
                score = float(row.score_ubs) if not pd.isna(row.score_ubs) else 0.0
                if score >= 85:
                    status_txt = "Adequado"
                    status_icon = "✅"
                elif score >= 60:
                    status_txt = "Atenção"
                    status_icon = "⚠️"
                else:
                    status_txt = "Crítico"
                    status_icon = "🚨"

                with st.container(border=True):
                    st.markdown(f"**🏥 {row.ubs}**")
                    st.metric("Score médio da unidade", f"{score:.1f}")
                    st.caption(f"{status_icon} {status_txt}")
                    st.caption(
                        f"Críticos: {int(row.crit_ubs)} · "
                        f"Atenção: {int(row.aten_ubs)} · "
                        f"Adequados: {int(row.adeq_ubs)}"
                    )
        st.write("")

    if permitir_abertura:
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            escolha = st.selectbox(
                "Abrir painel de uma UBS",
                options=["Todas"] + resumo["ubs"].astype(str).tolist(),
                key="select_abrir_ubs_responsivo"
            )
        with col_btn:
            st.write("")
            if st.button("Abrir", use_container_width=True, key="btn_abrir_ubs_responsivo"):
                st.session_state["ubs_selecionada_card"] = escolha
                st.session_state["aba_ativa"] = "🏥 UBS e Território" if escolha != "Todas" else st.session_state.get("aba_ativa", "📊 Painel Executivo")
                st.rerun()

        if st.session_state.get("ubs_selecionada_card", "Todas") != "Todas":
            st.success(f"Painel filtrado para a UBS: {st.session_state['ubs_selecionada_card']}")
            if st.button("Voltar para visão geral municipal", key="voltar_visao_geral_municipal_resp"):
                st.session_state["ubs_selecionada_card"] = "Todas"
                st.rerun()


# =========================================================
# PERFIL POPULAÇÃO: TRADUTOR CIDADÃO DO SMD
# =========================================================

def _formatar_lista_curta(valores, limite=3):
    valores = [str(v) for v in valores if str(v).strip()]
    if not valores:
        return "nenhuma unidade identificada"
    if len(valores) <= limite:
        return ", ".join(valores)
    return ", ".join(valores[:limite]) + f" e mais {len(valores) - limite}"


def _validar_ubs_publica(nome):
    nome_txt = str(nome).strip()
    nome_low = nome_txt.lower()
    if not nome_txt:
        return False
    if nome_low in ["todas", "nan", "none"]:
        return False
    if "2024" in nome_txt or "/" in nome_txt:
        return False
    return True


def _classificar_servico_publico(score_atual, pct_critico, tendencia=0, score_previsto=None):
    """Traduz score técnico em linguagem cidadã.

    Importante: o sistema não mede fila real minuto a minuto. Portanto, as
    mensagens falam em tendência/probabilidade de demora, e não em promessa
    exata de tempo de espera.
    """
    score_ref = score_atual if pd.isna(score_previsto) else score_previsto
    score_ref = 0 if pd.isna(score_ref) else float(score_ref)
    pct_critico = 0 if pd.isna(pct_critico) else float(pct_critico)
    tendencia = 0 if pd.isna(tendencia) else float(tendencia)

    if score_ref < 55 or pct_critico >= 55:
        return {
            "situacao": "Crítica",
            "icone": "🚨",
            "resumo": "Pode haver maior demora ou dificuldade no atendimento.",
            "atendimento": "Procure confirmar funcionamento, horários e forma de agendamento antes de sair de casa.",
            "prioridade": "Alta",
        }

    if score_ref < 70 or pct_critico >= 35 or tendencia < -5:
        return {
            "situacao": "Atenção",
            "icone": "⚠️",
            "resumo": "Pode haver espera moderada ou oscilação no atendimento.",
            "atendimento": "Tente agendar, acompanhe avisos da unidade e organize documentos para evitar retorno desnecessário.",
            "prioridade": "Média",
        }

    return {
        "situacao": "Fluxo melhor",
        "icone": "✅",
        "resumo": "A unidade tende a funcionar com menor risco de lentidão.",
        "atendimento": "Mantenha consultas, vacinação, pré-natal e acompanhamento de rotina em dia.",
        "prioridade": "Baixa",
    }


def _calcular_tendencia_por_grupo(df_grupo):
    if df_grupo is None or df_grupo.empty or "periodo_num" not in df_grupo.columns:
        return 0
    serie = (
        df_grupo
        .groupby("periodo_num")["score"]
        .mean()
        .dropna()
        .sort_index()
    )
    if len(serie) < 2:
        return 0
    return float(serie.iloc[-1] - serie.iloc[-2])


def _prever_score_publico(df_grupo):
    """Gera previsão simples para linguagem pública.

    Usa os modelos já existentes quando houver série suficiente. Quando não há
    dados suficientes, usa a tendência observada como fallback conservador.
    """
    if df_grupo is None or df_grupo.empty:
        return np.nan, "sem série suficiente"

    try:
        resultado_ml, erro_ml = treinar_modelos_ml(df_grupo)
        if erro_ml or resultado_ml is None:
            tendencia = _calcular_tendencia_por_grupo(df_grupo)
            base = float(df_grupo["score"].mean())
            return max(0, min(100, base + tendencia)), "tendência simples"

        resultados = preparar_resultados_ml_para_graficos(resultado_ml["resultados"])
        resultados_validos = resultados.dropna(subset=["MAE", "previsao_proximo_periodo"])
        if resultados_validos.empty:
            tendencia = _calcular_tendencia_por_grupo(df_grupo)
            base = float(df_grupo["score"].mean())
            return max(0, min(100, base + tendencia)), "tendência simples"

        melhor = resultados_validos.sort_values("MAE").iloc[0]
        return float(max(0, min(100, melhor["previsao_proximo_periodo"]))), str(melhor["modelo"])
    except Exception:
        tendencia = _calcular_tendencia_por_grupo(df_grupo)
        base = float(df_grupo["score"].mean())
        return max(0, min(100, base + tendencia)), "tendência simples"


def montar_resumo_publico_ubs(dados_base):
    if dados_base is None or dados_base.empty or "ubs" not in dados_base.columns:
        return pd.DataFrame()

    base = dados_base.copy()
    base["ubs"] = base["ubs"].apply(normalizar_nome_ubs)
    base = base[base["ubs"].apply(_validar_ubs_publica)].copy()

    # Preferimos UBS reais. Se houver apenas agregado municipal, usamos o agregado.
    base_ubs = base[~base["ubs"].astype(str).str.lower().isin(["aps municipal", "municipal"])]
    if not base_ubs.empty:
        base = base_ubs.copy()

    linhas = []
    for ubs, grupo in base.groupby("ubs"):
        total = len(grupo)
        crit = int((grupo["status"] == "Crítico").sum()) if "status" in grupo.columns else 0
        aten = int((grupo["status"] == "Atenção").sum()) if "status" in grupo.columns else 0
        adeq = int((grupo["status"] == "Adequado").sum()) if "status" in grupo.columns else 0
        pct_crit = 100 * crit / max(1, total)
        score = float(grupo["score"].mean())
        tendencia = _calcular_tendencia_por_grupo(grupo)
        score_previsto, metodo = _prever_score_publico(grupo)
        classe = _classificar_servico_publico(score, pct_crit, tendencia, score_previsto)

        dims_criticas = []
        if "dimensao" in grupo.columns:
            dims_criticas = (
                grupo.groupby("dimensao")["score"]
                .mean()
                .sort_values()
                .head(2)
                .index.astype(str)
                .tolist()
            )

        linhas.append({
            "ubs": ubs,
            "situacao": classe["situacao"],
            "icone": classe["icone"],
            "resumo": classe["resumo"],
            "orientacao": classe["atendimento"],
            "prioridade": classe["prioridade"],
            "score_atual": score,
            "score_previsto": score_previsto,
            "tendencia": tendencia,
            "pct_critico": pct_crit,
            "criticos": crit,
            "atencao": aten,
            "adequados": adeq,
            "registros": total,
            "metodo_previsao": metodo,
            "pontos_que_afetam": ", ".join(dims_criticas) if dims_criticas else "sem detalhe disponível",
        })

    resumo = pd.DataFrame(linhas)
    if resumo.empty:
        return resumo

    ordem = {"Crítica": 0, "Atenção": 1, "Fluxo melhor": 2}
    resumo["ordem"] = resumo["situacao"].map(ordem).fillna(9)
    return resumo.sort_values(["ordem", "score_previsto", "score_atual"], ascending=[True, True, True])


def traduzir_dimensao_para_cidadao(dim):
    dim_low = str(dim).lower()
    if "operacional" in dim_low:
        return "Organização da unidade, marcação, fluxo de atendimento e possibilidade de espera."
    if "qualidade" in dim_low or "assistencial" in dim_low:
        return "Continuidade do cuidado, acompanhamento e resolutividade do atendimento."
    if "trabalho" in dim_low:
        return "Condição da equipe para atender bem, sem sobrecarga excessiva."
    if "vigil" in dim_low and "territ" not in dim_low:
        return "Vacinação, prevenção, busca ativa e acompanhamento de grupos prioritários."
    if "territ" in dim_low:
        return "Cobertura do território, visitas, acompanhamento das famílias e resposta às áreas mais vulneráveis."
    return "Aspecto do serviço que pode influenciar acesso, cuidado e acompanhamento da população."


def montar_resumo_publico_dimensoes(dados_base):
    if dados_base is None or dados_base.empty or "dimensao" not in dados_base.columns:
        return pd.DataFrame()

    resumo = (
        dados_base
        .groupby("dimensao")
        .agg(score_medio=("score", "mean"), registros=("score", "count"))
        .reset_index()
        .sort_values("score_medio")
    )
    resumo["o_que_significa_para_voce"] = resumo["dimensao"].apply(traduzir_dimensao_para_cidadao)
    resumo["leitura"] = resumo["score_medio"].apply(
        lambda v: "Precisa de reforço" if v < 60 else ("Acompanhar" if v < 85 else "Funcionando melhor")
    )
    return resumo


def render_alerta_saude_publica():
    st.info(
        "Este painel ajuda a entender tendência de funcionamento das UBS, mas não substitui orientação profissional. "
        "Em caso de falta de ar, dor forte no peito, desmaio, sangramento intenso, sinais de AVC, emergência obstétrica "
        "ou outra situação grave, procure imediatamente o serviço de urgência/emergência."
    )


def render_cards_publicos_resumo(resumo_ubs):
    if resumo_ubs is None or resumo_ubs.empty:
        render_metric_grid([
            responsive_metric_card("Situação geral", "Em análise", "🏥", "Não há dados por UBS suficientes para resumo público.", COLORS["blue"]),
        ], wide=True)
        return

    n_criticas = int((resumo_ubs["situacao"] == "Crítica").sum())
    n_atencao = int((resumo_ubs["situacao"] == "Atenção").sum())
    n_ok = int((resumo_ubs["situacao"] == "Fluxo melhor").sum())
    ubs_prioritarias = resumo_ubs[resumo_ubs["situacao"].isin(["Crítica", "Atenção"])]
    nomes_prioritarios = _formatar_lista_curta(ubs_prioritarias["ubs"].tolist(), limite=3)

    render_metric_grid([
        responsive_metric_card("UBS com possível demora", str(n_criticas), "🚨", "Unidades em situação crítica para o cidadão", COLORS["red"]),
        responsive_metric_card("UBS que exigem atenção", str(n_atencao), "⚠️", "Pode haver espera moderada ou instabilidade", COLORS["amber"]),
        responsive_metric_card("UBS com fluxo melhor", str(n_ok), "✅", "Tendência de atendimento mais estável", COLORS["green"]),
        responsive_metric_card("Priorize informação", nomes_prioritarios, "📌", "Confirme horários antes de se deslocar", COLORS["blue"]),
    ], wide=True)


def render_tabela_publica_ubs(resumo_ubs, titulo="Situação das UBS em linguagem simples"):
    st.markdown(f"### {titulo}")

    if resumo_ubs is None or resumo_ubs.empty:
        st.warning("Ainda não há dados suficientes por UBS para exibir a situação pública.")
        return

    tabela = resumo_ubs[[
        "ubs", "situacao", "resumo", "orientacao", "pontos_que_afetam", "score_previsto", "metodo_previsao"
    ]].copy()
    tabela = tabela.rename(columns={
        "ubs": "UBS",
        "situacao": "Como está agora",
        "resumo": "O que pode acontecer",
        "orientacao": "O que fazer antes de ir",
        "pontos_que_afetam": "O que mais pesa no momento",
        "score_previsto": "Tendência do próximo período",
        "metodo_previsao": "Base da previsão",
    })
    tabela["Tendência do próximo período"] = tabela["Tendência do próximo período"].apply(lambda v: "sem previsão" if pd.isna(v) else f"{v:.1f}")

    st.dataframe(tabela, use_container_width=True, hide_index=True)


def render_cards_publicos_por_ubs(resumo_ubs):
    if resumo_ubs is None or resumo_ubs.empty:
        return

    st.markdown("### Orientação por UBS")
    for linha in _chunks(list(resumo_ubs.itertuples(index=False)), 2):
        cols = st.columns(len(linha))
        for col, row in zip(cols, linha):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {row.icone} {row.ubs}")
                    st.markdown(f"**Situação:** {row.situacao}")
                    st.write(row.resumo)
                    st.caption(f"O que fazer: {row.orientacao}")
                    if not pd.isna(row.score_previsto):
                        st.progress(int(max(0, min(100, row.score_previsto))))
                        st.caption(f"Tendência calculada: {row.score_previsto:.1f} | Base: {row.metodo_previsao}")
                    st.caption(f"Pontos que mais pesam: {row.pontos_que_afetam}")
        st.write("")


def render_grafico_publico_ubs(resumo_ubs):
    if resumo_ubs is None or resumo_ubs.empty:
        return
    fig = px.bar(
        resumo_ubs.sort_values("score_previsto", ascending=True),
        x="score_previsto",
        y="ubs",
        color="situacao",
        orientation="h",
        text="situacao",
        title="Tendência de funcionamento por UBS",
        labels={
            "score_previsto": "tendência do próximo período",
            "ubs": "UBS",
            "situacao": "situação"
        },
        color_discrete_map={
            "Crítica": COLORS["red"],
            "Atenção": COLORS["amber"],
            "Fluxo melhor": COLORS["green"],
        }
    )
    fig.update_traces(textposition="outside")
    fig.add_vrect(x0=0, x1=55, fillcolor="#FDECEC", opacity=0.18, line_width=0)
    fig.add_vrect(x0=55, x1=70, fillcolor="#FFF8E1", opacity=0.18, line_width=0)
    fig.add_vrect(x0=70, x1=100, fillcolor="#E8F5E9", opacity=0.18, line_width=0)
    fig = plotly_layout(fig, height=max(420, 70 * len(resumo_ubs)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def render_dimensoes_em_linguagem_cidada(dados_base):
    resumo_dim = montar_resumo_publico_dimensoes(dados_base)
    if resumo_dim.empty:
        return

    st.markdown("### O que pode estar afetando o atendimento")
    tabela = resumo_dim[["dimensao", "leitura", "o_que_significa_para_voce"]].copy()
    tabela = tabela.rename(columns={
        "dimensao": "Área observada",
        "leitura": "Leitura simples",
        "o_que_significa_para_voce": "O que isso pode significar para você",
    })
    st.dataframe(tabela, use_container_width=True, hide_index=True)


def render_portal_populacao_cidada(dados_base):
    resumo_ubs = montar_resumo_publico_ubs(dados_base)

    st.markdown("## 🏠 Orientações para o cidadão")
    st.write(
        "Aqui o APS-Twin traduz os resultados técnicos em informações úteis para quem usa a UBS: "
        "onde pode haver mais demora, quais unidades exigem atenção e o que fazer antes de procurar atendimento."
    )
    render_alerta_saude_publica()
    render_cards_publicos_resumo(resumo_ubs)

    criticas = resumo_ubs[resumo_ubs["situacao"] == "Crítica"] if not resumo_ubs.empty else pd.DataFrame()
    atencao_pub = resumo_ubs[resumo_ubs["situacao"] == "Atenção"] if not resumo_ubs.empty else pd.DataFrame()

    if not criticas.empty:
        st.warning(
            "Unidades com maior chance de lentidão: " +
            _formatar_lista_curta(criticas["ubs"].tolist(), limite=5) +
            ". A recomendação é confirmar atendimento, horários e agendamento antes de se deslocar."
        )
    elif not atencao_pub.empty:
        st.warning(
            "Algumas unidades estão em atenção: " +
            _formatar_lista_curta(atencao_pub["ubs"].tolist(), limite=5) +
            ". Pode haver oscilação no atendimento."
        )
    else:
        st.success("No recorte atual, as UBS apresentam tendência geral mais estável para atendimento de rotina.")

    render_grafico_publico_ubs(resumo_ubs)
    render_cards_publicos_por_ubs(resumo_ubs)
    render_dimensoes_em_linguagem_cidada(dados_base)


def render_situacao_ubs_populacao(dados_base):
    resumo_ubs = montar_resumo_publico_ubs(dados_base)
    st.markdown("## 🏥 Situação das UBS")
    st.write(
        "Esta visão mostra a situação das UBS em linguagem pública. O objetivo não é expor detalhes técnicos, "
        "mas ajudar a população a entender se uma unidade tende a estar mais sobrecarregada ou mais estável."
    )
    render_alerta_saude_publica()
    render_tabela_publica_ubs(resumo_ubs)
    render_cards_publicos_por_ubs(resumo_ubs)


def render_previsao_atendimento_populacao(dados_base):
    resumo_ubs = montar_resumo_publico_ubs(dados_base)
    st.markdown("## ⏱️ Previsão de atendimento")
    st.write(
        "A previsão abaixo é uma leitura aproximada dos modelos analíticos e da tendência recente. "
        "Ela indica chance de lentidão, não tempo exato de fila."
    )
    render_alerta_saude_publica()

    if resumo_ubs.empty:
        st.warning("Ainda não há série suficiente para gerar previsão pública por UBS.")
        return

    escolha = st.selectbox(
        "Escolha sua UBS",
        options=resumo_ubs["ubs"].tolist(),
        key="select_ubs_publica_previsao"
    )
    linha = resumo_ubs[resumo_ubs["ubs"] == escolha].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Situação", f"{linha['icone']} {linha['situacao']}")
    c2.metric("Tendência", f"{linha['score_previsto']:.1f}" if not pd.isna(linha['score_previsto']) else "sem previsão")
    c3.metric("Prioridade", linha["prioridade"])

    st.markdown(f"### O que isso quer dizer para quem vai à {escolha}")
    st.write(linha["resumo"])
    st.info(linha["orientacao"])
    st.caption(f"Base da previsão: {linha['metodo_previsao']}. Pontos que mais pesam: {linha['pontos_que_afetam']}.")

    st.markdown("### Recomendações práticas")
    if linha["situacao"] == "Crítica":
        st.markdown(
            "- Confirme se a unidade está atendendo antes de sair de casa.\n"
            "- Tente agendar ou verificar o melhor horário disponível.\n"
            "- Leve documento, cartão SUS, exames e receitas para evitar nova ida.\n"
            "- Para sintomas graves, procure urgência/emergência."
        )
    elif linha["situacao"] == "Atenção":
        st.markdown(
            "- Pode haver espera moderada; organize-se com antecedência.\n"
            "- Confira documentos e horários.\n"
            "- Mantenha acompanhamento de rotina, vacinação e consultas agendadas."
        )
    else:
        st.markdown(
            "- A unidade tende a estar mais estável para atendimento de rotina.\n"
            "- Mantenha consultas, vacinação, pré-natal e acompanhamento em dia.\n"
            "- Continue acompanhando avisos oficiais da unidade."
        )


def render_sobre_populacao():
    st.markdown("## ℹ️ Sobre esta informação")
    st.write(
        "O APS-Twin SMD organiza dados da Atenção Primária à Saúde para apoiar a gestão e também oferecer "
        "uma leitura pública simples. Para a população, o sistema evita termos técnicos e traduz os resultados "
        "em informações sobre possível demora, unidades em atenção, pontos que podem afetar o atendimento e "
        "orientações práticas antes de procurar a UBS."
    )
    st.warning(
        "As previsões são indicativas. Elas ajudam a orientar a procura por atendimento, mas não substituem "
        "a comunicação oficial da Secretaria de Saúde, da UBS ou a avaliação de um profissional de saúde."
    )


# =========================================================
# PERFIL PREFEITO: PAINEL ESTRATEGICO MUNICIPAL
# =========================================================

def obter_coluna_existente(df, opcoes):
    """Retorna a primeira coluna existente dentro de uma lista de nomes possiveis."""
    for col in opcoes:
        if col in df.columns:
            return col
    return None


def preparar_base_prefeito(dados_f):
    """Agrega dados por UBS e traduz em prioridade estrategica municipal."""
    if dados_f is None or dados_f.empty:
        return pd.DataFrame()

    df = dados_f.copy()
    col_ubs = obter_coluna_existente(df, ["ubs", "UBS", "unidade", "Unidade", "nome_ubs", "Nome UBS"])
    col_score = obter_coluna_existente(df, ["score", "Score", "score_medio", "indice", "indice_geral", "valor_score"])
    col_status = obter_coluna_existente(df, ["status", "Status", "classificacao", "Classificação"])
    col_dimensao = obter_coluna_existente(df, ["dimensao", "Dimensão", "dimension", "categoria"])
    col_mes = obter_coluna_existente(df, ["periodo_num", "mes", "mês", "Mes", "Mês", "data", "Data", "competencia", "Competência"])

    if col_ubs is None:
        df["ubs_temp_prefeito"] = "APS Municipal"
        col_ubs = "ubs_temp_prefeito"

    if col_score is None:
        df["score_temp_prefeito"] = 0
        col_score = "score_temp_prefeito"

    df[col_ubs] = df[col_ubs].apply(normalizar_nome_ubs)
    df = df[df[col_ubs].apply(_validar_ubs_publica)].copy()
    df_ubs = df[~df[col_ubs].astype(str).str.lower().isin(["aps municipal", "municipal", "todas"])]
    if not df_ubs.empty:
        df = df_ubs.copy()

    df[col_score] = pd.to_numeric(df[col_score], errors="coerce")
    linhas = []

    for ubs, grupo in df.groupby(col_ubs):
        total = len(grupo)
        if total == 0:
            continue

        score_medio = grupo[col_score].mean()
        score_medio = 0 if pd.isna(score_medio) else float(score_medio)

        if col_status:
            status_normalizado = grupo[col_status].astype(str).str.lower()
            criticos = status_normalizado.str.contains("crítico|critico|baixo|ruim|vermelho", regex=True, na=False).sum()
            atencao = status_normalizado.str.contains("atenção|atencao|moderado|amarelo", regex=True, na=False).sum()
            adequados = status_normalizado.str.contains("adequado|bom|verde|satisfatório|satisfatorio", regex=True, na=False).sum()
        else:
            criticos = (grupo[col_score] < 55).sum()
            atencao = ((grupo[col_score] >= 55) & (grupo[col_score] < 70)).sum()
            adequados = (grupo[col_score] >= 70).sum()

        pct_critico = (criticos / max(1, total)) * 100
        pct_adequado = (adequados / max(1, total)) * 100

        tendencia = _calcular_tendencia_por_grupo(grupo) if "periodo_num" in grupo.columns else 0
        if tendencia == 0 and col_mes and col_mes in grupo.columns and grupo[col_mes].nunique() >= 2:
            try:
                serie = grupo.groupby(col_mes)[col_score].mean().reset_index().sort_values(col_mes)
                if len(serie) >= 2:
                    tendencia = float(serie[col_score].iloc[-1] - serie[col_score].iloc[0])
            except Exception:
                tendencia = 0

        score_previsto, metodo_previsao = _prever_score_publico(grupo)

        dimensao_prioritaria = "Não identificada"
        if col_dimensao:
            try:
                dim = grupo.groupby(col_dimensao)[col_score].mean().sort_values()
                if len(dim) > 0:
                    dimensao_prioritaria = str(dim.index[0])
            except Exception:
                dimensao_prioritaria = "Não identificada"

        linhas.append({
            "ubs": ubs,
            "score_medio": round(score_medio, 1),
            "score_previsto": np.nan if pd.isna(score_previsto) else round(float(score_previsto), 1),
            "metodo_previsao": metodo_previsao,
            "registros": int(total),
            "criticos": int(criticos),
            "atencao": int(atencao),
            "adequados": int(adequados),
            "pct_critico": round(float(pct_critico), 1),
            "pct_adequado": round(float(pct_adequado), 1),
            "tendencia": round(float(tendencia), 1),
            "dimensao_prioritaria": dimensao_prioritaria,
        })

    base = pd.DataFrame(linhas)
    if base.empty:
        return base

    base["indice_prioridade"] = (
        (100 - base["score_medio"]) * 0.45
        + base["pct_critico"] * 0.35
        + base["registros"].rank(pct=True) * 10
        + base["tendencia"].apply(lambda x: 10 if x < -5 else 0)
    ).round(1)

    return base.sort_values(["indice_prioridade", "pct_critico"], ascending=False)


def classificar_ubs_prefeito(row):
    """Traduz score e criticidade em linguagem de decisao para o prefeito."""
    score = row.get("score_previsto", np.nan)
    if pd.isna(score):
        score = row.get("score_medio", 0)
    pct_critico = row.get("pct_critico", 0)
    tendencia = row.get("tendencia", 0)

    score = 0 if pd.isna(score) else float(score)
    pct_critico = 0 if pd.isna(pct_critico) else float(pct_critico)
    tendencia = 0 if pd.isna(tendencia) else float(tendencia)

    if score < 50 or pct_critico >= 55:
        return {
            "situacao": "Crítica",
            "icone": "🔴",
            "risco_populacao": "Alto risco de demora ou dificuldade no atendimento",
            "prioridade": "Prioridade máxima",
            "acao": "Solicitar resposta imediata da Secretaria de Saúde e verificar equipe, agenda e fluxo de acolhimento.",
        }
    if score < 65 or pct_critico >= 35 or tendencia < -5:
        return {
            "situacao": "Atenção",
            "icone": "🟡",
            "risco_populacao": "Pode haver espera moderada ou instabilidade no atendimento",
            "prioridade": "Acompanhar nesta semana",
            "acao": "Monitorar a unidade, revisar indicadores frágeis e prevenir agravamento.",
        }
    return {
        "situacao": "Estável",
        "icone": "🟢",
        "risco_populacao": "Menor risco de demora no momento",
        "prioridade": "Manter acompanhamento",
        "acao": "Manter rotina de monitoramento e usar boas práticas como referência.",
    }


def render_kpis_prefeito(base_prefeito):
    if base_prefeito.empty:
        st.warning("Não há dados suficientes para gerar o painel do prefeito.")
        return

    classificacoes = base_prefeito.apply(classificar_ubs_prefeito, axis=1).tolist()
    total_criticas = sum(c["situacao"] == "Crítica" for c in classificacoes)
    total_atencao = sum(c["situacao"] == "Atenção" for c in classificacoes)
    score_municipal = base_prefeito["score_medio"].mean()
    pct_critico_municipal = base_prefeito["pct_critico"].mean()

    if score_municipal < 50 or pct_critico_municipal >= 55:
        situacao_geral, risco_demora = "Crítica", "Alto"
    elif score_municipal < 65 or pct_critico_municipal >= 35:
        situacao_geral, risco_demora = "Atenção", "Moderado"
    else:
        situacao_geral, risco_demora = "Estável", "Baixo"

    prioridade_mes = str(base_prefeito.iloc[0].get("dimensao_prioritaria", "Não identificada"))

    render_metric_grid([
        responsive_metric_card("Situação da APS", situacao_geral, "🏛️", "Leitura estratégica municipal", COLORS["blue"]),
        responsive_metric_card("UBS críticas", str(int(total_criticas)), "🔴", "Exigem ação imediata", COLORS["red"]),
        responsive_metric_card("UBS em atenção", str(int(total_atencao)), "🟡", "Prevenir agravamento", COLORS["amber"]),
        responsive_metric_card("Risco de demora", risco_demora, "⏱️", "Pressão percebida pela população", COLORS["orange"]),
        responsive_metric_card("Prioridade do mês", prioridade_mes, "📌", "Dimensão mais frágil", COLORS["teal"]),
    ], wide=True)


def render_mapa_prioridade_prefeito(base_prefeito):
    st.subheader("🚦 Semáforo estratégico das UBS")
    st.caption("Classificação territorial para priorização do governo municipal.")
    if base_prefeito.empty:
        st.info("Sem dados de UBS para exibir.")
        return

    grupos = {"Crítica": [], "Atenção": [], "Estável": []}
    for _, row in base_prefeito.iterrows():
        c = classificar_ubs_prefeito(row)
        grupos[c["situacao"]].append(f"{c['icone']} {row['ubs']} — score {row['score_medio']:.1f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("🔴 Prioridade máxima")
        for item in grupos["Crítica"] or ["Nenhuma UBS crítica no recorte atual."]:
            st.write(item)
    with col2:
        st.warning("🟡 Acompanhar")
        for item in grupos["Atenção"] or ["Nenhuma UBS em atenção no recorte atual."]:
            st.write(item)
    with col3:
        st.success("🟢 Estável")
        for item in grupos["Estável"] or ["Nenhuma UBS classificada como estável no recorte atual."]:
            st.write(item)


def render_prioridade_ubs_prefeito(base_prefeito):
    st.subheader("🏥 Prioridade territorial das UBS")
    st.caption("Onde a gestão municipal deve priorizar atenção, considerando desempenho, criticidade, tendência e volume de registros.")
    if base_prefeito.empty:
        st.info("Sem dados de UBS para exibir.")
        return

    tabela = base_prefeito.copy()
    classificacoes = tabela.apply(classificar_ubs_prefeito, axis=1).tolist()
    tabela["situação"] = [c["icone"] + " " + c["situacao"] for c in classificacoes]
    tabela["risco para população"] = [c["risco_populacao"] for c in classificacoes]
    tabela["prioridade"] = [c["prioridade"] for c in classificacoes]
    tabela["ação recomendada"] = [c["acao"] for c in classificacoes]

    tabela_final = tabela[[
        "ubs", "situação", "score_medio", "score_previsto", "pct_critico", "tendencia",
        "dimensao_prioritaria", "prioridade", "risco para população", "ação recomendada"
    ]].rename(columns={
        "ubs": "UBS",
        "score_medio": "Score atual",
        "score_previsto": "Tendência prevista",
        "pct_critico": "% crítico",
        "tendencia": "Variação recente",
        "dimensao_prioritaria": "Dimensão mais frágil",
    })
    st.dataframe(tabela_final, use_container_width=True, hide_index=True)


def render_risco_populacao_prefeito(base_prefeito):
    st.subheader("👥 O que pode impactar a população")
    if base_prefeito.empty:
        st.info("Sem dados suficientes para traduzir risco populacional.")
        return

    top = base_prefeito.head(3)
    for _, row in top.iterrows():
        c = classificar_ubs_prefeito(row)
        with st.container(border=True):
            st.markdown(f"### {c['icone']} {row['ubs']}")
            st.write(f"**Situação:** {c['situacao']}")
            st.write(f"**Risco para a população:** {c['risco_populacao']}")
            st.write(f"**Principal fragilidade:** {row['dimensao_prioritaria']}")
            st.write(f"**Ação recomendada:** {c['acao']}")
            if not pd.isna(row.get("score_previsto", np.nan)):
                st.caption(f"Tendência prevista: {row['score_previsto']:.1f} | Base: {row.get('metodo_previsao', 'modelo/tendência')}")


def render_cenarios_decisao_prefeito(base_prefeito):
    st.subheader("🧭 Cenários e decisões recomendadas")
    if base_prefeito.empty:
        st.info("Sem dados para gerar cenários.")
        return

    ubs_prioritaria = base_prefeito.iloc[0]
    c = classificar_ubs_prefeito(ubs_prioritaria)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### Cenário 1: Sem intervenção")
            st.write(f"A UBS **{ubs_prioritaria['ubs']}** tende a continuar como prioridade de risco se não houver ação gerencial.")
            st.warning("Não recomendado para unidades críticas.")
    with col2:
        with st.container(border=True):
            st.markdown("### Cenário 2: Ação moderada")
            st.write("Monitoramento semanal, revisão de agenda e acompanhamento dos indicadores mais frágeis.")
            st.info("Pode reduzir o risco, mas exige acompanhamento.")
    with col3:
        with st.container(border=True):
            st.markdown("### Cenário 3: Ação prioritária")
            st.write("Reunião com a Secretaria de Saúde, plano de resposta da UBS e acompanhamento direto da gestão.")
            st.success("Recomendado para prioridade máxima.")

    st.markdown("#### Decisão sugerida para o prefeito")
    st.write(
        f"Priorizar a UBS **{ubs_prioritaria['ubs']}**, classificada como **{c['situacao']}**, "
        f"com atenção especial à dimensão **{ubs_prioritaria['dimensao_prioritaria']}**."
    )


def render_relatorio_governo_prefeito(base_prefeito):
    st.subheader("📄 Resumo para reunião de governo")
    if base_prefeito.empty:
        st.info("Sem dados para gerar resumo de governo.")
        return

    total_ubs = base_prefeito["ubs"].nunique()
    classificacoes = base_prefeito.apply(classificar_ubs_prefeito, axis=1).tolist()
    total_criticas = sum(c["situacao"] == "Crítica" for c in classificacoes)
    total_atencao = sum(c["situacao"] == "Atenção" for c in classificacoes)
    pior = base_prefeito.iloc[0]
    melhor = base_prefeito.sort_values("score_medio", ascending=False).iloc[0]

    texto = f"""
A Atenção Primária à Saúde do município apresenta **{total_ubs} UBS monitoradas** no recorte atual.

Foram identificadas **{total_criticas} UBS em situação crítica** e **{total_atencao} UBS em atenção**.

A principal prioridade territorial no momento é a **UBS {pior['ubs']}**, com score médio de **{pior['score_medio']}**, tendência prevista de **{pior['score_previsto']}**, percentual crítico de **{pior['pct_critico']}%** e principal fragilidade associada à dimensão **{pior['dimensao_prioritaria']}**.

A unidade com melhor desempenho relativo é **{melhor['ubs']}**, com score médio de **{melhor['score_medio']}**.

Recomenda-se solicitar à Secretaria Municipal de Saúde um plano de resposta para as UBS em situação crítica, com foco em fluxo de atendimento, equipe disponível, agenda, qualidade assistencial e condições de trabalho.

Este painel não substitui a análise técnica da Secretaria, mas orienta a decisão estratégica do governo municipal.
"""
    st.markdown(texto)
    st.download_button(
        label="Baixar resumo da reunião em TXT",
        data=texto,
        file_name="resumo_prefeito_aps_twin.txt",
        mime="text/plain",
    )


def render_sobre_prefeito():
    st.markdown("## ℹ️ Sobre o painel do prefeito")
    st.write(
        "Este perfil traduz os dados do APS-Twin SMD em prioridades estratégicas para o governo municipal. "
        "O objetivo não é expor detalhes técnicos de modelo, API ou base de dados, mas indicar onde priorizar, "
        "quais UBS podem gerar maior pressão para a população e quais decisões devem ser cobradas da Secretaria."
    )
    st.warning(
        "As previsões são indicativas e dependem da qualidade dos dados disponíveis. A decisão final deve considerar "
        "a análise técnica da Secretaria Municipal de Saúde, informações de campo e contexto político-administrativo."
    )


def render_layout_prefeito(dados_base):
    st.markdown("## 🏛️ Painel Estratégico do Prefeito")
    st.caption("Tradução dos dados da Atenção Primária em prioridade territorial, risco para a população e decisão de governo.")

    base_prefeito = preparar_base_prefeito(dados_base)
    render_kpis_prefeito(base_prefeito)
    st.divider()

    aba = st.session_state.get("aba_ativa", "🏛️ Visão do Prefeito")
    if aba == "🏛️ Visão do Prefeito":
        render_mapa_prioridade_prefeito(base_prefeito)
    elif aba == "🏥 Prioridade das UBS":
        render_prioridade_ubs_prefeito(base_prefeito)
    elif aba == "👥 Risco para a população":
        render_risco_populacao_prefeito(base_prefeito)
    elif aba == "🧭 Cenários e decisões":
        render_cenarios_decisao_prefeito(base_prefeito)
    elif aba == "📄 Resumo de governo":
        render_relatorio_governo_prefeito(base_prefeito)
    elif aba == "ℹ️ Sobre o SMD":
        render_sobre_prefeito()
    else:
        render_mapa_prioridade_prefeito(base_prefeito)

def render_topo_por_perfil(
    dados_base,
    indice_geral,
    sistematicidade,
    criticos,
    atencao,
    adequados,
    qualidade_dados,
    maturidade_dt,
    risco_gerencial,
    classe_risco,
    ranking_ubs,
    ubs_critica,
    ubs_melhor,
    api_ativa,
):
    total_status = max(1, criticos + atencao + adequados)
    pct_adequados = 100 * adequados / total_status
    perfil = perfil_atual()

    if perfil == "prefeito":
        st.markdown(
            """
            <div class="public-hero">
                <div class="public-hero-title">Painel Estratégico do Prefeito</div>
                <div class="public-hero-text">Prioridades de governo para a Atenção Primária: UBS que exigem ação, risco para a população, cenários de decisão e resumo para reunião municipal.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    if perfil == "populacao":
        resumo_ubs = montar_resumo_publico_ubs(dados_base)
        st.markdown(
            """
            <div class="public-hero">
                <div class="public-hero-title">Informação para quem usa a UBS</div>
                <div class="public-hero-text">Em vez de indicadores técnicos, esta área traduz o SMD em orientação prática: possível demora, UBS em atenção, tendência de atendimento e o que fazer antes de procurar a unidade.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_cards_publicos_resumo(resumo_ubs)
        return

    if perfil == "gerente":
        ubs_nome = st.session_state.get("ubs_usuario", "Minha UBS")
        st.markdown(f'<div class="section-title">Painel da UBS {html_lib.escape(str(ubs_nome))}</div>', unsafe_allow_html=True)
        render_metric_grid([
            responsive_metric_card("Score da UBS", f"{indice_geral:.1f}", "🏥", "Desempenho médio do recorte", COLORS["blue"]),
            responsive_metric_card("Indicadores críticos", f"{criticos}", "🚨", "Prioridade operacional", COLORS["red"]),
            responsive_metric_card("Em atenção", f"{atencao}", "⚠️", "Acompanhar tendência", COLORS["amber"]),
            responsive_metric_card("Adequados", f"{adequados}", "✅", "Registros satisfatórios", COLORS["green"]),
        ])
        st.markdown('<div class="section-title">Síntese operacional da unidade</div>', unsafe_allow_html=True)
        dec1, dec2 = st.columns(2)
        with dec1:
            exibir_decision_card(
                "O que olhar primeiro",
                f"A UBS apresenta {criticos} registros críticos e {atencao} em atenção no recorte atual.",
                "Alto" if criticos > 0 else "Moderado"
            )
        with dec2:
            exibir_decision_card(
                "Plano de acompanhamento",
                "Priorize os indicadores críticos, monitore a tendência mensal e registre ações corretivas para comparação futura.",
                "Moderado"
            )
        return

    if perfil == "secretario":
        st.markdown('<div class="section-title">Centro de comando executivo da Secretaria</div>', unsafe_allow_html=True)
        render_metric_grid([
            responsive_metric_card("Índice Geral", f"{indice_geral:.1f}", "📈", "Score consolidado municipal", COLORS["blue"]),
            responsive_metric_card("Risco Gerencial", f"{risco_gerencial:.1f}", "🚦", classe_risco, COLORS["red"] if classe_risco in ["Crítico", "Alto"] else COLORS["amber"]),
            responsive_metric_card("UBS crítica", ubs_critica, "🏥", "Menor desempenho médio", COLORS["red"]),
            responsive_metric_card("Adequação", f"{pct_adequados:.1f}%", "✅", "Percentual adequado", COLORS["green"]),
        ])
        dec1, dec2, dec3 = st.columns(3)
        with dec1:
            exibir_decision_card(
                "Prioridade municipal",
                f"O recorte atual está classificado como {classe_risco}. A menor média aparece em {ubs_critica}.",
                classe_risco
            )
        with dec2:
            exibir_decision_card(
                "Qualidade informacional",
                f"A qualidade geral dos dados está em {qualidade_dados['qualidade_geral']:.1f}.",
                "Baixo" if qualidade_dados["qualidade_geral"] >= 85 else "Moderado"
            )
        with dec3:
            exibir_decision_card(
                "Apoio à decisão",
                f"Digital Twin com maturidade {maturidade_dt['maturidade_dt']:.1f}. API {'ativa' if api_ativa else 'parcial/inativa'}.",
                "Moderado"
            )
        render_ubs_grid_responsivo(dados_base, titulo="🏥 Unidades priorizadas", permitir_abertura=True)
        return

    if perfil == "aluno":
        st.markdown('<div class="section-title">Laboratório científico do APS-Twin SMD</div>', unsafe_allow_html=True)
        render_metric_grid([
            responsive_metric_card("Índice SMD", f"{indice_geral:.1f}", "📊", "Base para análise longitudinal", COLORS["blue"]),
            responsive_metric_card("Sistematicidade", f"{sistematicidade:.1f}" if not pd.isna(sistematicidade) else "N/A", "🔗", "Interdependência entre indicadores", COLORS["teal"]),
            responsive_metric_card("Qualidade dos dados", f"{qualidade_dados['qualidade_geral']:.1f}", "🧪", "Robustez para ML/DT", COLORS["green"]),
            responsive_metric_card("Maturidade DT", f"{maturidade_dt['maturidade_dt']:.1f}", "🧠", "Dados + ML + simulação", COLORS["navy"]),
        ])
        return

    # ADM e perfil padrao: visao completa
    st.markdown('<div class="section-title">Painel completo técnico-administrativo</div>', unsafe_allow_html=True)
    render_metric_grid([
        responsive_metric_card("Índice Geral", f"{indice_geral:.1f}", "📈", "Score consolidado", COLORS["blue"]),
        responsive_metric_card("Sistematicidade", f"{sistematicidade:.1f}" if not pd.isna(sistematicidade) else "N/A", "🔗", "Integração dos indicadores", COLORS["teal"]),
        responsive_metric_card("Qualidade dos Dados", f"{qualidade_dados['qualidade_geral']:.1f}", "🧪", "Completude e consistência", COLORS["green"]),
        responsive_metric_card("Maturidade DT", f"{maturidade_dt['maturidade_dt']:.1f}", "🧠", "Dados + API + ML + simulação", COLORS["navy"]),
        responsive_metric_card("Risco Gerencial", f"{risco_gerencial:.1f}", "🚦", classe_risco, COLORS["red"] if classe_risco in ["Crítico", "Alto"] else COLORS["amber"]),
        responsive_metric_card("Críticos", f"{criticos}", "🚨", "Registros em risco", COLORS["red"]),
        responsive_metric_card("Adequados", f"{adequados}", "✅", "Registros satisfatórios", COLORS["green"]),
        responsive_metric_card("Indicadores", f"{dados_base['indicador'].nunique()}", "🧩", "Dimensões monitoradas", COLORS["blue2"]),
    ])

    fontes_ativas = ", ".join(sorted(dados_base["fonte"].dropna().unique())) if "fonte" in dados_base.columns else "-"
    st.caption(
        f"Menor desempenho: {ubs_critica} | Melhor desempenho: {ubs_melhor} | Fontes ativas: {fontes_ativas}"
    )

    st.markdown('<div class="section-title">Síntese executiva para decisão</div>', unsafe_allow_html=True)
    dec1, dec2, dec3 = st.columns(3)
    with dec1:
        exibir_decision_card(
            "Prioridade gerencial",
            f"O recorte atual apresenta risco gerencial classificado como {classe_risco}. A unidade de menor desempenho é {ubs_critica}.",
            classe_risco
        )
    with dec2:
        prioridade_dados = "Baixo" if qualidade_dados["qualidade_geral"] >= 85 else ("Moderado" if qualidade_dados["qualidade_geral"] >= 70 else "Alto")
        exibir_decision_card(
            "Qualidade dos dados",
            f"A qualidade geral dos dados está em {qualidade_dados['qualidade_geral']:.1f}. O sistema reconhece {qualidade_dados['n_indicadores']} indicadores no recorte ativo.",
            prioridade_dados
        )
    with dec3:
        prioridade_dt = "Baixo" if maturidade_dt["maturidade_dt"] >= 75 else ("Moderado" if maturidade_dt["maturidade_dt"] >= 55 else "Alto")
        exibir_decision_card(
            "Maturidade do Digital Twin",
            f"A maturidade DT está em {maturidade_dt['maturidade_dt']:.1f}. A integração por API está {'ativa' if api_ativa else 'parcial/inativa'} neste recorte.",
            prioridade_dt
        )

    render_ubs_grid_responsivo(dados_base, titulo="🏥 Mapa executivo das UBS", permitir_abertura=True)



# =========================================================
# VERIFICAÇÃO DE LOGIN
# =========================================================

verificar_login()


# =========================================================
# CARREGAR DADOS
# =========================================================

dados, dic, arquivo_municipal, arquivo_ubs = preparar_dados()

if dados is None:
    st.error("Não foi possível preparar os dados.")
    st.write("Verifique se a pasta de dados contém pelo menos estes arquivos:")
    st.code(f"""
Pasta base do app:
{BASE_DIR}

Pasta de dados detectada:
{DATA_DIR}

Arquivos esperados:
{DATA_DIR}/dictionary_indicators.csv
{DATA_DIR}/indicadores_aps.xlsx
{DATA_DIR}/indicadores_ubs.xlsx

Também são aceitas as pastas:
- dados
- data
""")

    st.write("Arquivos encontrados na pasta do projeto:")
    try:
        encontrados = []
        for p in BASE_DIR.rglob("*"):
            if p.is_file():
                encontrados.append(str(p.relative_to(BASE_DIR)))
        st.code("\n".join(encontrados[:80]) if encontrados else "Nenhum arquivo listado.")
    except Exception as e:
        st.caption(str(e))

    st.stop()



MODULOS_APS = [
    "📊 Painel Executivo",
    "🧪 Qualidade dos Dados",
    "🎯 18 Indicadores",
    "🏥 UBS e Território",
    "📈 Séries Temporais",
    "🤖 Machine Learning",
    "🧠 Digital Twin Ciberfísico",
    "🕸️ Sistematicidade",
    "📄 Relatórios",
    "ℹ️ Sobre o SMD"
]

modulos_disponiveis = modulos_por_perfil()

if "aba_ativa" not in st.session_state or st.session_state["aba_ativa"] not in modulos_disponiveis:
    st.session_state["aba_ativa"] = modulo_inicial_por_perfil()
    if st.session_state["aba_ativa"] not in modulos_disponiveis:
        st.session_state["aba_ativa"] = modulos_disponiveis[0]

# =========================================================
# CABEÇALHO RESPONSIVO
# =========================================================

botao_logout()
render_header_principal()
inject_responsive_css()

st.caption(
    f"Usuário: {st.session_state.get('nome_usuario', '')} | "
    f"Perfil: {st.session_state.get('perfil_usuario', '').upper()} | "
    f"Escopo: {st.session_state.get('ubs_usuario', 'Todas')}"
)

render_nav_responsiva(modulos_disponiveis)

if usuario_pode_admin():
    with st.expander("Arquivos e bases utilizadas"):
        st.write("Base municipal detectada:", arquivo_municipal)
        st.write("Base UBS detectada:", arquivo_ubs)
        st.write("Dicionário de indicadores:", "dictionary_indicators.csv")
        st.write("Imagem de login:", str(LOGO_PATH) if LOGO_PATH else "Nenhuma imagem encontrada em assets")
        st.write("Logo UNIVASF:", str(UNIVASF_LOGO_PATH) if UNIVASF_LOGO_PATH else "Adicionar em assets/logo_univasf.png")
        st.write("Logo UFBA:", str(UFBA_LOGO_PATH) if UFBA_LOGO_PATH else "Adicionar em assets/logo_ufba.png")
        st.write("Pasta do projeto:", str(BASE_DIR))


# =========================================================
# ESTADO DA UBS SELECIONADA
# =========================================================

if "ubs_selecionada_card" not in st.session_state:
    st.session_state["ubs_selecionada_card"] = "Todas"


# =========================================================
# SIDEBAR E FILTROS
# =========================================================

dados_api_status = None

if usuario_pode_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-title">APS-Twin SMD</div>
            <div class="sidebar-brand-subtitle">Central de filtros, fontes de dados e integração em tempo quase real.</div>
            <div class="sidebar-brand-version">v1.9 institucional</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.title("Filtros do APS-Twin")

    with st.sidebar.expander("🔌 API Google Forms/Sheets em tempo real", expanded=False):
        st.caption(
            "Cole o link da planilha de respostas do Google Forms. "
            "Quando ativada, a planilha passa a alimentar o dashboard, os filtros, os gráficos, o ML e o Digital Twin."
        )

        google_url = st.text_input(
            "Link da planilha Google Sheets",
            value=st.session_state.get("google_sheet_url", ""),
            placeholder="https://docs.google.com/spreadsheets/d/..."
        )

        modo_api = st.radio(
            "Como usar os dados da API?",
            options=[
                "Somente bases locais",
                "Local + Google Sheets",
                "Somente Google Sheets"
            ],
            index=1,
            help="Use 'Local + Google Sheets' para somar os dados da tese com os dados atualizados do formulário."
        )

        atualizar_tempo_real = st.checkbox(
            "Atualizar automaticamente a cada execução (cache de 15 segundos)",
            value=True
        )

        st.session_state["google_sheet_url"] = google_url
        st.session_state["google_sheet_mode"] = modo_api
        st.session_state["google_sheet_realtime"] = atualizar_tempo_real

        if st.button("Testar conexão API", use_container_width=True):
            try:
                df_google, csv_url_google = carregar_google_sheet_csv_realtime(google_url)
                if df_google is not None:
                    st.session_state["google_sheet_ok"] = True
                    st.session_state["google_sheet_rows"] = len(df_google)
                    st.session_state["google_sheet_cols"] = len(df_google.columns)
                    st.session_state["google_sheet_preview"] = df_google.head(5)
                    st.session_state["google_sheet_csv_url"] = csv_url_google
                    st.success(f"Conexão realizada: {len(df_google)} linhas e {len(df_google.columns)} colunas.")
                    st.caption("Endpoint CSV utilizado:")
                    st.code(csv_url_google)
                else:
                    st.warning("Informe um link válido.")
            except Exception as e:
                st.session_state["google_sheet_ok"] = False
                st.error("Falha ao conectar. Verifique se a planilha está compartilhada/publicada.")
                st.caption(str(e))

        if st.session_state.get("google_sheet_ok"):
            st.success(
                f"API ativa: {st.session_state.get('google_sheet_rows')} linhas | "
                f"{st.session_state.get('google_sheet_cols')} colunas"
            )
            with st.expander("Prévia dos dados recebidos"):
                st.caption("Endpoint CSV ativo:")
                st.code(st.session_state.get("google_sheet_csv_url", ""))
                st.dataframe(st.session_state.get("google_sheet_preview"), use_container_width=True)

    # Incorporação dos dados da API apenas para ADM/Aluno.
    if st.session_state.get("google_sheet_mode", "Local + Google Sheets") != "Somente bases locais":
        url_api = st.session_state.get("google_sheet_url", "")

        if url_api:
            try:
                df_google_runtime, csv_url_runtime = carregar_google_sheet_csv_realtime(url_api)
                dados_google = preparar_google_sheet_para_smd(df_google_runtime, dic)

                if not dados_google.empty:
                    st.session_state["google_sheet_ok"] = True
                    st.session_state["google_sheet_rows"] = len(df_google_runtime)
                    st.session_state["google_sheet_cols"] = len(df_google_runtime.columns)
                    st.session_state["google_sheet_preview"] = df_google_runtime.head(5)
                    st.session_state["google_sheet_csv_url"] = csv_url_runtime

                    if st.session_state.get("google_sheet_mode") == "Somente Google Sheets":
                        dados = dados_google.copy()
                        dados_api_status = f"Dashboard alimentado somente pela API Google Sheets: {len(dados_google)} registros tratados."
                    else:
                        dados = pd.concat([dados, dados_google], ignore_index=True)
                        dados_api_status = f"Dashboard alimentado por bases locais + API Google Sheets: {len(dados_google)} registros tratados da API."
            except Exception as e:
                dados_api_status = f"API configurada, mas não foi possível alimentar o dashboard: {e}"

    if dados_api_status:
        if "não foi possível" in dados_api_status.lower():
            st.sidebar.warning(dados_api_status)
        else:
            st.sidebar.success(dados_api_status)

    fontes_lista = sorted(dados["fonte"].dropna().unique())
    fonte_sel = st.sidebar.multiselect("Fonte dos dados", options=fontes_lista, default=fontes_lista)

    dimensoes = sorted(dados["dimensao"].dropna().unique())
    dim_sel = st.sidebar.multiselect("Dimensão do SMD", options=dimensoes, default=dimensoes)

    niveis = sorted(dados["nivel"].dropna().unique())
    nivel_sel = st.sidebar.multiselect("Nível", options=niveis, default=niveis)

    dados["ubs"] = dados["ubs"].apply(normalizar_nome_ubs)
    ubs_lista = sorted(set(dados["ubs"].dropna().astype(str).tolist()))
    ubs_sel = st.sidebar.multiselect("UBS / nível territorial", options=ubs_lista, default=ubs_lista)

    meses_lista = ordenar_meses_lista(dados["mes"].astype(str).dropna().unique())
    mes_sel = st.sidebar.multiselect("Mês/período", options=meses_lista, default=meses_lista)

    indicadores_lista = sorted(dados["indicador"].dropna().unique())
    indicador_sel = st.sidebar.multiselect("Indicadores", options=indicadores_lista, default=indicadores_lista)

else:
    # Sem sidebar administrativa para Secretário, Gerente e População.
    st.sidebar.empty()

    fonte_sel = sorted(dados["fonte"].dropna().unique()) if "fonte" in dados.columns else []
    dim_sel = sorted(dados["dimensao"].dropna().unique()) if "dimensao" in dados.columns else []
    nivel_sel = sorted(dados["nivel"].dropna().unique()) if "nivel" in dados.columns else []
    dados["ubs"] = dados["ubs"].apply(normalizar_nome_ubs)
    meses_lista = ordenar_meses_lista(dados["mes"].astype(str).dropna().unique()) if "mes" in dados.columns else []
    mes_sel = meses_lista
    indicador_sel = sorted(dados["indicador"].dropna().unique()) if "indicador" in dados.columns else []

    if usuario_e_gerente():
        ubs_sel = [normalizar_nome_ubs(st.session_state.get("ubs_usuario"))]
    elif usuario_e_publico():
        # A população precisa enxergar a situação das UBS em linguagem simples.
        # Por isso, não restringimos ao agregado municipal; o filtro mantém todas
        # as unidades públicas disponíveis, exceto valores vazios/administrativos.
        ubs_disponiveis = sorted(set(dados["ubs"].dropna().astype(str).tolist()))
        ubs_sel = [
            u for u in ubs_disponiveis
            if str(u).strip()
            and str(u).lower() not in ["todas", "nan", "none"]
        ]
    else:
        ubs_sel = sorted(set(dados["ubs"].dropna().astype(str).tolist()))

dados_f = dados[
    (dados["fonte"].isin(fonte_sel)) &
    (dados["dimensao"].isin(dim_sel)) &
    (dados["nivel"].isin(nivel_sel)) &
    (dados["ubs"].isin(ubs_sel)) &
    (dados["mes"].astype(str).isin(mes_sel)) &
    (dados["indicador"].isin(indicador_sel))
].copy()

dados_f = aplicar_restricao_de_perfil(dados_f)

if st.session_state["ubs_selecionada_card"] != "Todas":
    dados_f = dados_f[dados_f["ubs"] == st.session_state["ubs_selecionada_card"]].copy()

if dados_f.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados ou para o perfil de acesso atual.")
    st.stop()


# =========================================================
# INDICADORES EXECUTIVOS
# =========================================================

indice_geral = dados_f["score"].mean()
sistematicidade, matriz_corr, conectividade = calcular_sistematicidade(dados_f)

criticos = (dados_f["status"] == "Crítico").sum()
atencao = (dados_f["status"] == "Atenção").sum()
adequados = (dados_f["status"] == "Adequado").sum()

qualidade_dados = calcular_qualidade_dados(dados_f, dic)
api_ativa = "API Google Forms/Sheets" in dados_f["fonte"].astype(str).unique()
maturidade_dt = calcular_maturidade_dt(
    dados_f,
    sistematicidade,
    api_ativa=api_ativa,
    ml_ativo=True,
    simulacao_ativa=True
)
risco_gerencial, classe_risco = calcular_risco_gerencial(
    dados_f,
    qualidade_dados["qualidade_geral"],
    sistematicidade
)

ranking_ubs = (
    dados_f
    .groupby("ubs")["score"]
    .mean()
    .reset_index()
    .sort_values("score")
)

ubs_critica = ranking_ubs.iloc[0]["ubs"] if not ranking_ubs.empty else "-"
ubs_melhor = ranking_ubs.iloc[-1]["ubs"] if not ranking_ubs.empty else "-"

render_topo_por_perfil(
    dados_base=dados_f,
    indice_geral=indice_geral,
    sistematicidade=sistematicidade,
    criticos=criticos,
    atencao=atencao,
    adequados=adequados,
    qualidade_dados=qualidade_dados,
    maturidade_dt=maturidade_dt,
    risco_gerencial=risco_gerencial,
    classe_risco=classe_risco,
    ranking_ubs=ranking_ubs,
    ubs_critica=ubs_critica,
    ubs_melhor=ubs_melhor,
    api_ativa=api_ativa,
)

st.divider()

# =========================================================
# ROTEIRO EXCLUSIVO DO PERFIL PREFEITO
# =========================================================

if usuario_e_prefeito():
    render_layout_prefeito(dados_f)
    st.divider()
    st.caption(
        "APS-Twin SMD | Perfil prefeito | Painel estratégico municipal com previsões orientativas e apoio à decisão de governo."
    )
    st.stop()

# =========================================================
# ROTEIRO EXCLUSIVO DO PERFIL POPULAÇÃO
# =========================================================

if usuario_e_publico():
    aba_publica = st.session_state.get("aba_ativa", "🏠 Orientações para o cidadão")

    if aba_publica == "🏠 Orientações para o cidadão":
        render_portal_populacao_cidada(dados_f)
    elif aba_publica == "🏥 Situação das UBS":
        render_situacao_ubs_populacao(dados_f)
    elif aba_publica == "⏱️ Previsão de atendimento":
        render_previsao_atendimento_populacao(dados_f)
    elif aba_publica == "ℹ️ Sobre o SMD":
        render_sobre_populacao()
    else:
        render_portal_populacao_cidada(dados_f)

    st.divider()
    st.caption(
        "APS-Twin SMD | Perfil público cidadão | As previsões são orientativas e não substituem atendimento profissional, avisos oficiais da UBS ou serviços de urgência."
    )
    st.stop()

# =========================================================
# PAINEL EXECUTIVO
# =========================================================

if st.session_state["aba_ativa"] == "📊 Painel Executivo":
    st.markdown('<div class="section-title">📊 Painel Executivo da APS</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns([1, 1, 1])
    g1.plotly_chart(fig_gauge(indice_geral, "Índice Geral do SMD", "Score consolidado"), use_container_width=True)
    g2.plotly_chart(fig_gauge(sistematicidade, "Sistematicidade", "Interdependência dos indicadores"), use_container_width=True)
    g3.plotly_chart(fig_gauge(100 * adequados / max(1, (adequados + atencao + criticos)), "Registros adequados", "Percentual de adequação"), use_container_width=True)

    col_a, col_b = st.columns([1, 1])

    status_df = dados_f["status"].value_counts().reset_index()
    status_df.columns = ["status", "quantidade"]

    fig_status = px.pie(
        status_df,
        names="status",
        values="quantidade",
        title="Distribuição dos status",
        color="status",
        color_discrete_map=STATUS_COLORS
    )
    fig_status.update_traces(hole=0.58, textinfo="percent+label", pull=[0.04] * len(status_df))
    fig_status = plotly_layout(fig_status, height=450)
    col_a.plotly_chart(fig_status, use_container_width=True)

    dim_df = (
        dados_f
        .groupby("dimensao")["score"]
        .mean()
        .reset_index()
        .sort_values("score")
    )

    fig_dim = px.bar(
        dim_df,
        x="score",
        y="dimensao",
        orientation="h",
        title="Score médio por dimensão",
        color="score",
        color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
        text="score"
    )
    fig_dim.update_traces(marker_line_width=0, opacity=0.94, texttemplate="%{text:.1f}", textposition="outside")
    fig_dim.update_layout(coloraxis_showscale=False)
    fig_dim = plotly_layout(fig_dim, height=470, showlegend=False)
    col_b.plotly_chart(fig_dim, use_container_width=True)

    st.subheader("Resumo executivo por dimensão")
    resumo_dim = (
        dados_f
        .groupby("dimensao")
        .agg(
            score_medio=("score", "mean"),
            valor_medio=("valor", "mean"),
            registros=("score", "count"),
            indicadores=("indicador", "nunique")
        )
        .reset_index()
        .sort_values("score_medio")
    )
    st.dataframe(resumo_dim, use_container_width=True)



# =========================================================
# QUALIDADE DOS DADOS
# =========================================================

if st.session_state["aba_ativa"] == "🧪 Qualidade dos Dados":
    if not usuario_pode_api():
        st.warning("Este módulo é restrito ao Administrador e ao perfil Aluno/Pesquisador.")
        st.stop()
    st.markdown('<div class="section-title">🧪 Qualidade dos Dados e Integração</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Esta aba avalia se os dados que alimentam o APS-Twin SMD possuem completude,
        consistência mínima, cobertura dos 18 indicadores e integração adequada para sustentar
        monitoramento, Machine Learning e Digital Twin.
        """
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Qualidade geral", f"{qualidade_dados['qualidade_geral']:.1f}")
    q2.metric("Registros tratados", qualidade_dados["n_registros"])
    q3.metric("Indicadores reconhecidos", f"{qualidade_dados['n_indicadores']}/18")
    q4.metric("Meses/períodos", qualidade_dados["n_meses"])

    col_q1, col_q2 = st.columns(2)
    col_q1.plotly_chart(grafico_qualidade_dados(qualidade_dados), use_container_width=True)
    col_q2.plotly_chart(grafico_maturidade_dt(maturidade_dt), use_container_width=True)

    st.subheader("Fontes de dados ativas")

    fontes_df = (
        dados_f
        .groupby("fonte")
        .agg(
            registros=("score", "count"),
            indicadores=("indicador", "nunique"),
            ubs=("ubs", "nunique"),
            score_medio=("score", "mean")
        )
        .reset_index()
        .sort_values("registros", ascending=False)
    )

    fig_fontes = px.bar(
        fontes_df,
        x="fonte",
        y="registros",
        color="fonte",
        text="registros",
        title="Registros por fonte de dados"
    )
    fig_fontes.update_traces(textposition="outside")
    fig_fontes = plotly_layout(fig_fontes, height=430)
    st.plotly_chart(fig_fontes, use_container_width=True)

    st.dataframe(fontes_df, use_container_width=True)

    st.subheader("Campos e indicadores não reconhecidos")

    nao_classificados = (
        dados_f[dados_f["dimensao"].eq("Sem dimensão")]
        .groupby("indicador_original")
        .size()
        .reset_index(name="ocorrencias")
        .sort_values("ocorrencias", ascending=False)
    )

    if nao_classificados.empty:
        st.success("Todos os indicadores tratados no recorte atual foram reconhecidos pelo dicionário.")
    else:
        st.warning("Existem campos não reconhecidos. Avalie se devem ser incluídos no dicionário de indicadores.")
        st.dataframe(nao_classificados, use_container_width=True)

    st.subheader("Diagnóstico automático")

    diagnosticos = []

    if qualidade_dados["cobertura_indicadores"] < 100:
        diagnosticos.append("Nem todos os 18 indicadores estão presentes no recorte atual.")
    if qualidade_dados["completude"] < 90:
        diagnosticos.append("Há perda de completude nos valores dos indicadores.")
    if qualidade_dados["consistencia_metas"] < 100:
        diagnosticos.append("Alguns indicadores não possuem meta associada no dicionário.")
    if not api_ativa:
        diagnosticos.append("A API Google Forms/Sheets não está ativa no recorte atual.")

    if diagnosticos:
        for d in diagnosticos:
            st.warning(d)
    else:
        st.success("A base apresenta boa condição operacional para alimentar o APS-Twin SMD.")


# =========================================================
# 18 INDICADORES
# =========================================================

if st.session_state["aba_ativa"] == "🎯 18 Indicadores":
    st.markdown('<div class="section-title">Painel dos 18 indicadores do SMD</div>', unsafe_allow_html=True)

    resumo_ind = (
        dados_f
        .groupby(["indicador", "dimensao", "nivel"])
        .agg(
            valor_medio=("valor", "mean"),
            meta=("meta", "mean"),
            score_medio=("score", "mean"),
            registros=("score", "count")
        )
        .reset_index()
    )

    resumo_ind["status"] = resumo_ind["score_medio"].apply(classificar_status)
    resumo_ind = resumo_ind.sort_values("score_medio")

    st.dataframe(resumo_ind, use_container_width=True)

    fig_ind = px.bar(
        resumo_ind,
        x="score_medio",
        y="indicador",
        color="status",
        color_discrete_map=STATUS_COLORS,
        orientation="h",
        title="Score médio dos indicadores",
        text="score_medio"
    )
    fig_ind.update_traces(marker_line_width=0, opacity=0.94, texttemplate="%{text:.1f}", textposition="outside")
    fig_ind = plotly_layout(fig_ind, height=760)
    st.plotly_chart(fig_ind, use_container_width=True)

    indicador_detalhe = st.selectbox(
        "Escolha um indicador para detalhar",
        options=sorted(dados_f["indicador"].unique())
    )

    dados_ind = dados_f[dados_f["indicador"] == indicador_detalhe].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor médio", f"{dados_ind['valor'].mean():.3f}")
    c2.metric("Meta", f"{dados_ind['meta'].mean():.3f}")
    c3.metric("Score médio", f"{dados_ind['score'].mean():.1f}")
    c4.metric("Registros", len(dados_ind))

    fig_box = px.box(
        dados_ind,
        x="ubs",
        y="score",
        color="status",
        color_discrete_map=STATUS_COLORS,
        title=f"Distribuição do score por UBS: {indicador_detalhe}"
    )
    fig_box = plotly_layout(fig_box, height=540)
    st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# UBS E TERRITÓRIO
# =========================================================

if st.session_state["aba_ativa"] == "🏥 UBS e Território":
    st.markdown('<div class="section-title">Análise por UBS e território</div>', unsafe_allow_html=True)

    if st.session_state["ubs_selecionada_card"] != "Todas":
        st.subheader(f"🏥 Painel individual da UBS: {st.session_state['ubs_selecionada_card']}")

        dados_ubs_atual = dados_f.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score médio da UBS", f"{dados_ubs_atual['score'].mean():.1f}")
        c2.metric("Indicadores avaliados", dados_ubs_atual["indicador"].nunique())
        c3.metric("Registros críticos", (dados_ubs_atual["status"] == "Crítico").sum())
        c4.metric("Dimensões avaliadas", dados_ubs_atual["dimensao"].nunique())

        resumo_ubs_ind = (
            dados_ubs_atual
            .groupby(["indicador", "dimensao"])
            .agg(
                score_medio=("score", "mean"),
                valor_medio=("valor", "mean"),
                meta=("meta", "mean"),
                registros=("score", "count")
            )
            .reset_index()
            .sort_values("score_medio")
        )

        resumo_ubs_ind["status"] = resumo_ubs_ind["score_medio"].apply(classificar_status)

        fig_ubs_ind = px.bar(
            resumo_ubs_ind,
            x="score_medio",
            y="indicador",
            color="status",
            color_discrete_map=STATUS_COLORS,
            orientation="h",
            title="Desempenho dos indicadores na UBS selecionada",
            text="score_medio"
        )
        fig_ubs_ind.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_ubs_ind = plotly_layout(fig_ubs_ind, height=660)
        st.plotly_chart(fig_ubs_ind, use_container_width=True)

        st.dataframe(resumo_ubs_ind, use_container_width=True)

    ranking = (
        dados_f
        .groupby("ubs")["score"]
        .mean()
        .reset_index()
        .sort_values("score")
    )

    fig_rank = px.bar(
        ranking,
        x="score",
        y="ubs",
        orientation="h",
        title="Ranking médio por UBS/nível territorial",
        color="score",
        color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
        text="score"
    )
    fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_rank.update_layout(coloraxis_showscale=False)
    fig_rank = plotly_layout(fig_rank, height=520, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    heatmap = dados_f.pivot_table(
        index="ubs",
        columns="indicador",
        values="score",
        aggfunc="mean"
    )

    fig_heat = px.imshow(
        heatmap,
        aspect="auto",
        title="Mapa de calor: UBS × Indicadores",
        color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
        zmin=0,
        zmax=100
    )
    fig_heat = plotly_layout(fig_heat, height=660, showlegend=False)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Tabela UBS × dimensão")
    tabela_ubs_dim = (
        dados_f
        .groupby(["ubs", "dimensao"])["score"]
        .mean()
        .reset_index()
        .sort_values(["ubs", "score"])
    )
    st.dataframe(tabela_ubs_dim, use_container_width=True)


# =========================================================
# SÉRIES TEMPORAIS
# =========================================================

if st.session_state["aba_ativa"] == "📈 Séries Temporais":
    st.markdown('<div class="section-title">Séries temporais dos indicadores</div>', unsafe_allow_html=True)

    indicador_ts = st.selectbox(
        "Indicador da série temporal",
        options=sorted(dados_f["indicador"].unique()),
        key="ts_ind"
    )

    dados_ts = dados_f[dados_f["indicador"] == indicador_ts].copy()

    serie = (
        dados_ts
        .groupby(["mes", "ubs"])["score"]
        .mean()
        .reset_index()
    )

    serie = ordenar_dataframe_por_mes(serie, "mes")
    ordem_meses_serie = ordenar_meses_lista(serie["mes"].astype(str).unique())

    fig_ts = px.line(
        serie,
        x="mes",
        y="score",
        color="ubs",
        markers=True,
        category_orders={"mes": ordem_meses_serie},
        title=f"Evolução temporal do score: {indicador_ts}"
    )
    fig_ts.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_ts.add_hrect(y0=85, y1=100, fillcolor="green", opacity=0.08, line_width=0)
    fig_ts.add_hrect(y0=60, y1=85, fillcolor="orange", opacity=0.08, line_width=0)
    fig_ts.add_hrect(y0=0, y1=60, fillcolor="red", opacity=0.06, line_width=0)
    fig_ts = plotly_layout(fig_ts, height=600)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.dataframe(serie, use_container_width=True)


# =========================================================
# MACHINE LEARNING
# =========================================================

if st.session_state["aba_ativa"] == "🤖 Machine Learning":
    st.markdown('<div class="section-title">Modelagem preditiva com Machine Learning</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Esta aba compara quatro modelos preditivos aplicados ao comportamento temporal dos indicadores do SMD.
        O objetivo é apoiar a interpretação gerencial, antecipar tendências e alimentar a camada de Digital Twin.
        """
    )

    st.info(
        "As previsões são exploratórias e funcionam como apoio à decisão. "
        "A qualidade dos resultados depende da quantidade de períodos, da estabilidade dos registros e da consistência dos dados."
    )

    col_ml1, col_ml2 = st.columns(2)

    indicador_ml = col_ml1.selectbox(
        "Indicador para predição",
        options=sorted(dados_f["indicador"].unique()),
        key="ml_ind"
    )

    ubs_ml = col_ml2.selectbox(
        "UBS/nível para predição",
        options=sorted(dados_f["ubs"].unique()),
        key="ml_ubs"
    )

    dados_ml = dados_f[
        (dados_f["indicador"] == indicador_ml) &
        (dados_f["ubs"] == ubs_ml)
    ].copy()

    resultado_ml, erro_ml = treinar_modelos_ml(dados_ml)

    if erro_ml:
        st.warning(erro_ml)
        st.markdown(
            """
            **Sugestão metodológica:** para melhorar a predição, amplie a série histórica do indicador,
            mantenha periodicidade regular de atualização e reduza lacunas de preenchimento.
            """
        )
    else:
        resultados = preparar_resultados_ml_para_graficos(resultado_ml["resultados"])
        serie_ml = resultado_ml["serie"]

        melhor = resultados.sort_values("MAE").iloc[0]

        st.subheader("Resumo do melhor modelo")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Melhor modelo", melhor["modelo"])
        c2.metric("MAE", f"{melhor['MAE']:.2f}")
        c3.metric("RMSE", f"{melhor['RMSE']:.2f}")
        c4.metric("R²", f"{melhor['R2']:.2f}" if not pd.isna(melhor["R2"]) else "N/A")
        c5.metric("Próximo score", f"{melhor['previsao_proximo_periodo']:.1f}")

        st.success(
            f"O modelo com melhor desempenho para {indicador_ml} em {ubs_ml} foi "
            f"**{melhor['modelo']}**, com MAE de {melhor['MAE']:.2f}. "
            f"A previsão exploratória para o próximo período é de "
            f"**{melhor['previsao_proximo_periodo']:.1f} pontos**."
        )

        st.subheader("Comparação técnica dos modelos")
        tabela_modelos = resultados[
            [
                "modelo",
                "R2",
                "RMSE",
                "MAE",
                "previsao_proximo_periodo",
                "score_modelo",
                "qualidade"
            ]
        ].copy()

        tabela_modelos = tabela_modelos.rename(columns={
            "modelo": "Modelo",
            "R2": "R²",
            "RMSE": "RMSE",
            "MAE": "MAE",
            "previsao_proximo_periodo": "Previsão próximo período",
            "score_modelo": "Score técnico do modelo",
            "qualidade": "Classificação"
        })

        st.dataframe(tabela_modelos, use_container_width=True, hide_index=True)

        col_g1, col_g2 = st.columns(2)

        fig_mae = px.bar(
            resultados.sort_values("MAE", ascending=True),
            x="MAE",
            y="modelo",
            color="qualidade",
            orientation="h",
            title="Ranking dos modelos por erro absoluto médio",
            text="MAE",
            color_discrete_sequence=[COLORS["green"], COLORS["blue"], COLORS["amber"], COLORS["red"]]
        )
        fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0, opacity=0.94)
        fig_mae = plotly_layout(fig_mae, height=450)
        col_g1.plotly_chart(fig_mae, use_container_width=True)

        fig_prev = px.bar(
            resultados.sort_values("previsao_proximo_periodo", ascending=False),
            x="modelo",
            y="previsao_proximo_periodo",
            color="modelo",
            title="Previsão do score para o próximo período",
            text="previsao_proximo_periodo",
            color_discrete_map=MODEL_COLORS
        )
        fig_prev.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0, opacity=0.94)
        fig_prev = plotly_layout(fig_prev, height=450)
        col_g2.plotly_chart(fig_prev, use_container_width=True)

        st.subheader("Série temporal observada e previsão exploratória")
        serie_plot = serie_ml[["periodo_num", "score"]].copy()
        serie_plot["tipo"] = "Observado"

        proximo_periodo = serie_plot["periodo_num"].max() + 1
        previsao_plot = pd.DataFrame({
            "periodo_num": [proximo_periodo],
            "score": [melhor["previsao_proximo_periodo"]],
            "tipo": ["Previsão"]
        })

        serie_completa = pd.concat([serie_plot, previsao_plot], ignore_index=True)

        fig_tend = px.line(
            serie_completa,
            x="periodo_num",
            y="score",
            color="tipo",
            markers=True,
            title=f"Tendência observada e previsão: {indicador_ml} | {ubs_ml}",
            color_discrete_map={"Observado": COLORS["blue"], "Previsão": COLORS["red"]}
        )
        fig_tend.update_traces(line=dict(width=4), marker=dict(size=9))
        fig_tend.add_hrect(y0=85, y1=100, fillcolor="green", opacity=0.08, line_width=0)
        fig_tend.add_hrect(y0=60, y1=85, fillcolor="orange", opacity=0.08, line_width=0)
        fig_tend.add_hrect(y0=0, y1=60, fillcolor="red", opacity=0.06, line_width=0)
        fig_tend.add_hline(y=85, line_dash="dash", line_color=COLORS["green"], annotation_text="Adequado")
        fig_tend.add_hline(y=60, line_dash="dash", line_color=COLORS["amber"], annotation_text="Atenção")
        fig_tend = plotly_layout(fig_tend, height=560)
        st.plotly_chart(fig_tend, use_container_width=True)

        st.subheader("Aderência preditiva do melhor modelo")
        previsoes_dict = resultado_ml["previsoes"]

        if melhor["modelo"] in previsoes_dict:
            df_previsto = previsoes_dict[melhor["modelo"]].copy()
            df_previsto["observacao"] = range(1, len(df_previsto) + 1)

            df_longo = df_previsto.melt(
                id_vars="observacao",
                value_vars=["observado", "previsto"],
                var_name="tipo",
                value_name="score"
            )

            fig_obs_prev = px.line(
                df_longo,
                x="observacao",
                y="score",
                color="tipo",
                markers=True,
                title=f"Observado × previsto — {melhor['modelo']}",
                color_discrete_map={"observado": COLORS["blue"], "previsto": COLORS["red"]}
            )
            fig_obs_prev.update_traces(line=dict(width=4), marker=dict(size=9))
            fig_obs_prev = plotly_layout(fig_obs_prev, height=520)
            st.plotly_chart(fig_obs_prev, use_container_width=True)

        st.subheader("Radar comparativo dos modelos")
        radar = resultados.copy()
        radar["R2_norm"] = radar["R2"].fillna(0).clip(lower=0) * 100

        max_rmse = radar["RMSE"].max()
        max_mae = radar["MAE"].max()

        radar["RMSE_norm"] = 100 * (1 - radar["RMSE"] / max_rmse) if max_rmse > 0 else 100
        radar["MAE_norm"] = 100 * (1 - radar["MAE"] / max_mae) if max_mae > 0 else 100
        radar["Previsao_norm"] = radar["previsao_proximo_periodo"]

        radar_long = radar.melt(
            id_vars="modelo",
            value_vars=["R2_norm", "RMSE_norm", "MAE_norm", "Previsao_norm"],
            var_name="métrica",
            value_name="valor"
        )

        radar_long["métrica"] = radar_long["métrica"].replace({
            "R2_norm": "R²",
            "RMSE_norm": "RMSE ajustado",
            "MAE_norm": "MAE ajustado",
            "Previsao_norm": "Previsão"
        })

        fig_radar = px.line_polar(
            radar_long,
            r="valor",
            theta="métrica",
            color="modelo",
            line_close=True,
            title="Radar comparativo dos modelos",
            color_discrete_map=MODEL_COLORS
        )
        fig_radar.update_traces(fill="toself")
        fig_radar.update_layout(
            height=570,
            paper_bgcolor="white",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title_font=dict(size=18, color=COLORS["navy"])
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Interpretação gerencial automática")

        if melhor["previsao_proximo_periodo"] >= 85:
            mensagem_status = "A previsão indica manutenção ou alcance de uma condição adequada."
        elif melhor["previsao_proximo_periodo"] >= 60:
            mensagem_status = "A previsão indica condição intermediária, exigindo acompanhamento gerencial."
        else:
            mensagem_status = "A previsão indica risco crítico e necessidade de priorização da unidade ou indicador."

        st.markdown(
            f"""
            **Síntese:** O indicador **{indicador_ml}**, na unidade/nível **{ubs_ml}**, apresentou como melhor modelo
            o **{melhor['modelo']}**. O erro absoluto médio foi de **{melhor['MAE']:.2f}**, enquanto o RMSE foi de
            **{melhor['RMSE']:.2f}**. A previsão exploratória para o próximo período é de
            **{melhor['previsao_proximo_periodo']:.1f} pontos**.

            **Leitura gerencial:** {mensagem_status}

            **Uso no Digital Twin:** essa previsão pode alimentar simulações de cenário, alertas e priorização de ações
            dentro do SMD da Atenção Primária à Saúde.
            """
        )


# =========================================================
# DIGITAL TWIN CIBERFÍSICO
# =========================================================

if st.session_state["aba_ativa"] == "🧠 Digital Twin Ciberfísico":
    st.markdown('<div class="section-title">Digital Twin ciberfísico da APS</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Esta aba transforma o SMD em uma representação ciberfísica da APS: a unidade real gera dados,
        os dados alimentam o gêmeo digital, os modelos analíticos simulam cenários e a gestão recebe
        apoio para decidir e agir.
        """
    )

    st.plotly_chart(criar_fluxo_ciberfisico(), use_container_width=True)

    dt1, dt2, dt3 = st.columns(3)
    dt1.plotly_chart(fig_gauge(indice_geral, "Estado atual", "Desempenho espelhado"), use_container_width=True)
    dt2.plotly_chart(fig_gauge(sistematicidade, "Sistema integrado", "Sistematicidade"), use_container_width=True)
    dt3.plotly_chart(fig_gauge(100 * adequados / max(1, (adequados + atencao + criticos)), "Estabilidade operacional", "% adequado"), use_container_width=True)

    st.subheader("Painel de simulação ciberfísica")

    col_dt1, col_dt2, col_dt3 = st.columns(3)

    indicador_dt = col_dt1.selectbox(
        "Indicador para intervenção",
        options=sorted(dados_f["indicador"].unique()),
        key="dt_ind"
    )

    ubs_dt = col_dt2.selectbox(
        "UBS/nível de intervenção",
        options=["Todas"] + sorted(dados_f["ubs"].unique()),
        key="dt_ubs"
    )

    tipo_intervencao = col_dt3.selectbox(
        "Tipo de intervenção",
        options=[
            "Melhoria operacional",
            "Aumento de capacidade",
            "Ação corretiva",
            "Integração tecnológica",
            "Treinamento e aprendizagem"
        ],
        key="dt_tipo"
    )

    melhoria = st.slider(
        "Intensidade da intervenção simulada (%)",
        min_value=0,
        max_value=100,
        value=15
    )

    latencia = st.slider(
        "Latência do sistema ciberfísico — tempo de resposta dos dados (%)",
        min_value=0,
        max_value=100,
        value=20,
        help="Quanto maior a latência, menor o efeito imediato da intervenção no gêmeo digital."
    )

    confianca_sensor = st.slider(
        "Confiabilidade da coleta/API/sensores (%)",
        min_value=0,
        max_value=100,
        value=85,
        help="Representa a qualidade da entrada de dados no sistema ciberfísico."
    )

    dados_dt = dados_f.copy()
    indice_antes = dados_dt["score"].mean()

    mascara = dados_dt["indicador"] == indicador_dt
    if ubs_dt != "Todas":
        mascara = mascara & (dados_dt["ubs"] == ubs_dt)

    fator_latencia = 1 - (latencia / 100)
    fator_confianca = confianca_sensor / 100
    efeito_real = (melhoria / 100) * fator_latencia * fator_confianca

    dados_dt.loc[mascara, "score"] = dados_dt.loc[mascara, "score"] * (1 + efeito_real)
    dados_dt["score"] = dados_dt["score"].clip(0, 100)

    indice_depois = dados_dt["score"].mean()
    ganho = indice_depois - indice_antes

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Índice antes", f"{indice_antes:.1f}")
    c2.metric("Índice simulado", f"{indice_depois:.1f}")
    c3.metric("Ganho projetado", f"{ganho:.2f}")
    c4.metric("Efeito real aplicado", f"{efeito_real*100:.1f}%")

    sim_df = pd.DataFrame({
        "cenário": ["Antes", "Depois"],
        "Índice Geral do SMD": [indice_antes, indice_depois]
    })

    fig_sim = px.bar(
        sim_df,
        x="cenário",
        y="Índice Geral do SMD",
        color="cenário",
        text="Índice Geral do SMD",
        title=f"Simulação ciberfísica: {tipo_intervencao} em {indicador_dt}",
        color_discrete_map={"Antes": COLORS["gray"], "Depois": COLORS["teal"]}
    )
    fig_sim.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_sim = plotly_layout(fig_sim, height=500)
    st.plotly_chart(fig_sim, use_container_width=True)

    st.subheader("Maturidade ciberfísica do Digital Twin")

    camada_dt = calcular_camada_ciberfisica(dados_f, dados_dt)

    fig_camadas = px.bar(
        camada_dt.sort_values("valor"),
        x="valor",
        y="camada",
        orientation="h",
        text="valor",
        color="valor",
        color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
        title="Camadas de maturidade ciberfísica"
    )
    fig_camadas.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_camadas.update_layout(coloraxis_showscale=False)
    fig_camadas = plotly_layout(fig_camadas, height=520, showlegend=False)
    st.plotly_chart(fig_camadas, use_container_width=True)

    st.subheader("Espelhamento real × simulado por dimensão")

    antes_dim = (
        dados_f
        .groupby("dimensao")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "Antes"})
    )

    depois_dim = (
        dados_dt
        .groupby("dimensao")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "Depois"})
    )

    comparativo_dim = antes_dim.merge(depois_dim, on="dimensao", how="outer")
    comparativo_long = comparativo_dim.melt(
        id_vars="dimensao",
        value_vars=["Antes", "Depois"],
        var_name="cenário",
        value_name="score"
    )

    fig_dim_dt = px.bar(
        comparativo_long,
        x="score",
        y="dimensao",
        color="cenário",
        orientation="h",
        barmode="group",
        title="Comparação do espelho digital antes/depois da intervenção",
        color_discrete_map={"Antes": COLORS["gray"], "Depois": COLORS["teal"]}
    )
    fig_dim_dt = plotly_layout(fig_dim_dt, height=620)
    st.plotly_chart(fig_dim_dt, use_container_width=True)

    st.subheader("Interpretação do Digital Twin")

    if ganho > 5:
        leitura_dt = "A intervenção simulada apresenta impacto expressivo no desempenho geral do SMD."
    elif ganho > 1:
        leitura_dt = "A intervenção simulada apresenta impacto moderado e pode ser combinada com outras ações."
    elif ganho > 0:
        leitura_dt = "A intervenção simulada apresenta impacto incremental, sugerindo efeito localizado."
    else:
        leitura_dt = "A intervenção simulada não altera substancialmente o desempenho geral, indicando necessidade de revisar a estratégia."

    st.markdown(
        f"""
        **Leitura ciberfísica:** O sistema real da APS é representado pelo conjunto de UBS, processos e indicadores.
        A camada digital espelha esse funcionamento por meio dos dados do SMD. Nesta simulação, uma intervenção do tipo
        **{tipo_intervencao}** foi aplicada sobre o indicador **{indicador_dt}**, no nível **{ubs_dt}**.

        **Resultado:** o índice geral saiu de **{indice_antes:.1f}** para **{indice_depois:.1f}**, com ganho projetado de
        **{ganho:.2f} pontos**. Considerando latência de **{latencia}%** e confiabilidade de dados de
        **{confianca_sensor}%**, o efeito real aplicado foi de **{efeito_real*100:.1f}%**.

        **Interpretação gerencial:** {leitura_dt}
        """
    )


# =========================================================
# SISTEMATICIDADE
# =========================================================

if st.session_state["aba_ativa"] == "🕸️ Sistematicidade":
    st.markdown('<div class="section-title">Índice de Sistematicidade do SMD</div>', unsafe_allow_html=True)

    st.markdown(
        """
        O Índice de Sistematicidade estima o grau de interdependência entre os indicadores do SMD.
        Quanto maior o valor, maior a integração estatística entre os indicadores monitorados.
        """
    )

    st.metric(
        "Índice de Sistematicidade",
        f"{sistematicidade:.1f}" if not pd.isna(sistematicidade) else "N/A"
    )

    if matriz_corr is not None:
        fig_corr = px.imshow(
            matriz_corr,
            aspect="auto",
            title="Matriz de correlação absoluta entre indicadores",
            color_continuous_scale=["#F7FAFE", "#42A5F5", "#0B2545"],
            zmin=0,
            zmax=1
        )
        fig_corr = plotly_layout(fig_corr, height=740, showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)

    if conectividade is not None:
        st.subheader("Indicadores mais conectados ao sistema")

        conectividade["conectividade_media"] = conectividade["conectividade_media"] * 100

        fig_con = px.bar(
            conectividade.sort_values("conectividade_media"),
            x="conectividade_media",
            y="indicador",
            orientation="h",
            title="Conectividade média dos indicadores",
            color="conectividade_media",
            color_continuous_scale=["#F9A825", "#0097A7", "#0B2545"],
            text="conectividade_media"
        )
        fig_con.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_con.update_layout(coloraxis_showscale=False)
        fig_con = plotly_layout(fig_con, height=700, showlegend=False)
        st.plotly_chart(fig_con, use_container_width=True)

        st.dataframe(conectividade, use_container_width=True)



# =========================================================
# RELATÓRIOS
# =========================================================

if st.session_state["aba_ativa"] == "📄 Relatórios":
    st.markdown('<div class="section-title">Relatórios executivos automáticos</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Gere um relatório textual do recorte atualmente selecionado no dashboard.
        O relatório considera os filtros ativos, a UBS selecionada, os indicadores, os dados da API e as bases locais.
        """
    )

    ubs_relatorio = st.session_state["ubs_selecionada_card"]
    if ubs_relatorio == "Todas":
        ubs_relatorio = "Visão municipal / recorte filtrado"

    relatorio_md = gerar_relatorio_textual(
        dados_base=dados_f,
        ubs_ref=ubs_relatorio,
        indice_geral_ref=indice_geral,
        sistematicidade_ref=sistematicidade,
        qualidade_ref=qualidade_dados["qualidade_geral"],
        maturidade_ref=maturidade_dt["maturidade_dt"],
        risco_ref=risco_gerencial,
        classe_risco_ref=classe_risco
    )

    st.download_button(
        label="Baixar relatório em Markdown",
        data=relatorio_md.encode("utf-8"),
        file_name="relatorio_aps_twin_smd.md",
        mime="text/markdown",
        use_container_width=True
    )

    csv_recorte = dados_f.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Baixar dados tratados do recorte em CSV",
        data=csv_recorte,
        file_name="dados_tratados_aps_twin_smd.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("Prévia do relatório")
    st.markdown(relatorio_md)



# =========================================================
# SOBRE O SMD
# =========================================================

if st.session_state["aba_ativa"] == "ℹ️ Sobre o SMD":
    st.markdown('<div class="section-title">Sobre o APS-Twin SMD</div>', unsafe_allow_html=True)

    st.markdown(
        """
        O APS-Twin SMD é um protótipo tecnológico voltado à operacionalização de um
        Sistema de Medição de Desempenho aplicado à Atenção Primária à Saúde.

        A aplicação organiza os 18 indicadores do SMD, permite análise por UBS, mês,
        dimensão e indicador, calcula um índice geral de desempenho, estima a
        sistematicidade entre indicadores, executa modelos preditivos exploratórios e
        oferece uma camada de Digital Twin ciberfísico para simulação gerencial.

        A proposta não substitui a decisão humana. O sistema funciona como infraestrutura
        sociotécnica de apoio à interpretação, aprendizagem, monitoramento e tomada de
        decisão na gestão da APS.
        """
    )

    st.write("### Dimensões cadastradas")
    st.dataframe(
        dic[["nome_curto", "dimensao", "nivel", "polaridade", "meta"]],
        use_container_width=True
    )


# =========================================================
# RODAPÉ
# =========================================================

st.divider()
st.caption(
    "APS-Twin SMD v1.9.0.7.6.5.4.3.2 | Protótipo web em Python | API em tempo real | Qualidade dos Dados | Relatórios Executivos | Digital Twin Ciberfísico-Gerencial da APS | "
    "Tese de Carlos Jefferson de Melo Santos."
)
