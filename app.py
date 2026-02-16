"""
AI-Powered Receipt Analyzer with LLM Insights
Main Streamlit Application - Hackathon Project
"""

import os
import io

os.environ["PYTHONIOENCODING"] = "utf-8"

# Load from .env locally; on Streamlit Cloud use st.secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required on cloud

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
import time
from PIL import Image

# Bridge st.secrets → os.environ so all modules can use os.getenv()
def _load_secrets():
    """Load Streamlit Cloud secrets into environment variables."""
    try:
        for key in st.secrets:
            if isinstance(st.secrets[key], str):
                os.environ.setdefault(key, st.secrets[key])
    except Exception:
        pass  # No secrets configured

_load_secrets()

# Import our modules
from image_processor import ImageProcessor
from ocr_engine import OCREngine
from data_parser import DataParser
from categorizer import ExpenseCategorizer
from analyzer import SpendingAnalyzer
from llm_advisor import LLMAdvisor
from email_alerter import EmailAlerter


@st.cache_resource(show_spinner="Loading OCR engine (first time may take a minute)...")
def get_ocr_engine():
    """Cache the OCR engine so heavy models load only once."""
    return OCREngine()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ReceiptIQ — AI Receipt Analyzer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — STUNNING UI
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ─── Import Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── Root Variables ─── */
    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --primary-dark: #4F46E5;
        --accent: #06B6D4;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-dark: #0F172A;
        --bg-card: #1E293B;
        --bg-card-hover: #334155;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --border: #334155;
        --gradient-1: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
        --gradient-2: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
        --gradient-3: linear-gradient(135deg, #22C55E 0%, #06B6D4 100%);
    }

    /* ─── Global Styles ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Hero Banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1, #06B6D4);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }

    /* ─── Metric Cards ─── */
    .metric-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }

    .metric-card .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.3rem;
        font-weight: 600;
    }

    /* ─── Section Headers ─── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #334155;
    }

    .section-header .section-icon {
        font-size: 1.5rem;
    }

    .section-header .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0;
    }

    .section-header .section-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ─── Pipeline Step Cards ─── */
    .pipeline-step {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s ease;
    }

    .pipeline-step.active {
        border-color: #6366F1;
        background: linear-gradient(145deg, #1E293B, #1a1a3e);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.1);
    }

    .pipeline-step.completed {
        border-color: #22C55E;
    }

    .step-number {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }

    .step-number.pending {
        background: #334155;
        color: #94A3B8;
    }

    .step-number.active {
        background: linear-gradient(135deg, #6366F1, #06B6D4);
        color: white;
        animation: pulse 2s infinite;
    }

    .step-number.completed {
        background: #22C55E;
        color: white;
    }

    .step-content .step-title {
        font-weight: 600;
        color: #F1F5F9;
        font-size: 0.95rem;
    }

    .step-content .step-desc {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
    }

    /* ─── Insight Cards ─── */
    .insight-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    .insight-card.warning {
        border-left-color: #F59E0B;
    }

    .insight-card.positive {
        border-left-color: #22C55E;
    }

    .insight-card.danger {
        border-left-color: #EF4444;
    }

    .insight-text {
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ─── Score Circle ─── */
    .score-container {
        text-align: center;
        padding: 2rem;
    }

    .score-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        border: 6px solid;
        margin-bottom: 1rem;
    }

    .score-value {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .score-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ─── Upload Area ─── */
    .upload-area {
        border: 2px dashed #4F46E5;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(99, 102, 241, 0.05);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .upload-area:hover {
        border-color: #818CF8;
        background: rgba(99, 102, 241, 0.1);
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .upload-text {
        color: #94A3B8;
        font-size: 1rem;
    }

    .upload-formats {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    /* ─── Processing Animation ─── */
    .processing-container {
        text-align: center;
        padding: 2rem;
    }

    .processing-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 4px solid #334155;
        border-top-color: #6366F1;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* ─── Items Table ─── */
    .items-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
    }

    .items-table th {
        background: #1E293B;
        color: #94A3B8;
        padding: 0.8rem 1rem;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .items-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #1E293B;
        color: #E2E8F0;
        font-size: 0.9rem;
    }

    .items-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }

    /* ─── AI Advice Box ─── */
    .ai-advice-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }

    .ai-advice-box::before {
        content: '✨';
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        font-size: 2rem;
        opacity: 0.3;
    }

    .ai-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1, #06B6D4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    /* ─── Sidebar Styles ─── */
    .sidebar-section {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748B;
        font-size: 0.8rem;
        border-top: 1px solid #1E293B;
        margin-top: 3rem;
    }

    /* ─── Streamlit Overrides ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #1E293B;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        border: 1px solid #334155;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important;
        border-color: #6366F1 !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 600;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def render_hero():
    """Render the hero banner."""
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧾 ReceiptIQ</div>
        <div class="hero-subtitle">
            AI-Powered Receipt Analyzer — Transform receipts into financial intelligence
        </div>
        <div class="hero-badge">✨ POWERED BY AI + COMPUTER VISION</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(icon, value, label, color="#6366F1"):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_section_header(icon, title, badge=""):
    """Render a styled section header."""
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h2 class="section-title">{title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_steps(current_step=0):
    """Render the processing pipeline visualization."""
    steps = [
        ("Image Processing", "Preprocessing & enhancement"),
        ("OCR Extraction", "Text recognition from receipt"),
        ("Data Parsing", "Structuring items & prices"),
        ("Categorization", "Classifying expenses"),
        ("Analysis", "Spending patterns & anomalies"),
        ("AI Insights", "Personalized financial advice"),
        ("Email Alerts", "Overspending notifications"),
    ]

    for i, (title, desc) in enumerate(steps):
        if i < current_step:
            status = "completed"
            number_display = "✓"
        elif i == current_step:
            status = "active"
            number_display = str(i + 1)
        else:
            status = "pending"
            number_display = str(i + 1)

        st.markdown(f"""
        <div class="pipeline-step {status}">
            <div class="step-number {status}">{number_display}</div>
            <div class="step-content">
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def create_donut_chart(category_percentages, currency="$"):
    """Create a beautiful donut chart for spending breakdown."""
    categories = list(category_percentages.keys())
    values = [data["amount"] for data in category_percentages.values()]
    colors = [data["color"] for data in category_percentages.values()]

    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0F172A', width=3)),
        textinfo='label+percent',
        textfont=dict(size=12, color='white', family='Inter'),
        hovertemplate='<b>%{label}</b><br>Amount: ' + currency + '%{value:,.2f}<br>Share: %{percent}<extra></extra>',
        pull=[0.05 if i == 0 else 0 for i in range(len(categories))],
    )])

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=11, color='#94A3B8', family='Inter'),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            orientation='v',
            yanchor='middle',
            y=0.5,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        annotations=[dict(
            text=f'<b>{currency}{sum(values):,.0f}</b><br><span style="font-size:12px;color:#94A3B8">Total</span>',
            x=0.5, y=0.5, font_size=22, showarrow=False,
            font=dict(color='white', family='JetBrains Mono')
        )]
    )

    return fig


def create_bar_chart(category_percentages, currency="$"):
    """Create a horizontal bar chart for category comparison."""
    categories = list(category_percentages.keys())
    amounts = [data["amount"] for data in category_percentages.values()]
    colors = [data["color"] for data in category_percentages.values()]
    percentages = [data["percentage"] for data in category_percentages.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=amounts,
        y=categories,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            cornerradius=6,
        ),
        text=[f'{currency}{a:,.2f} ({p}%)' for a, p in zip(amounts, percentages)],
        textposition='auto',
        textfont=dict(color='white', size=12, family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>Amount: ' + currency + '%{x:,.2f}<extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True, gridcolor='rgba(51,65,85,0.5)',
            title=dict(text=f'Amount ({currency})', font=dict(color='#94A3B8')),
            tickfont=dict(color='#94A3B8')
        ),
        yaxis=dict(
            tickfont=dict(color='#E2E8F0', size=12),
            autorange='reversed'
        ),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(250, len(categories) * 55),
    )

    return fig


def create_treemap(categorized_items, currency="$"):
    """Create a treemap visualization of items."""
    if not categorized_items:
        return None

    names = [item["name"][:20] for item in categorized_items]
    parents = [item["category"] for item in categorized_items]
    values = [item["total"] for item in categorized_items]

    # Build treemap data
    all_labels = list(set(parents)) + names
    all_parents = [""] * len(set(parents)) + parents
    all_values = [0] * len(set(parents)) + values

    fig = go.Figure(go.Treemap(
        labels=all_labels,
        parents=all_parents,
        values=all_values,
        textinfo='label+value',
        textfont=dict(size=12, family='Inter'),
        hovertemplate='<b>%{label}</b><br>' + currency + '%{value:,.2f}<extra></extra>',
        marker=dict(
            cornerradius=5,
            line=dict(color='#0F172A', width=2)
        ),
        pathbar=dict(textfont=dict(size=13, color='white')),
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
    )

    return fig


def render_score_gauge(score_data):
    """Render a spending health score gauge."""
    score = score_data["score"]
    color = score_data["color"]
    label = score_data["label"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Spending Health Score", 'font': {'size': 16, 'color': '#94A3B8'}},
        number={'font': {'size': 48, 'color': color, 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'color': '#64748B'}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#1E293B',
            'borderwidth': 2,
            'bordercolor': '#334155',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239,68,68,0.15)'},
                {'range': [40, 60], 'color': 'rgba(251,191,36,0.15)'},
                {'range': [60, 80], 'color': 'rgba(96,165,250,0.15)'},
                {'range': [80, 100], 'color': 'rgba(34,197,94,0.15)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# INITIALIZE SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

if "processed" not in st.session_state:
    st.session_state.processed = False
if "results" not in st.session_state:
    st.session_state.results = None
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = 0


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🧾</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #F1F5F9; margin-top: 0.5rem;">
            ReceiptIQ
        </div>
        <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.3rem;">
            AI Receipt Analyzer v1.0
        </div>
    </div>
    <hr style="border-color: #334155; margin: 1rem 0;">
    """, unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY", "")

    # Email Alert Settings
    st.markdown("##### 📧 Email Alerts")
    email_alerter = EmailAlerter()
    if email_alerter.is_configured():
        st.success("Email configured! Alerts will be sent automatically.")
    else:
        st.warning("Set SENDER_EMAIL & SENDER_APP_PASSWORD in .env for email alerts.")

    alert_email = st.text_input(
        "Recipient Email",
        placeholder="yourname@email.com",
        help="Enter the email address to receive overspending alerts"
    )
    alert_name = st.text_input(
        "Your Name",
        placeholder="John Doe",
        help="Name to personalize the alert email"
    )
    email_alerts_enabled = st.toggle("Enable Auto Email Alerts", value=True,
                                      help="Automatically send email when overspending is detected")

    st.markdown("<hr style='border-color: #334155; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Pipeline Status
    st.markdown("##### ⚙️ Processing Pipeline")
    render_pipeline_steps(st.session_state.pipeline_step if st.session_state.processed else 0)

    st.markdown("<hr style='border-color: #334155; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Info
    st.markdown("##### 📋 How It Works")
    st.markdown("""
    <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.8;">
        1️⃣ Upload a receipt image<br>
        2️⃣ AI processes & extracts text<br>
        3️⃣ Items are parsed & categorized<br>
        4️⃣ Spending patterns analyzed<br>
        5️⃣ Get personalized budget advice
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #334155; margin: 1rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.75rem; padding: 0.5rem;">
        Built with ❤️ using Python<br>
        OpenCV • EasyOCR • Gemini • Streamlit
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

render_hero()

# Upload Section
if not st.session_state.processed:
    render_section_header("📤", "Upload Receipt", "STEP 1")

    uploaded_file = st.file_uploader(
        "📸 Upload your receipt image (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is not None:
        # Store bytes in session so the file object is never re-read
        if "upload_bytes" not in st.session_state or st.session_state.get("upload_name") != uploaded_file.name:
            st.session_state.upload_bytes = uploaded_file.getvalue()
            st.session_state.upload_name = uploaded_file.name

        image_bytes = st.session_state.upload_bytes

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Uploaded Receipt")
            st.image(image_bytes)

        with col2:
            st.markdown("##### File Details")
            file_size = len(image_bytes) / 1024
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
            st.markdown(f"- **File Name:** {uploaded_file.name}")
            st.markdown(f"- **File Size:** {file_size:.1f} KB")
            st.markdown(f"- **Image Size:** {pil_img.size[0]} x {pil_img.size[1]} px")
            st.markdown(f"- **Format:** {pil_img.format or 'N/A'}")

        st.markdown("")

        if st.button("🚀 Analyze Receipt", type="primary", use_container_width=True):
            uploaded_file.seek(0)

            # ──────────────────────────────────
            # FULL PROCESSING PIPELINE
            # ──────────────────────────────────

            progress_bar = st.progress(0, text="Starting analysis pipeline...")

            results = {}

            # Step 1: Image Processing
            progress_bar.progress(10, text="Step 1/7 - Preprocessing image...")
            st.session_state.pipeline_step = 1
            try:
                processor = ImageProcessor()
                processed_images = processor.process_receipt(io.BytesIO(image_bytes))
                results["processed_images"] = processed_images
            except Exception as e:
                st.error(f"Image processing error: {e}")
                st.stop()

            # Step 2: OCR
            progress_bar.progress(25, text="Step 2/7 - Extracting text with OCR...")
            st.session_state.pipeline_step = 2
            try:
                ocr = get_ocr_engine()
                ocr_results = ocr.process_image(processed_images)
                results["ocr_results"] = ocr_results
            except Exception as e:
                st.error(f"OCR error: {e}")
                st.stop()

            # Step 3: Data Parsing
            progress_bar.progress(40, text="Step 3/7 - Parsing receipt data...")
            st.session_state.pipeline_step = 3
            try:
                parser = DataParser()
                parsed_data = parser.parse_receipt(
                    ocr_results["raw_text"],
                    ocr_results["lines"]
                )
                results["parsed_data"] = parsed_data
            except Exception as e:
                st.error(f"Parsing error: {e}")
                st.stop()

            # Step 4: Categorization
            progress_bar.progress(55, text="Step 4/7 - Categorizing expenses...")
            st.session_state.pipeline_step = 4
            try:
                categorizer = ExpenseCategorizer()
                categorization_data = categorizer.process(parsed_data)
                results["categorization_data"] = categorization_data
            except Exception as e:
                st.error(f"Categorization error: {e}")
                st.stop()

            # Step 5: Analysis
            progress_bar.progress(70, text="Step 5/7 - Analyzing spending patterns...")
            st.session_state.pipeline_step = 5
            try:
                detected_currency = parsed_data.get("currency", "$")
                analyzer = SpendingAnalyzer()
                analysis_data = analyzer.analyze_spending(categorization_data, currency=detected_currency)
                results["analysis_data"] = analysis_data
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.stop()

            # Step 6: LLM Advice
            progress_bar.progress(85, text="Step 6/7 - Generating AI insights...")
            st.session_state.pipeline_step = 6
            try:
                advisor = LLMAdvisor(api_key=api_key if api_key else None)
                llm_results = advisor.get_advice(categorization_data, analysis_data, currency=detected_currency)
                results["llm_results"] = llm_results
            except Exception as e:
                results["llm_results"] = {
                    "advice": "Could not generate AI advice.",
                    "status": "error",
                    "message": str(e)
                }

            # Step 7: Email Alert (if overspending detected)
            progress_bar.progress(95, text="Step 7/7 - Checking for overspending alerts...")
            st.session_state.pipeline_step = 7
            email_result = {"sent": False, "message": "Email alerts disabled or no overspending detected."}

            if email_alerts_enabled and alert_email and email_alerter.is_configured():
                if email_alerter.should_send_alert(analysis_data):
                    email_result = email_alerter.send_alert(
                        recipient_email=alert_email,
                        recipient_name=alert_name or "User",
                        categorization_data=categorization_data,
                        analysis_data=analysis_data,
                        currency=detected_currency
                    )
                else:
                    email_result = {"success": True, "message": "No overspending detected. No alert needed."}
            elif email_alerts_enabled and alert_email and not email_alerter.is_configured():
                email_result = {"success": False, "message": "Email SMTP not configured in .env file."}
            elif email_alerts_enabled and not alert_email:
                email_result = {"success": False, "message": "No recipient email provided."}

            results["email_result"] = email_result

            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(0.5)

            st.session_state.results = results
            st.session_state.processed = True
            st.session_state.pipeline_step = 7
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.processed and st.session_state.results:
    results = st.session_state.results
    parsed_data = results.get("parsed_data", {})
    categorization_data = results.get("categorization_data", {})
    analysis_data = results.get("analysis_data", {})
    ocr_results = results.get("ocr_results", {})
    llm_results = results.get("llm_results", {})
    processed_images = results.get("processed_images", {})

    basic_stats = analysis_data.get("basic_stats", {})
    health_score = analysis_data.get("health_score", {})
    category_percentages = categorization_data.get("category_percentages", {})
    categorized_items = categorization_data.get("categorized_items", [])
    currency = parsed_data.get("currency", "$")

    # ── EMAIL ALERT STATUS ──
    email_result = results.get("email_result", {})
    if email_result:
        if email_result.get("success") is True and "sent successfully" in email_result.get("message", ""):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #065F46, #047857); border-radius: 12px; 
                        padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;
                        border: 1px solid #10B981;">
                <span style="font-size: 1.5rem;">📧✅</span>
                <div>
                    <div style="color: #ECFDF5; font-weight: 700; font-size: 0.95rem;">Overspending Alert Email Sent!</div>
                    <div style="color: #A7F3D0; font-size: 0.85rem;">{email_result.get('message', '')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif email_result.get("success") is False and "No overspending" not in email_result.get("message", ""):
            msg = email_result.get("message", "")
            if "not configured" not in msg.lower() and "no recipient" not in msg.lower() and "disabled" not in msg.lower():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #7F1D1D, #991B1B); border-radius: 12px; 
                            padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;
                            border: 1px solid #EF4444;">
                    <span style="font-size: 1.5rem;">📧❌</span>
                    <div>
                        <div style="color: #FEF2F2; font-weight: 700; font-size: 0.95rem;">Email Alert Failed</div>
                        <div style="color: #FECACA; font-size: 0.85rem;">{msg}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TOP METRICS ──
    cols = st.columns(5)
    metrics = [
        ("💰", f"{currency}{basic_stats.get('grand_total', 0):,.2f}", "Total Spent", "#6366F1"),
        ("📦", str(basic_stats.get('total_items', 0)), "Items Found", "#06B6D4"),
        ("📊", str(categorization_data.get('total_categories', 0)), "Categories", "#22C55E"),
        ("💵", f"{currency}{basic_stats.get('avg_item_price', 0):,.2f}", "Avg. Price", "#F59E0B"),
        ("❤️", f"{health_score.get('score', 0)}/100", "Health Score", health_score.get('color', '#6366F1')),
    ]

    for col, (icon, value, label, color) in zip(cols, metrics):
        with col:
            st.markdown(render_metric_card(icon, value, label, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "🧾 Receipt Items",
        "🔬 Deep Analysis",
        "🤖 AI Advisor",
        "🖼️ Image Processing",
        "📄 Raw OCR"
    ])

    # ──────────────────────────────────
    # TAB 1: DASHBOARD
    # ──────────────────────────────────
    with tab1:
        render_section_header("📊", "Spending Dashboard", "OVERVIEW")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### 🍩 Spending Breakdown")
            if category_percentages:
                fig = create_donut_chart(category_percentages, currency=currency)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categories detected.")

        with col2:
            st.markdown("##### 📊 Category Comparison")
            if category_percentages:
                fig = create_bar_chart(category_percentages, currency=currency)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categories detected.")

        # Treemap
        st.markdown("##### 🗺️ Spending Treemap")
        if categorized_items:
            fig = create_treemap(categorized_items, currency=currency)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Health Score + Quick Insights
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("##### ❤️ Spending Health")
            fig = render_score_gauge(health_score)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <span style="color: {health_score.get('color', '#6366F1')}; font-size: 1.2rem; font-weight: 700;">
                    {health_score.get('label', 'N/A')}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### 💡 Quick Insights")
            insights = analysis_data.get("insights", [])
            for insight in insights:
                card_type = insight.get("type", "info")
                css_class = {
                    "warning": "warning",
                    "positive": "positive",
                    "danger": "danger"
                }.get(card_type, "")

                st.markdown(f"""
                <div class="insight-card {css_class}">
                    <span style="font-size: 1.2rem;">{insight.get('icon', 'ℹ️')}</span>
                    <span class="insight-text">{insight.get('message', '')}</span>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────
    # TAB 2: RECEIPT ITEMS
    # ──────────────────────────────────
    with tab2:
        render_section_header("🧾", "Extracted Receipt Items", f"{len(categorized_items)} ITEMS")

        if categorized_items:
            # Build dataframe
            df = pd.DataFrame(categorized_items)
            display_df = df[["name", "price", "quantity", "total", "category"]].copy()
            cur_label = currency if currency != "$" else "$"
            display_df.columns = ["Item Name", f"Unit Price ({cur_label})", "Qty", f"Total ({cur_label})", "Category"]
            display_df.index = range(1, len(display_df) + 1)

            # Style the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(400, 40 + len(display_df) * 35),
                column_config={
                    f"Unit Price ({cur_label})": st.column_config.NumberColumn(format="%.2f"),
                    f"Total ({cur_label})": st.column_config.NumberColumn(format="%.2f"),
                    "Qty": st.column_config.NumberColumn(format="%d"),
                }
            )

            # Summary row
            total_amount = sum(item["total"] for item in categorized_items)
            total_qty = sum(item["quantity"] for item in categorized_items)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E293B, #0F172A);
                        border: 1px solid #6366F1; border-radius: 12px;
                        padding: 1rem 1.5rem; margin-top: 1rem;
                        display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94A3B8; font-size: 0.9rem; font-weight: 600;">
                    GRAND TOTAL ({len(categorized_items)} items, {total_qty} units)
                </span>
                <span style="color: #22C55E; font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">
                    {currency}{total_amount:,.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Category breakdown table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📂 Category Breakdown")

            for cat, data in categorization_data.get("category_summary", {}).items():
                with st.expander(f"{cat} — {currency}{data['total_amount']:,.2f} ({data['item_count']} items)"):
                    for item in data["items"]:
                        st.markdown(f"- **{item['name']}** — {currency}{item['total']:,.2f} (x{item['quantity']})")

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Receipt Data (CSV)",
                data=csv,
                file_name="receipt_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("⚠️ No items could be extracted from the receipt. Try a clearer image.")

    # ──────────────────────────────────
    # TAB 3: DEEP ANALYSIS
    # ──────────────────────────────────
    with tab3:
        render_section_header("🔬", "Deep Spending Analysis", "ANALYTICS")

        # Statistics Cards
        st.markdown("##### 📈 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Median Price", f"{currency}{basic_stats.get('median_price', 0):,.2f}")
        with col2:
            st.metric("Max Price", f"{currency}{basic_stats.get('max_price', 0):,.2f}")
        with col3:
            st.metric("Min Price", f"{currency}{basic_stats.get('min_price', 0):,.2f}")
        with col4:
            st.metric("Std Deviation", f"{currency}{basic_stats.get('price_std', 0):,.2f}")

        # Overspending Alerts
        overspending = analysis_data.get("overspending_alerts", [])
        if overspending:
            st.markdown("##### ⚠️ Overspending Alerts")
            for alert in overspending:
                severity_color = "#EF4444" if alert["severity"] == "high" else "#F59E0B"
                severity_icon = "🔴" if alert["severity"] == "high" else "🟡"
                st.markdown(f"""
                <div class="insight-card {'danger' if alert['severity'] == 'high' else 'warning'}">
                    <span style="font-size: 1.1rem;">{severity_icon}</span>
                    <span class="insight-text">
                        <strong>{alert['category']}</strong>: {alert['actual_pct']}% of spending
                        (benchmark: {alert['benchmark_pct']}%) —
                        <span style="color: {severity_color};">{alert['deviation_pct']}% over</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
            # Manual email send button
            st.markdown("")
            send_col1, send_col2 = st.columns([2, 1])
            with send_col1:
                manual_email = st.text_input(
                    "📧 Send alert to email:",
                    value=alert_email if alert_email else "",
                    key="manual_alert_email",
                    placeholder="yourname@email.com"
                )
            with send_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚨 Send Alert Email Now", use_container_width=True, type="primary"):
                    if manual_email and email_alerter.is_configured():
                        with st.spinner("📧 Sending overspending alert..."):
                            result = email_alerter.send_alert(
                                recipient_email=manual_email,
                                recipient_name=alert_name or "User",
                                categorization_data=categorization_data,
                                analysis_data=analysis_data,
                                currency=currency
                            )
                            if result.get("success"):
                                st.success(f"✅ {result['message']}")
                            else:
                                st.error(f"❌ {result['message']}")
                    elif not email_alerter.is_configured():
                        st.error("❌ Email not configured. Set SENDER_EMAIL & SENDER_APP_PASSWORD in .env")
                    else:
                        st.warning("⚠️ Please enter a recipient email address.")
        else:
            st.success("✅ No significant overspending detected! No alert needed.")

        # Price Anomalies
        anomalies = analysis_data.get("price_anomalies", [])
        if anomalies:
            st.markdown("##### 🔎 Price Anomalies")
            for anomaly in anomalies:
                icon = "📈" if anomaly["type"] == "expensive" else "📉"
                st.markdown(f"""
                <div class="insight-card">
                    <span style="font-size: 1.1rem;">{icon}</span>
                    <span class="insight-text">{anomaly['message']} (z-score: {anomaly['z_score']})</span>
                </div>
                """, unsafe_allow_html=True)

        # Top Expensive Items
        top_items = analysis_data.get("top_expensive_items", [])
        if top_items:
            st.markdown("##### 💎 Most Expensive Items")

            fig = go.Figure(go.Bar(
                x=[item["total"] for item in top_items],
                y=[item["name"][:25] for item in top_items],
                orientation='h',
                marker=dict(
                    color=['#6366F1', '#818CF8', '#A5B4FC', '#C7D2FE', '#E0E7FF'][:len(top_items)],
                    cornerradius=6,
                ),
                text=[f'{currency}{item["total"]:,.2f}' for item in top_items],
                textposition='auto',
                textfont=dict(color='white', family='JetBrains Mono'),
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(51,65,85,0.5)', tickfont=dict(color='#94A3B8')),
                yaxis=dict(tickfont=dict(color='#E2E8F0'), autorange='reversed'),
                margin=dict(l=10, r=20, t=10, b=30),
                height=max(200, len(top_items) * 50),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Savings Opportunities
        savings = analysis_data.get("savings_opportunities", [])
        if savings:
            st.markdown("##### 💰 Savings Opportunities")
            for s in savings:
                st.markdown(f"""
                <div class="insight-card positive">
                    <span style="font-size: 1.1rem;">💡</span>
                    <span class="insight-text">{s['tip']}</span>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────
    # TAB 4: AI ADVISOR
    # ──────────────────────────────────
    with tab4:
        render_section_header("🤖", "AI Financial Advisor", "GEMINI" if llm_results.get("status") == "success" else "BUILT-IN")

        # Status badge
        if llm_results.get("status") == "success":
            st.markdown("""
            <div class="ai-badge">🤖 POWERED BY GOOGLE GEMINI</div>
            """, unsafe_allow_html=True)

        # Advice content
        st.markdown(f"""
        <div class="ai-advice-box">
        """, unsafe_allow_html=True)

        st.markdown(llm_results.get("advice", "No advice available."))

        st.markdown("</div>", unsafe_allow_html=True)

        # Regenerate button
        if st.button("🔄 Regenerate Advice", use_container_width=True):
            with st.spinner("🤖 Generating fresh AI insights..."):
                advisor = LLMAdvisor(api_key=api_key if api_key else None)
                new_results = advisor.get_advice(categorization_data, analysis_data, currency=currency)
                st.session_state.results["llm_results"] = new_results
                st.rerun()

    # ──────────────────────────────────
    # TAB 5: IMAGE PROCESSING
    # ──────────────────────────────────
    with tab5:
        render_section_header("🖼️", "Image Processing Pipeline", "COMPUTER VISION")

        processing_steps = processed_images.get("processing_steps", [])

        if processing_steps:
            st.markdown("##### 🔄 Processing Steps Visualization")
            st.markdown("See how each preprocessing step transforms the receipt image for optimal OCR:")

            step_cols = st.columns(min(3, len(processing_steps)))
            for i, (name, img) in enumerate(processing_steps):
                col_idx = i % len(step_cols)
                with step_cols[col_idx]:
                    st.markdown(f"**Step {i+1}: {name}**")
                    if len(img.shape) == 2:
                        st.image(img, use_container_width=True, clamp=True)
                    else:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(rgb_img, use_container_width=True, clamp=True)

        # OCR Visualization
        vis_image = ocr_results.get("visualization")
        if vis_image is not None:
            st.markdown("##### 🔍 OCR Detection Overlay")
            st.markdown("Green = high confidence, Yellow = medium, Red = low")
            rgb_vis = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            st.image(rgb_vis, use_container_width=True, clamp=True)

        # Confidence stats
        conf_stats = ocr_results.get("confidence_stats", {})
        if conf_stats:
            st.markdown("##### 📊 OCR Confidence Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Confidence", f"{conf_stats.get('mean', 0):.1%}")
            c2.metric("Min Confidence", f"{conf_stats.get('min', 0):.1%}")
            c3.metric("Max Confidence", f"{conf_stats.get('max', 0):.1%}")
            c4.metric("Low Confidence Blocks", conf_stats.get('low_confidence_count', 0))

    # ──────────────────────────────────
    # TAB 6: RAW OCR
    # ──────────────────────────────────
    with tab6:
        render_section_header("📄", "Raw OCR Output", "DEBUG")

        st.markdown("##### 📝 Extracted Text")
        raw_text = ocr_results.get("raw_text", "No text extracted")
        st.code(raw_text, language=None)

        st.markdown("##### 📋 Line-by-Line Breakdown")
        lines = ocr_results.get("lines", [])
        for i, line in enumerate(lines):
            st.markdown(f"`Line {i+1}:` {line}")

        # Store info
        store_info = parsed_data.get("store_info", {})
        if store_info:
            st.markdown("##### 🏪 Detected Store Information")
            for key, value in store_info.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Totals found
        totals = parsed_data.get("totals", {})
        if totals:
            st.markdown("##### 💰 Detected Totals")
            for key, value in totals.items():
                st.markdown(f"- **{key.title()}:** {currency}{value:,.2f}")

    # ── RESET BUTTON ──
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Another Receipt", use_container_width=True, type="primary"):
            st.session_state.processed = False
            st.session_state.results = None
            st.session_state.pipeline_step = 0
            st.rerun()


# ── FOOTER ──
st.markdown("""
<div class="footer">
    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">🧾 ReceiptIQ</div>
    <div>AI-Powered Receipt Analyzer with LLM Insights</div>
    <div style="margin-top: 0.3rem;">
        Built with OpenCV • EasyOCR • Google Gemini • Streamlit • Plotly
    </div>
    <div style="margin-top: 0.5rem; color: #475569;">
        © 2026 — Hackathon Project
    </div>
</div>
""", unsafe_allow_html=True)


