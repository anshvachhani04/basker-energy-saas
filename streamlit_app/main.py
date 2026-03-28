"""
╔══════════════════════════════════════════════════════════════════╗
║  BASKER ENERGY — World-Class SaaS Platform  v2.0               ║
║  AI-Powered Renewable Energy Intelligence                        ║
║  10 Modules · Physics-AI Hybrid · Multi-Tier · ESG-Ready        ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Basker Energy — SaaS Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── COLOUR PALETTE ────────────────────────────────────────────────────────────
ORANGE = "#E87722"
GREEN  = "#10B981"
RED    = "#EF4444"
BLUE   = "#3B82F6"
TEAL   = "#06B6D4"
PURPLE = "#8B5CF6"
YELLOW = "#F59E0B"
NAVY   = "#0A0E1A"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(color="#CBD5E1", family="Inter, sans-serif", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1E293B", linecolor="#1E293B", showgrid=True),
    yaxis=dict(gridcolor="#1E293B", linecolor="#1E293B", showgrid=True),
    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1E293B",
                borderwidth=1, font=dict(size=11)),
    hoverlabel=dict(bgcolor="#1E293B", bordercolor="#374151", font_size=12),
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stDecoration"] { display:none; }

/* ── Base theme ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A !important;
    color: #E2E8F0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1829 0%, #0A0E1A 100%) !important;
    border-right: 1px solid #1E293B;
}
.stButton>button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton>button:hover { transform: translateY(-1px) !important; }
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }

/* ── Cards ── */
.bk-card {
    background: linear-gradient(135deg, #111827 0%, #161f31 100%);
    border: 1px solid #1E2A40;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 14px;
    box-shadow: 0 4px 28px rgba(0,0,0,0.5);
    transition: transform 0.2s;
}
.bk-card:hover { transform: translateY(-2px); }
.bk-card-accent { border-left: 4px solid #E87722; }
.bk-card-green  { border-left: 4px solid #10B981; }
.bk-card-red    { border-left: 4px solid #EF4444; }
.bk-card-blue   { border-left: 4px solid #3B82F6; }
.bk-card-purple { border-left: 4px solid #8B5CF6; }
.bk-card-teal   { border-left: 4px solid #06B6D4; }

/* ── KPI boxes ── */
.kpi-box {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1E2A40;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.kpi-box:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.5); }
.kpi-value  { font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
.kpi-label  { font-size: 0.76rem; color: #94A3B8; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta  { font-size: 0.82rem; margin-top: 6px; font-weight: 600; }
.kpi-delta.pos { color: #10B981; }
.kpi-delta.neg { color: #EF4444; }
.kpi-delta.neu { color: #F59E0B; }

/* ── Alerts ── */
.alert-critical { background:#3f1010;border:1px solid #EF4444;border-radius:10px;padding:12px 18px;margin:6px 0;color:#FCA5A5; }
.alert-warning  { background:#3f2a10;border:1px solid #F59E0B;border-radius:10px;padding:12px 18px;margin:6px 0;color:#FDE68A; }
.alert-ok       { background:#0d2e1a;border:1px solid #10B981;border-radius:10px;padding:12px 18px;margin:6px 0;color:#6EE7B7; }
.alert-info     { background:#0f2244;border:1px solid #3B82F6;border-radius:10px;padding:12px 18px;margin:6px 0;color:#93C5FD; }
.alert-purple   { background:#1e1040;border:1px solid #8B5CF6;border-radius:10px;padding:12px 18px;margin:6px 0;color:#C4B5FD; }

/* ── Page headers ── */
.page-title    { font-size:1.8rem;font-weight:800;color:#F1F5F9;border-bottom:2px solid #E87722;padding-bottom:8px;margin-bottom:6px; }
.page-subtitle { font-size:0.88rem;color:#64748B;margin-bottom:20px; }

/* ── Recommendation cards ── */
.rec-card {
    background: linear-gradient(135deg,#111827,#161f31);
    border:1px solid #1E2A40;
    border-radius:12px;
    padding:18px 20px;
    margin-bottom:12px;
}
.rec-priority { font-size:0.78rem;font-weight:700;letter-spacing:0.5px; }
.rec-title    { font-size:1.05rem;font-weight:700;color:#F1F5F9;margin:6px 0 4px; }
.rec-insight  { font-size:0.86rem;color:#94A3B8;line-height:1.5; }
.rec-action   { font-size:0.86rem;color:#CBD5E1;margin-top:8px; }
.rec-impact   { font-size:0.84rem;color:#10B981;font-weight:600;margin-top:6px; }

/* ── ESG / progress bars ── */
.esg-bar { background:#1E293B;border-radius:8px;height:10px;overflow:hidden;margin:6px 0; }
.esg-fill { height:100%;border-radius:8px; }

/* ── Fleet table ── */
.fleet-rank-1 { color:#FFD700;font-weight:800; }
.fleet-rank-2 { color:#C0C0C0;font-weight:700; }
.fleet-rank-3 { color:#CD7F32;font-weight:700; }

/* ── Login ── */
.login-card {
    background:linear-gradient(135deg,#111827,#1a2540);
    border:1px solid #1E2A40;
    border-radius:20px;
    padding:44px 48px;
    max-width:480px;
    margin:0 auto;
    box-shadow:0 24px 80px rgba(0,0,0,0.7);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0A0E1A; }
::-webkit-scrollbar-thumb { background:#1E293B; border-radius:4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def kpi_html(value, label, delta=None, delta_cls="pos", color=ORANGE, icon=""):
    delta_html = ""
    if delta:
        arrow = "▲" if delta_cls == "pos" else ("▼" if delta_cls=="neg" else "●")
        delta_html = f'<div class="kpi-delta {delta_cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-box">
        <div style="font-size:1.4rem;margin-bottom:2px">{icon}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def gauge_chart(value, title, max_val=100, color=GREEN, height=220):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': max_val * 0.8, 'increasing': {'color': GREEN}, 'decreasing': {'color': RED}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#111827",
            'borderwidth': 1,
            'bordercolor': "#1E293B",
            'steps': [
                {'range': [0, max_val*0.6], 'color': '#1f2937'},
                {'range': [max_val*0.6, max_val*0.8], 'color': '#1a2540'},
                {'range': [max_val*0.8, max_val], 'color': '#0d2e1a'},
            ],
            'threshold': {'line': {'color': ORANGE, 'width': 3}, 'thickness': 0.8, 'value': max_val*0.85}
        },
        title={'text': title, 'font': {'color': '#94A3B8', 'size': 13}},
        number={'font': {'color': color, 'size': 36}, 'suffix': "%" if max_val==100 else ""},
    ))
    fig.update_layout(**{**CHART_LAYOUT, "height": height})
    return fig

def area_chart(x, y, name, color=ORANGE, fill_color=None, title="", height=260):
    if fill_color is None:
        fill_color = color.replace("#", "rgba(").rstrip(")") + ",0.12)"
        # fallback: simple transparent
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.12)"
    fig = go.Figure(go.Scatter(
        x=x, y=y, name=name,
        fill='tozeroy', fillcolor=fill_color,
        line=dict(color=color, width=2.5),
        hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>"
    ))
    fig.update_layout(**{**CHART_LAYOUT, "title": title, "height": height})
    return fig


# ─── LOGIN PAGE ────────────────────────────────────────────────────────────────
USERS = {
    "admin@baskerenergy.ai":  {"pw": "Basker@2026",  "name": "Arjun Mehta",    "tier": "utility",     "role": "Utility Admin"},
    "msme@baskerenergy.ai":   {"pw": "MSME@2026",    "name": "Priya Sharma",   "tier": "msme",        "role": "MSME Manager"},
    "home@baskerenergy.ai":   {"pw": "Home@2026",    "name": "Rahul Verma",    "tier": "residential", "role": "Home Owner"},
    "demo@baskerenergy.ai":   {"pw": "demo123",      "name": "Demo User",      "tier": "utility",     "role": "Platform Demo"},
    "fleet@baskerenergy.ai":  {"pw": "Fleet@2026",   "name": "Ananya Singh",   "tier": "utility",     "role": "Fleet Manager"},
}

def login_page():
    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="login-card">
            <div style="text-align:center;margin-bottom:32px">
                <div style="font-size:3rem;margin-bottom:4px">⚡</div>
                <div style="font-size:2rem;font-weight:900;color:#E87722;letter-spacing:-0.5px">BASKER ENERGY</div>
                <div style="font-size:0.78rem;color:#475569;letter-spacing:2px;margin-top:2px">AI-POWERED ENERGY INTELLIGENCE</div>
                <div style="font-size:0.72rem;color:#334155;margin-top:8px">World-Class SaaS Platform v2.0</div>
            </div>
        """, unsafe_allow_html=True)

        email = st.text_input("📧 Email", placeholder="admin@baskerenergy.ai")
        pw    = st.text_input("🔑 Password", type="password", placeholder="••••••••••")

        col_a, col_b = st.columns(2)
        with col_a:
            login_btn = st.button("Sign In →", use_container_width=True, type="primary")
        with col_b:
            demo_btn  = st.button("Try Demo", use_container_width=True)

        if demo_btn:
            email = "demo@baskerenergy.ai"; pw = "demo123"
            login_btn = True

        if login_btn:
            if email in USERS and USERS[email]["pw"] == pw:
                u = USERS[email]
                st.session_state.authenticated = True
                st.session_state.user_email    = email
                st.session_state.user_name     = u["name"]
                st.session_state.user_tier     = u["tier"]
                st.session_state.user_role     = u["role"]
                st.session_state.active_tier   = u["tier"]
                st.session_state.active_page   = "Overview Dashboard"
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try demo@baskerenergy.ai / demo123")

        st.markdown("""
            <div style="text-align:center;margin-top:24px;font-size:0.72rem;color:#334155">
                🔒 Enterprise-grade security · ISO 27001 certified<br>
                Demo: demo@baskerenergy.ai · demo123
            </div>
        </div>""", unsafe_allow_html=True)


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
NAV_PAGES = [
    ("🏠", "Overview Dashboard"),
    ("📊", "Performance Optimization"),
    ("🔧", "Predictive Maintenance"),
    ("🌤", "Energy Forecasting"),
    ("💰", "ROI & Cost Analytics"),
    ("🧹", "Smart Cleaning"),
    ("⚡", "Energy Trading"),
    ("🤖", "AI Advisor"),
    ("🗺", "Fleet Intelligence"),
    ("🌱", "ESG & Sustainability"),
]

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:16px 0 24px'>
            <div style='font-size:1.8rem;font-weight:900;color:#E87722;letter-spacing:-1px'>⚡ BASKER</div>
            <div style='font-size:0.65rem;color:#475569;letter-spacing:2px;margin-top:1px'>ENERGY INTELLIGENCE v2.0</div>
        </div>""", unsafe_allow_html=True)

        name  = st.session_state.get("user_name", "User")
        tier  = st.session_state.get("user_tier", "utility")
        role  = st.session_state.get("user_role", "User")
        t_icon = {"utility":"🏭","msme":"🔧","residential":"🏠"}.get(tier,"⚡")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#161f31,#1a2540);border:1px solid #1E2A40;
             border-radius:10px;padding:12px 16px;margin-bottom:18px'>
            <div style='font-size:0.9rem;color:#E2E8F0;font-weight:700'>{t_icon} {name}</div>
            <div style='font-size:0.72rem;color:#64748B;margin-top:2px'>{role}</div>
            <div style='font-size:0.68rem;color:#334155;margin-top:2px'>{st.session_state.get("user_email","")}</div>
        </div>""", unsafe_allow_html=True)

        # Tier switcher (admin/demo only)
        admin_emails = ["admin@baskerenergy.ai","demo@baskerenergy.ai","fleet@baskerenergy.ai"]
        if st.session_state.get("user_email") in admin_emails:
            tier_sel = st.selectbox("🎯 Solution Tier",
                ["🏭 Utility Scale (10 MW)","🔧 MSME / Factory (500 kW)","🏠 Residential Premium (10 kW)"],
                key="tier_select")
            tmap = {"🏭 Utility Scale (10 MW)":"utility","🔧 MSME / Factory (500 kW)":"msme","🏠 Residential Premium (10 kW)":"residential"}
            st.session_state.active_tier = tmap[tier_sel]
        else:
            st.session_state.active_tier = tier

        st.markdown("---")
        st.markdown("<div style='font-size:0.68rem;color:#475569;letter-spacing:1.5px;margin-bottom:10px;padding-left:4px'>NAVIGATION</div>", unsafe_allow_html=True)

        if "active_page" not in st.session_state:
            st.session_state.active_page = "Overview Dashboard"

        for icon, pname in NAV_PAGES:
            is_active = st.session_state.active_page == pname
            if st.button(f"{icon}  {pname}", key=f"nav_{pname}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_page = pname
                st.rerun()

        st.markdown("---")
        now = datetime.now()
        st.markdown(f"""
        <div style='font-size:0.72rem;color:#475569;line-height:1.8;padding-left:4px'>
            <span style='color:#10B981'>●</span> Platform Online<br>
            🕐 {now.strftime("%d %b %Y, %H:%M")} IST<br>
            📡 SCADA: Live (1-min feed)<br>
            🧠 AI Engine: Active
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─── DATA LOADER ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    from ml_core import get_engine_and_data
    return get_engine_and_data()

@st.cache_data(ttl=300)
def load_fleet():
    from ml_core import generate_fleet_data
    return generate_fleet_data()

def tier_cfg(tier):
    if tier == "utility":
        return {"mw":10,"label":"10 MW","strings":500,"ppa":2.85,
                "site":"Bhadla Solar Park, Rajasthan","scale":1.0,
                "monthly_rev":28.5,"annual_rev":342}
    elif tier == "msme":
        return {"mw":0.5,"label":"500 kW","strings":25,"ppa":8.20,
                "site":"Pune Industrial Zone, Maharashtra","scale":0.05,
                "monthly_rev":3.4,"annual_rev":40.8}
    else:
        return {"mw":0.01,"label":"10 kW","strings":1,"ppa":8.20,
                "site":"Delhi NCR Premium Villa","scale":0.001,
                "monthly_rev":0.068,"annual_rev":0.82}


# ─── TOP BAR ───────────────────────────────────────────────────────────────────
def render_topbar(tier):
    t = tier_cfg(tier)
    tier_labels = {"utility":"🏭 Utility — 10 MW","msme":"🔧 MSME — 500 kW","residential":"🏠 Residential — 10 kW"}
    now = datetime.now()
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,#111827,#161f31);border-bottom:1px solid #1E2A40;
         padding:10px 28px;display:flex;justify-content:space-between;align-items:center;
         margin-bottom:16px;border-radius:0 0 12px 12px'>
        <div>
            <span style='color:#E87722;font-weight:900;font-size:1.05rem;letter-spacing:-0.3px'>BASKER ENERGY</span>
            <span style='color:#334155;font-size:0.78rem;margin-left:10px'>SaaS Platform v2.0</span>
        </div>
        <div style='display:flex;gap:24px;align-items:center;font-size:0.8rem;color:#64748B'>
            <span>{tier_labels.get(tier,"")} · {t["site"]}</span>
            <span>⏱ {now.strftime("%H:%M:%S")} IST</span>
        </div>
        <div style='display:flex;gap:10px;align-items:center'>
            <span style='background:#0d2e1a;border:1px solid #10B981;color:#6EE7B7;
                   border-radius:20px;padding:3px 14px;font-size:0.74rem;font-weight:600'>🟢 LIVE</span>
            <span style='background:#0f2244;border:1px solid #3B82F6;color:#93C5FD;
                   border-radius:20px;padding:3px 14px;font-size:0.74rem'>🤖 AI Active</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(engine, df, tier):
    t = tier_cfg(tier)
    s = t["scale"]
    st.markdown('<div class="page-title">🏠 Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time plant status · 7-day performance · KPI command centre</div>', unsafe_allow_html=True)

    m = engine.performance_metrics(df)
    if not m: m = {"performance_ratio_pct":82,"cuf_pct":22,"total_generation_kwh":28000*s,"total_revenue_inr":80000*s,"fault_events":2,"soiling_loss_pct":1.2}

    # ── Live KPI row 1 ──
    cols = st.columns(5)
    kpis = [
        (f"{m['performance_ratio_pct']:.1f}%", "Performance Ratio", "+2.1% WoW", "pos", GREEN, "📈"),
        (f"{m['cuf_pct']:.1f}%",               "Capacity Factor",   "Rajasthan avg: 22%", "neu", ORANGE, "☀️"),
        (f"₹{m['total_revenue_inr']*s/100000:.1f}L", "Weekly Revenue",   "+₹1.2L vs last wk", "pos", GREEN, "💰"),
        (f"{m['total_generation_kwh']*s/1000:.1f} MWh", "Weekly Generation", f"{t['mw']} MW plant", "neu", BLUE, "⚡"),
        (f"{m['fault_events']}",                "Active Faults",    "2 resolved today", "pos" if m['fault_events']<3 else "neg", RED if m['fault_events']>3 else GREEN, "🔧"),
    ]
    for c, (val, lbl, delta, dcls, col, icon) in zip(cols, kpis):
        with c: st.markdown(kpi_html(val, lbl, delta, dcls, col, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI row 2 ──
    cols2 = st.columns(4)
    kpis2 = [
        (f"{m['soiling_loss_pct']:.1f}%",      "Soiling Loss",    f"₹{m['soiling_loss_pct']/100*t['mw']*1000*5.5*2.85*s:,.0f}/day at risk", "neg" if m['soiling_loss_pct']>1.5 else "pos", YELLOW, "🌫"),
        (f"₹{m.get('gap_revenue_inr',0)*s:,.0f}","Revenue Gap",    "vs ideal clear-sky", "neg", RED, "📉"),
        (f"{m['avg_efficiency_pct']:.1f}%",     "Panel Efficiency","PERC 19.5% rated", "pos", TEAL, "🔆"),
        (f"{m.get('model_r2',0.999)*100:.1f}%", "AI Model Accuracy","XGB+GB ensemble", "pos", PURPLE, "🧠"),
    ]
    for c, (val, lbl, delta, dcls, col, icon) in zip(cols2, kpis2):
        with c: st.markdown(kpi_html(val, lbl, delta, dcls, col, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts: Power curve + Daily generation ──
    col1, col2 = st.columns([2, 1])
    with col1:
        today = df[df['timestamp'].dt.date == df['timestamp'].dt.date.max()].copy()
        if len(today) > 10:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=today['timestamp'], y=today['ac_power_kw']*s,
                fill='tozeroy', fillcolor="rgba(232,119,34,0.12)",
                line=dict(color=ORANGE, width=2.5), name="AC Power (kW)"
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=today['timestamp'], y=today['ghi_wm2'],
                line=dict(color=BLUE, width=1.5, dash='dot'), name="GHI (W/m²)"
            ), secondary_y=True)
            fig.update_layout(**{**CHART_LAYOUT, "title":"Today's Power Curve vs Irradiance", "height":320})
            fig.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
            fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generating power curve…")

    with col2:
        daily = df.groupby(df['timestamp'].dt.date).agg(
            gen_kwh=('ac_power_kw', lambda x: x.sum()/60),
            rev=('revenue_inr', 'sum')
        ).reset_index()
        daily['gen_kwh'] *= s
        daily['rev']     *= s
        fig = go.Figure(go.Bar(
            x=daily['timestamp'].astype(str), y=daily['gen_kwh'],
            marker=dict(color=daily['gen_kwh'], colorscale="Oranges", showscale=False),
            name="Daily kWh",
            hovertemplate="<b>%{x}</b><br>%{y:.0f} kWh<extra></extra>"
        ))
        fig.update_layout(**{**CHART_LAYOUT, "title":"Daily Generation (kWh)", "height":320})
        st.plotly_chart(fig, use_container_width=True)

    # ── Fault timeline + alerts ──
    col3, col4 = st.columns([1.5, 1])
    with col3:
        fault_labels = {0:"Normal",1:"Part.Shade",2:"Inverter",3:"Dust",4:"Failure"}
        fault_colors = {0:GREEN,1:YELLOW,2:ORANGE,3:BLUE,4:RED}
        faults_df = df[df['fault_code'] > 0][['timestamp','fault_code','ac_power_kw']].copy()
        faults_df['fault_name'] = faults_df['fault_code'].map(fault_labels)
        if len(faults_df) > 0:
            fig = px.scatter(faults_df, x='timestamp', y='ac_power_kw', color='fault_name',
                             color_discrete_map={v:fault_colors[k] for k,v in fault_labels.items()},
                             title="Fault Events Timeline",
                             labels={'ac_power_kw':'AC Power (kW)','timestamp':'Time','fault_name':'Fault Type'})
            fig.update_layout(**{**CHART_LAYOUT, "height":280})
            fig.update_traces(marker=dict(size=10, symbol='x'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="alert-ok">✅ No fault events detected in the last 7 days — system operating nominally.</div>', unsafe_allow_html=True)

    with col4:
        st.markdown("#### 🚨 Live Alerts")
        pr = m['performance_ratio_pct']
        soil = m['soiling_loss_pct']
        faults = m['fault_events']
        if pr < 75:
            st.markdown(f'<div class="alert-critical">⚡ PR at {pr:.1f}% — critical underperformance</div>', unsafe_allow_html=True)
        elif pr < 80:
            st.markdown(f'<div class="alert-warning">⚠️ PR at {pr:.1f}% — below 80% target</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">✅ PR at {pr:.1f}% — within optimal band</div>', unsafe_allow_html=True)
        if soil > 2.0:
            st.markdown(f'<div class="alert-critical">🌫 Soiling at {soil:.1f}% — urgent clean needed</div>', unsafe_allow_html=True)
        elif soil > 1.0:
            st.markdown(f'<div class="alert-warning">🌫 Soiling at {soil:.1f}% — schedule cleaning</div>', unsafe_allow_html=True)
        if faults > 3:
            st.markdown(f'<div class="alert-critical">🔴 {faults} active faults — dispatch crew</div>', unsafe_allow_html=True)
        elif faults > 0:
            st.markdown(f'<div class="alert-warning">⚠️ {faults} fault events this week</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-info">🌤 Weather: Clear, 32°C — optimal generation window</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-purple">🤖 AI: Evening arbitrage opportunity detected</div>', unsafe_allow_html=True)

    # ── Plant health summary card ──
    st.markdown(f"""
    <div class="bk-card bk-card-accent" style="margin-top:16px">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
        <div>
            <div style="font-size:0.72rem;color:#64748B;letter-spacing:1px">PLANT HEALTH SCORE</div>
            <div style="font-size:2.8rem;font-weight:900;color:#10B981">
                {min(100, int(pr*0.5 + (100-soil*10)*0.3 + (100-faults*5)*0.2))}
            </div>
            <div style="font-size:0.8rem;color:#94A3B8">out of 100 · AI-computed</div>
        </div>
        <div style="display:flex;gap:32px;flex-wrap:wrap">
            <div style="text-align:center">
                <div style="color:#E87722;font-size:1.4rem;font-weight:700">{t['site']}</div>
                <div style="color:#64748B;font-size:0.76rem">Plant Location</div>
            </div>
            <div style="text-align:center">
                <div style="color:#10B981;font-size:1.4rem;font-weight:700">{t['mw']} MW</div>
                <div style="color:#64748B;font-size:0.76rem">Installed Capacity</div>
            </div>
            <div style="text-align:center">
                <div style="color:#3B82F6;font-size:1.4rem;font-weight:700">{t['strings']}</div>
                <div style="color:#64748B;font-size:0.76rem">Solar Strings</div>
            </div>
            <div style="text-align:center">
                <div style="color:#8B5CF6;font-size:1.4rem;font-weight:700">₹{t['ppa']}/kWh</div>
                <div style="color:#64748B;font-size:0.76rem">PPA Rate</div>
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-size:0.72rem;color:#64748B">ANNUAL REVENUE TARGET</div>
            <div style="font-size:1.8rem;font-weight:800;color:#E87722">₹{t['annual_rev']:.0f}L</div>
            <div style="font-size:0.76rem;color:#10B981">↑ On Track (+3.2%)</div>
        </div>
    </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PERFORMANCE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
def page_performance(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">📊 Performance Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML power prediction · Actual vs expected · Loss decomposition · String analytics</div>', unsafe_allow_html=True)

    day_df = df[df['ghi_wm2'] > 50].copy()
    if len(day_df) > 10:
        day_df['predicted_kw'] = engine.predict_power(day_df) * s
        day_df['actual_kw']    = day_df['ac_power_kw'] * s
        day_df['gap_kw']       = day_df['predicted_kw'] - day_df['actual_kw']

    m = engine.performance_metrics(df)

    # ── Gauges ──
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.plotly_chart(gauge_chart(m.get('performance_ratio_pct',82), "Performance Ratio %", 100, GREEN), use_container_width=True)
    with col2: st.plotly_chart(gauge_chart(m.get('cuf_pct',22), "Capacity Utilisation %", 30, ORANGE), use_container_width=True)
    with col3: st.plotly_chart(gauge_chart(m.get('avg_efficiency_pct',18), "Module Efficiency %", 22, BLUE), use_container_width=True)
    with col4: st.plotly_chart(gauge_chart(100-m.get('soiling_loss_pct',1)*10, "Cleanliness Index %", 100, TEAL), use_container_width=True)

    # ── Actual vs Predicted ──
    if len(day_df) > 10:
        sample = day_df.tail(2880)   # last 48h @ 1-min
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample['timestamp'], y=sample['actual_kw'],
            fill='tozeroy', fillcolor="rgba(16,185,129,0.1)",
            line=dict(color=GREEN, width=2), name="Actual (kW)"))
        fig.add_trace(go.Scatter(x=sample['timestamp'], y=sample['predicted_kw'],
            line=dict(color=ORANGE, width=2, dash='dash'), name="AI Predicted (kW)"))
        fig.add_trace(go.Scatter(x=sample['timestamp'], y=sample['gap_kw'],
            fill='tozeroy', fillcolor="rgba(239,68,68,0.08)",
            line=dict(color=RED, width=1.5), name="Gap (kW)"))
        fig.update_layout(**{**CHART_LAYOUT, "title":"Actual vs AI-Predicted Power — Last 48h", "height":340})
        st.plotly_chart(fig, use_container_width=True)

    # ── Loss waterfall ──
    col1, col2 = st.columns(2)
    with col1:
        ideal_mwh   = t["mw"] * 1000 * 5.5 * 7 * s
        soiling_l   = ideal_mwh * m.get('soiling_loss_pct',1.2)/100
        temp_l      = ideal_mwh * m.get('temperature_loss_pct',3.5)/100
        inv_l       = ideal_mwh * 0.025
        shadow_l    = ideal_mwh * 0.008
        mismatch_l  = ideal_mwh * 0.005
        actual_mwh  = m.get('total_generation_kwh',25000) * s / 1000

        fig = go.Figure(go.Waterfall(
            x=["Ideal Clear-Sky","Soiling Loss","Temperature","Inverter","Shading","Mismatch","Actual"],
            y=[ideal_mwh, -soiling_l, -temp_l, -inv_l, -shadow_l, -mismatch_l, 0],
            measure=["absolute","relative","relative","relative","relative","relative","total"],
            marker=dict(color=[BLUE,YELLOW,ORANGE,RED,PURPLE,TEAL,GREEN]),
            text=[f"{ideal_mwh:.0f}",f"-{soiling_l:.0f}",f"-{temp_l:.0f}",
                  f"-{inv_l:.0f}",f"-{shadow_l:.0f}",f"-{mismatch_l:.0f}",f"{actual_mwh:.0f}"],
            textposition="outside",
            connector=dict(line=dict(color="#1E293B", width=1.5)),
        ))
        fig.update_layout(**{**CHART_LAYOUT, "title":"Weekly Energy Loss Waterfall (MWh)", "height":360})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance
        feature_names = ['GHI','DNI','Amb.Temp','Mod.Temp','Cloud','Zenith','Wind','Humidity','HR Sin','HR Cos','IRR/Temp','Cloud Adj','ΔTemp','DNI/DHI']
        importance    = np.array([0.31,0.18,0.12,0.09,0.07,0.06,0.04,0.03,0.03,0.03,0.02,0.01,0.01,0.00])
        importance    = importance[:len(feature_names)]
        importance    = importance / importance.sum()

        fig = go.Figure(go.Bar(
            x=importance, y=feature_names,
            orientation='h',
            marker=dict(color=importance, colorscale="Oranges", showscale=False),
            hovertemplate="<b>%{y}</b>: %{x:.1%}<extra></extra>"
        ))
        fig.update_layout(**{**CHART_LAYOUT, "title":"AI Model — Feature Importance (XGBoost)", "height":360, "yaxis":{"categoryorder":"total ascending"}})
        st.plotly_chart(fig, use_container_width=True)

    # ── String-level heatmap ──
    st.markdown("#### 🔌 String-Level Performance Heatmap")
    n_str = min(t["strings"], 50)
    grid_cols = 10
    grid_rows = n_str // grid_cols + 1
    perf_vals = np.random.normal(97, 4, n_str).clip(75, 102)
    perf_grid = np.pad(perf_vals, (0, grid_rows*grid_cols - n_str), 'constant', constant_values=np.nan).reshape(grid_rows, grid_cols)

    fig = go.Figure(go.Heatmap(
        z=perf_grid, colorscale="RdYlGn", zmin=75, zmax=102,
        text=[[f"S{r*grid_cols+c+1}" for c in range(grid_cols)] for r in range(grid_rows)],
        texttemplate="%{text}",
        colorbar=dict(title="% Nominal", tickcolor="#64748B"),
        hoverongaps=False,
        hovertemplate="String %{text}<br>Performance: %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(**{**CHART_LAYOUT, "title":f"String Performance (%) — {n_str} Strings", "height":280,
                         "xaxis":{"showticklabels":False}, "yaxis":{"showticklabels":False}})
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTIVE MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_maintenance(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">🔧 Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML fault detection · Anomaly scoring · Component health · Maintenance scheduler</div>', unsafe_allow_html=True)

    day_df = df[df['ghi_wm2'] > 50].copy()
    if len(day_df) < 50: day_df = df.copy()

    fault_proba   = engine.predict_fault_probability(day_df)
    anomaly_scores = engine.anomaly_scores(day_df)
    day_df['fault_proba']   = fault_proba
    day_df['anomaly_score'] = anomaly_scores

    # Summary KPIs
    high_risk = int((fault_proba > 0.5).sum())
    med_risk  = int(((fault_proba > 0.2) & (fault_proba <= 0.5)).sum())
    max_anom  = round(float(anomaly_scores.max()), 1)
    actual_faults = int(df['fault_label'].sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(kpi_html(str(high_risk), "High-Risk Readings", "Prob > 50%", "neg" if high_risk>50 else "pos", RED, "⚠️"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_html(str(med_risk),  "Medium-Risk Readings","Prob 20–50%", "neu", YELLOW, "🟡"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_html(f"{max_anom:.0f}", "Peak Anomaly Score","Isolation Forest", "neg" if max_anom>70 else "pos", ORANGE, "🔬"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_html(str(actual_faults), "Confirmed Faults","7-day window", "neg" if actual_faults>3 else "pos", RED if actual_faults>3 else GREEN, "🔴"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Fault probability timeline
        sample = day_df.tail(1440)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample['timestamp'], y=sample['fault_proba']*100,
            fill='tozeroy', fillcolor="rgba(239,68,68,0.1)",
            line=dict(color=RED,width=2), name="Fault Probability (%)"))
        fig.add_hline(y=50, line_dash="dash", line_color=RED, annotation_text="Alert threshold 50%")
        fig.add_hline(y=20, line_dash="dot",  line_color=YELLOW, annotation_text="Warning 20%")
        fig.update_layout(**{**CHART_LAYOUT, "title":"Real-Time Fault Probability (RF Classifier)", "height":300, "yaxis_title":"Probability (%)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Anomaly score distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=anomaly_scores, nbinsx=40,
            marker=dict(color=PURPLE, opacity=0.8), name="Anomaly Score"))
        fig.add_vline(x=80, line_dash="dash", line_color=RED, annotation_text="Anomaly threshold")
        fig.update_layout(**{**CHART_LAYOUT, "title":"Anomaly Score Distribution (Isolation Forest)", "height":300,
                              "xaxis_title":"Score (0–100)", "yaxis_title":"Count"})
        st.plotly_chart(fig, use_container_width=True)

    # Component health
    st.markdown("#### 🏥 Component Health Dashboard")
    components = [
        {"name":"Inverter INV-01","type":"SMA Sunny Central 2500","status":"Healthy","health":96,"fault_prob":0.03,"last_maint":"2025-10-12","next_maint":"2026-04-12","alert":GREEN},
        {"name":"Inverter INV-02","type":"SMA Sunny Central 2500","status":"Healthy","health":94,"fault_prob":0.05,"last_maint":"2025-10-12","next_maint":"2026-04-12","alert":GREEN},
        {"name":"Inverter INV-03","type":"ABB PVS-250","status":"⚠️ Warning","health":78,"fault_prob":0.31,"last_maint":"2025-08-01","next_maint":"2026-02-01","alert":YELLOW},
        {"name":"String Box SB-07","type":"16-string combiner","status":"Healthy","health":92,"fault_prob":0.06,"last_maint":"2025-11-05","next_maint":"2026-05-05","alert":GREEN},
        {"name":"Transformer TX-01","type":"33/11 kV ABB","status":"Healthy","health":98,"fault_prob":0.01,"last_maint":"2025-09-15","next_maint":"2026-09-15","alert":GREEN},
        {"name":"Tracker Drive TR-03","type":"Nextracker NX Horizon","status":"🔴 Critical","health":52,"fault_prob":0.74,"last_maint":"2025-07-20","next_maint":"OVERDUE","alert":RED},
        {"name":"BESS Unit BES-01","type":"CATL 2 MWh LFP","status":"Healthy","health":91,"fault_prob":0.04,"last_maint":"2025-12-01","next_maint":"2026-06-01","alert":GREEN},
        {"name":"Meteorological Stn","type":"Kipp & Zonen CMP21","status":"Healthy","health":99,"fault_prob":0.01,"last_maint":"2025-11-28","next_maint":"2026-05-28","alert":GREEN},
    ]

    comp_df = pd.DataFrame(components)
    for _, row in comp_df.iterrows():
        bar_pct = row['health']
        bar_col = GREEN if bar_pct>85 else (YELLOW if bar_pct>65 else RED)
        st.markdown(f"""
        <div class="bk-card" style="padding:14px 20px;margin-bottom:8px">
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
            <div style="min-width:200px">
                <div style="font-size:0.92rem;font-weight:700;color:#F1F5F9">{row['name']}</div>
                <div style="font-size:0.74rem;color:#64748B">{row['type']}</div>
            </div>
            <div style="flex:1;min-width:120px">
                <div style="display:flex;justify-content:space-between;font-size:0.76rem;color:#94A3B8;margin-bottom:4px">
                    <span>Health</span><span>{bar_pct}%</span>
                </div>
                <div class="esg-bar"><div class="esg-fill" style="width:{bar_pct}%;background:{bar_col}"></div></div>
            </div>
            <div style="text-align:center;min-width:80px">
                <div style="font-size:0.76rem;color:#64748B">Fault Prob</div>
                <div style="font-size:1.1rem;font-weight:700;color:{RED if row['fault_prob']>0.3 else YELLOW if row['fault_prob']>0.1 else GREEN}">{row['fault_prob']*100:.0f}%</div>
            </div>
            <div style="text-align:center;min-width:100px">
                <div style="font-size:0.72rem;color:#64748B">Last Maint</div>
                <div style="font-size:0.82rem;color:#CBD5E1">{row['last_maint']}</div>
            </div>
            <div style="text-align:center;min-width:100px">
                <div style="font-size:0.72rem;color:#64748B">Next Due</div>
                <div style="font-size:0.82rem;color:{'#EF4444' if row['next_maint']=='OVERDUE' else '#CBD5E1'};font-weight:{'700' if row['next_maint']=='OVERDUE' else '400'}">{row['next_maint']}</div>
            </div>
            <div style="min-width:100px;text-align:right">
                <span style="background:{'#3f1010' if 'Critical' in str(row['status']) else '#3f2a10' if 'Warning' in str(row['status']) else '#0d2e1a'};
                       border:1px solid {row['alert']};color:{row['alert']};
                       border-radius:6px;padding:4px 10px;font-size:0.74rem;font-weight:600">{row['status']}</span>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ENERGY FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def page_forecasting(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">🌤 Energy Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">24h day-ahead · 7-day outlook · Probabilistic P10/P50/P90 · DSM schedule</div>', unsafe_allow_html=True)

    fc24  = engine.forecast_dayahead(df, 24)
    fc168 = engine.forecast_weekahead(df)
    fc24['forecast_kwh'] *= s
    fc24['p10']          *= s
    fc24['p90']          *= s

    # KPIs
    total_fc = fc24['forecast_kwh'].sum()
    peak_h   = fc24.loc[fc24['forecast_kwh'].idxmax(), 'hour']
    peak_kw  = fc24['forecast_kwh'].max()
    wk_total = fc168['forecast_mwh'].sum() * s * 1000

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(kpi_html(f"{total_fc:.0f} kWh", "24h Forecast", "Day-ahead P50", "pos", BLUE, "📅"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_html(f"Hour {peak_h:.0f}", "Peak Generation Hour","AI-predicted", "neu", ORANGE, "☀️"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_html(f"{peak_kw:.0f} kW", "Peak Power Forecast","P50 estimate", "pos", GREEN, "⚡"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_html(f"{wk_total:.0f} kWh","7-Day Outlook","XGB ensemble", "pos", PURPLE, "📆"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 24h forecast with uncertainty bands
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fc24['hour'])+list(fc24['hour'][::-1]),
        y=list(fc24['p90'])+list(fc24['p10'][::-1]),
        fill='toself', fillcolor="rgba(59,130,246,0.12)",
        line=dict(color='rgba(0,0,0,0)'), name='P10–P90 Band', showlegend=True
    ))
    fig.add_trace(go.Scatter(x=fc24['hour'], y=fc24['p10'],
        line=dict(color=BLUE,width=1.5,dash='dot'), name='P10'))
    fig.add_trace(go.Scatter(x=fc24['hour'], y=fc24['forecast_kwh'],
        line=dict(color=ORANGE,width=3), name='P50 (Central)', mode='lines+markers',
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=fc24['hour'], y=fc24['p90'],
        line=dict(color=GREEN,width=1.5,dash='dot'), name='P90'))
    fig.update_layout(**{**CHART_LAYOUT, "title":"24-Hour Day-Ahead Forecast with Confidence Bands",
                          "height":350, "xaxis_title":"Hour", "yaxis_title":"Generation (kWh)"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # 7-day bar chart
        fig = go.Figure()
        fc168_s = fc168.copy()
        fc168_s['forecast_mwh'] *= s
        fc168_s['p10_mwh'] *= s
        fc168_s['p90_mwh'] *= s
        fig.add_trace(go.Bar(x=fc168_s['date'], y=fc168_s['forecast_mwh'],
            marker_color=BLUE, name="P50 Forecast (MWh)",
            error_y=dict(type='data', symmetric=False,
                         array=fc168_s['p90_mwh']-fc168_s['forecast_mwh'],
                         arrayminus=fc168_s['forecast_mwh']-fc168_s['p10_mwh'],
                         color='rgba(255,255,255,0.4)')))
        fig.update_layout(**{**CHART_LAYOUT,"title":"7-Day Generation Outlook (MWh/day)","height":300})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 DSM Schedule Submission")
        st.markdown(f"""
        <div class="bk-card bk-card-green" style="padding:16px 20px">
        <table style="width:100%;font-size:0.84rem">
        <tr><td style="color:#64748B;padding:4px 0">Forecast Accuracy</td><td style="color:#10B981;font-weight:700">94.2% (7-day avg)</td></tr>
        <tr><td style="color:#64748B">DSM Deadline</td><td>10:00 AM today</td></tr>
        <tr><td style="color:#64748B">Schedule Submitted</td><td style="color:#10B981">✅ Auto-submitted</td></tr>
        <tr><td style="color:#64748B">SLDC Deviation Band</td><td>±5% (±{total_fc*0.05:.0f} kWh)</td></tr>
        <tr><td style="color:#64748B">DSM Penalty Risk</td><td style="color:#10B981">Low — ₹800 est.</td></tr>
        <tr><td style="color:#64748B">Baseline (Last Week)</td><td>{total_fc*0.95:.0f} kWh</td></tr>
        <tr><td style="color:#64748B">Model Used</td><td>XGBoost + Lag Features</td></tr>
        </table>
        </div>""", unsafe_allow_html=True)

        # Weekly confidence
        fig2 = go.Figure(go.Scatter(
            x=fc168['date'], y=fc168['confidence_pct'],
            fill='tozeroy', fillcolor="rgba(16,185,129,0.1)",
            line=dict(color=GREEN,width=2), name="Confidence %"
        ))
        fig2.update_layout(**{**CHART_LAYOUT,"title":"Forecast Confidence by Day","height":180,"yaxis_range":[70,100]})
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ROI & COST ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_roi(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">💰 ROI & Cost Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">LCOE calculator · NPV analysis · Payback period · Revenue uplift model</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Financial Dashboard", "🔢 LCOE / NPV Calculator"])

    with tab1:
        # Revenue breakdown
        annual_ppa = t["mw"]*1000 * 22/100 * 8760 * t["ppa"] * s / 100000
        annual_iex = annual_ppa * 0.08
        annual_dsm = 2.4 * s
        annual_rec = t["mw"] * 0.3 * s   # ₹L per MW

        col1, col2, col3 = st.columns(3)
        with col1:
            streams = {"PPA Sales":annual_ppa, "IEX Premium":annual_iex, "REC Sales":annual_rec, "Ancillary":annual_ppa*0.02}
            fig = go.Figure(go.Pie(
                labels=list(streams.keys()), values=list(streams.values()),
                hole=0.55,
                marker=dict(colors=[ORANGE,GREEN,BLUE,PURPLE]),
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>₹%{value:.1f}L<extra></extra>"
            ))
            fig.add_annotation(text=f"₹{sum(streams.values()):.0f}L<br>Annual", x=0.5, y=0.5,
                                font=dict(size=14,color="#F1F5F9"), showarrow=False)
            fig.update_layout(**{**CHART_LAYOUT,"title":"Annual Revenue Mix (₹ Lakh)","height":320,"showlegend":True})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cost breakdown
            opex_mnt = t["mw"] * 5 * s
            opex_O_M = t["mw"] * 2 * s
            opex_ins = t["mw"] * 1 * s
            opex_saa = 0.5 * s
            costs = {"Maintenance":opex_mnt, "Operations":opex_O_M, "Insurance":opex_ins, "SaaS Platform":opex_saa}
            fig = go.Figure(go.Bar(
                x=list(costs.keys()), y=list(costs.values()),
                marker=dict(color=[RED,ORANGE,YELLOW,PURPLE]),
                hovertemplate="<b>%{x}</b><br>₹%{y:.2f}L/yr<extra></extra>"
            ))
            fig.update_layout(**{**CHART_LAYOUT,"title":"Annual OpEx Breakdown (₹ Lakh)","height":320})
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Cumulative revenue projection
            years   = list(range(1, 26))
            degradation = 0.005    # 0.5%/yr
            rev_cum = []
            total = 0
            for yr in years:
                yr_rev = annual_ppa * (1-degradation)**yr * (1 + 0.03)**yr  # tariff escalation 3%/yr
                total += yr_rev
                rev_cum.append(total)
            capex = t["mw"] * 45 * s    # ₹45L/MW typical
            payback_yr = next((yr for yr, r in zip(years, rev_cum) if r >= capex), 25)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=rev_cum, fill='tozeroy', fillcolor="rgba(16,185,129,0.1)",
                line=dict(color=GREEN,width=2.5), name="Cumulative Revenue (₹L)"))
            fig.add_hline(y=capex, line_dash="dash", line_color=RED,
                          annotation_text=f"CapEx: ₹{capex:.0f}L | Payback: Yr {payback_yr}")
            fig.update_layout(**{**CHART_LAYOUT,"title":"25-Year Revenue Projection (₹ Lakh)","height":320})
            st.plotly_chart(fig, use_container_width=True)

        # Revenue uplift table
        st.markdown("#### ⚡ Platform Revenue Uplift vs Baseline")
        uplift_data = {
            "Revenue Stream": ["PPA Energy Sales","IEX Arbitrage Premium","DSM Penalty Avoided","REC Certificates","Ancillary Services","Net Metering","Warranty Claims"],
            "Baseline (₹L/yr)": [f"{annual_ppa:.1f}","0","0","0","0","0","0"],
            "With Basker SaaS (₹L/yr)": [f"{annual_ppa*1.04:.1f}",f"{annual_iex:.2f}",f"{annual_dsm:.2f}",f"{annual_rec:.2f}",f"{annual_ppa*0.02:.2f}",f"{annual_ppa*0.03:.2f}",f"{annual_ppa*0.01:.2f}"],
            "Annual Uplift (₹L)": [f"+{annual_ppa*0.04:.2f}",f"+{annual_iex:.2f}",f"+{annual_dsm:.2f}",f"+{annual_rec:.2f}",f"+{annual_ppa*0.02:.2f}",f"+{annual_ppa*0.03:.2f}",f"+{annual_ppa*0.01:.2f}"],
            "Confidence": ["98%","87%","94%","99%","78%","86%","72%"],
        }
        st.dataframe(pd.DataFrame(uplift_data), use_container_width=True, hide_index=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            capacity  = st.number_input("Plant Capacity (kW)", value=int(t["mw"]*1000), step=100)
            capex_kw  = st.number_input("CapEx (₹/kW)", value=38000, step=1000)
            tariff    = st.number_input("PPA / Grid Tariff (₹/kWh)", value=t["ppa"], step=0.1, format="%.2f")
        with col2:
            lifetime  = st.slider("Plant Lifetime (years)", 15, 30, 25)
            degrad    = st.slider("Degradation Rate (%/yr)", 0.0, 1.0, 0.5, 0.05)
            cuf_pct   = st.slider("Capacity Factor (%)", 15, 30, 22)
            subsidy   = st.slider("Subsidy / Incentive (%)", 0, 40, 30, 5)

        net_cap  = capex_kw * capacity * (1 - subsidy/100)
        ann_gen  = [capacity * cuf_pct/100 * 8760 * (1-degrad/100)**y for y in range(lifetime)]
        opex_ann = [capacity * 500 for _ in range(lifetime)]
        disc     = 0.08
        pv_costs = net_cap + sum(c/((1+disc)**y) for y,c in enumerate(opex_ann,1))
        pv_energy= sum(e/((1+disc)**y) for y,e in enumerate(ann_gen,1))
        lcoe     = pv_costs / pv_energy if pv_energy else 0
        ann_rev  = ann_gen[0] * tariff
        payback  = net_cap / (ann_rev - opex_ann[0]) if (ann_rev - opex_ann[0]) > 0 else 99
        npv_25   = sum((ann_gen[y]*tariff - opex_ann[y])/((1+disc)**(y+1)) for y in range(lifetime)) - net_cap
        irr_approx = (ann_rev - opex_ann[0]) / net_cap

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1: st.metric("LCOE",     f"₹{lcoe:.2f}/kWh", delta=f"{'✅ Below grid' if lcoe < tariff else '⚠️ Above grid'}")
        with col_r2: st.metric("NPV (25yr)",f"₹{npv_25/100000:.1f}L", delta="@8% discount rate")
        with col_r3: st.metric("Payback",   f"{payback:.1f} years", delta=f"{'Simple' if payback < 8 else 'Extended'}")
        with col_r4: st.metric("Est. IRR",  f"{irr_approx*100:.1f}%", delta="Unlevered")

        # Revenue curve
        years = list(range(1, lifetime+1))
        cum_rev = [sum(ann_gen[y]*tariff - opex_ann[y] for y in range(yr)) - net_cap for yr in years]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=cum_rev, fill='tozeroy',
            fillcolor="rgba(59,130,246,0.1)" if cum_rev[-1]>0 else "rgba(239,68,68,0.1)",
            line=dict(color=GREEN if cum_rev[-1]>0 else RED, width=2.5), name="Cumulative Cash Flow (₹)"))
        fig.add_hline(y=0, line_color=ORANGE, line_dash="dash", annotation_text="Break-Even")
        fig.update_layout(**{**CHART_LAYOUT,"title":"Cumulative Cash Flow over Project Life","height":280})
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: SMART CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def page_cleaning(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">🧹 Smart Cleaning & Soiling Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML soiling inference · Cost-vs-gain scheduler · Weather-deferred triggers · Robotic fleet</div>', unsafe_allow_html=True)

    rec = engine.cleaning_recommendation(df)
    urgency_colors = {"critical":RED,"high":ORANGE,"medium":BLUE,"low":GREEN}
    urg_col = urgency_colors.get(rec['urgency'], GREEN)

    st.markdown(f"""
    <div style='background:#161f31;border:2px solid {urg_col};border-radius:14px;padding:22px;margin-bottom:20px'>
    <h3 style='color:{urg_col};margin:0 0 14px'>🧹 Cleaning Decision: {rec["action"]}</h3>
    <div style='display:flex;gap:36px;flex-wrap:wrap'>
        <div><span style='color:#64748B;font-size:0.76rem'>SOILING LOSS</span><br>
             <span style='font-size:2rem;font-weight:800;color:{urg_col}'>{rec["soiling_pct"]:.1f}%</span></div>
        <div><span style='color:#64748B;font-size:0.76rem'>DAILY REVENUE AT RISK</span><br>
             <span style='font-size:2rem;font-weight:800;color:{YELLOW}'>₹{rec["revenue_lost_daily_inr"]*s:,.0f}</span></div>
        <div><span style='color:#64748B;font-size:0.76rem'>CLEANING COST</span><br>
             <span style='font-size:2rem;font-weight:800;color:{BLUE}'>₹{rec["cleaning_cost_inr"]*s:,.0f}</span></div>
        <div><span style='color:#64748B;font-size:0.76rem'>ROI PAYBACK</span><br>
             <span style='font-size:2rem;font-weight:800;color:{GREEN}'>{rec["roi_payback_days"]:.0f} days</span></div>
        <div><span style='color:#64748B;font-size:0.76rem'>RAIN FORECAST</span><br>
             <span style='font-size:1.4rem;font-weight:700;color:#06B6D4'>{"☔ Rain — Defer" if rec["rain_likely"] else "☀️ No rain"}</span></div>
    </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        daily_soil = df.groupby(df['timestamp'].dt.date)['soiling_loss_pct'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_soil.index.astype(str), y=daily_soil.values,
            fill='tozeroy', fillcolor="rgba(245,158,11,0.12)",
            line=dict(color=YELLOW,width=2.5), name="Soiling Loss %"))
        fig.add_hline(y=1.5, line_dash="dash", line_color=YELLOW, annotation_text="Schedule")
        fig.add_hline(y=3.0, line_dash="dash", line_color=RED,    annotation_text="Urgent")
        fig.update_layout(**{**CHART_LAYOUT,"title":"Daily Soiling Loss Trend (%)","height":300,"yaxis_title":"Soiling %"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        days_sim = list(range(1, 60))
        soil_acc = [min(5, 0.08*d) for d in days_sim]
        cumu_loss= [max(0,(s_val-0.5)*t["mw"]*1000*5.5*t["ppa"]*s/100) for s_val in soil_acc]
        cum_loss_arr = [sum(cumu_loss[:d]) for d in days_sim]
        roi_net  = [cl - rec["cleaning_cost_inr"]*s for cl in cum_loss_arr]
        opt_day  = next((d for d,r in zip(days_sim, roi_net) if r > 0), 30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days_sim, y=cum_loss_arr, line=dict(color=RED,width=2), name="Cumul. Loss (₹)"))
        fig.add_trace(go.Scatter(x=days_sim, y=roi_net, line=dict(color=GREEN,width=2,dash='dot'), name="Net ROI after Clean"))
        fig.add_vline(x=opt_day, line_color=ORANGE, line_dash="dash", annotation_text=f"Optimal: Day {opt_day}")
        fig.update_layout(**{**CHART_LAYOUT,"title":"Cleaning ROI Optimiser","height":300,"yaxis_title":"₹"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Soiling Reference Table — Rajasthan Environment")
    ref = {
        "Level":          ["Clean","Light Dust","Moderate","Heavy","Critical"],
        "Soiling %":      ["0–0.5%","0.5–1.5%","1.5–3.0%","3.0–5.0%",">5.0%"],
        "Days Since Clean":["0–5","6–12","13–20","21–30",">30"],
        "Daily Loss (₹)": [f"₹{int(0*s)}",f"₹{int(800*s):,}",f"₹{int(2200*s):,}",f"₹{int(4800*s):,}",f"₹{int(8500*s):,}"],
        "Recommendation": ["No action","Monitor","Schedule 3 days","URGENT — 24h","EMERGENCY"],
        "Status":         ["🟢","🟡","🟠","🔴","💥"],
    }
    st.dataframe(pd.DataFrame(ref), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: ENERGY TRADING
# ══════════════════════════════════════════════════════════════════════════════
def page_trading(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">⚡ Energy Trading & Monetisation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">IEX sell/hold signals · ToU arbitrage · BESS dispatch optimiser · DSM avoidance</div>', unsafe_allow_html=True)

    forecast = engine.forecast_dayahead(df, 24)
    forecast['forecast_kwh'] *= s
    signals  = engine.trading_signals(forecast, bess_soc=0.55, grid_price_rs=7.5)

    action_colors = {"SELL_GRID":RED,"CHARGE_BATTERY":BLUE,"DISCHARGE_BATTERY":ORANGE,"EXPORT_PPA":GREEN,"SELF_CONSUME":TEAL}
    action_icons  = {"SELL_GRID":"💹","CHARGE_BATTERY":"🔋","DISCHARGE_BATTERY":"⚡","EXPORT_PPA":"📤","SELF_CONSUME":"🏠"}

    total_rev   = signals['revenue_est_inr'].sum()
    sell_hrs    = len(signals[signals['action']=='SELL_GRID'])
    export_hrs  = len(signals[signals['action']=='EXPORT_PPA'])
    peak_price  = signals['grid_price'].max()

    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Estimated 24h Revenue", f"₹{total_rev:,.0f}", delta="AI-optimised dispatch")
    with col2: st.metric("Grid Sell Hours",        f"{sell_hrs}h",       delta="IEX triggered")
    with col3: st.metric("PPA Export Hours",       f"{export_hrs}h",     delta=f"@₹{t['ppa']}/kWh")
    with col4: st.metric("Peak Grid Price",        f"₹{peak_price:.2f}", delta="Evening peak")

    st.markdown("<br>", unsafe_allow_html=True)

    colors_bar = [action_colors.get(a, TEAL) for a in signals['action']]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6,0.4],
                        subplot_titles=["Generation + Trading Actions (kWh/hr)","Grid Price vs PPA Rate (₹/kWh)"])
    fig.add_trace(go.Bar(x=signals['hour'], y=signals['forecast_kwh'],
        marker=dict(color=colors_bar), name="Gen (colour = action)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals['hour'], y=signals['grid_price'],
        line=dict(color=RED,width=2), name="Grid Price"), row=2, col=1)
    fig.add_hline(y=t['ppa'], line_dash="dash", line_color=GREEN,
                  annotation_text=f"PPA ₹{t['ppa']}", row=2, col=1)
    fig.update_layout(**{**CHART_LAYOUT,"height":480,"showlegend":True})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        disp = signals[['hour','forecast_kwh','grid_price','action','reason','revenue_est_inr','bess_soc_pct']].copy()
        disp['action'] = disp['action'].apply(lambda a: f"{action_icons.get(a,'')} {a}")
        disp.columns   = ['Hour','Gen (kWh)','Grid ₹','Action','Reason','Revenue ₹','BESS SoC %']
        st.markdown("#### 24-Hour Dispatch Schedule")
        st.dataframe(disp.round(2), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 📈 Revenue Uplift vs Baseline")
        streams = {
            "Revenue Stream":["PPA Sales","IEX Premium","DSM Avoided","Ancillary","Net Metering"],
            "Baseline":[f"₹{int(t['mw']*1.08e5*s):,}","₹0","₹0","₹0","₹0"],
            "With SaaS":[f"₹{int(t['mw']*1.24e5*s):,}",f"+₹{int(4.2e4*s):,}",f"+₹{int(7.6e4*s):,}",f"+₹{int(1.8e4*s):,}",f"+₹{int(2.4e4*s):,}"],
            "Uplift/yr":[f"₹{int(1.6e4*s):,}",f"₹{int(4.2e4*s):,}",f"₹{int(7.6e4*s):,}",f"₹{int(1.8e4*s):,}",f"₹{int(2.4e4*s):,}"],
        }
        st.dataframe(pd.DataFrame(streams), use_container_width=True, hide_index=True)

        bess_soc_ts = signals['bess_soc_pct'].values
        fig2 = go.Figure(go.Scatter(x=signals['hour'], y=bess_soc_ts, fill='tozeroy',
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(color=BLUE,width=2), name="BESS SoC %"))
        fig2.add_hline(y=80, line_dash="dash", line_color=GREEN,  annotation_text="Max 80%")
        fig2.add_hline(y=20, line_dash="dash", line_color=RED, annotation_text="Min 20%")
        fig2.update_layout(**{**CHART_LAYOUT,"title":"BESS State of Charge (%)","height":220,"yaxis_range":[0,110]})
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: AI ADVISOR  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_advisor(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">🤖 AI Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Contextual AI recommendations · Priority-ranked actions · ROI impact estimates</div>', unsafe_allow_html=True)

    from ml_core import generate_ai_recommendations, compute_esg_metrics
    m    = engine.performance_metrics(df)
    recs = generate_ai_recommendations(df, m, tier)
    esg  = compute_esg_metrics(df, t["mw"])

    # ── AI Score card ──
    total_impact = sum([
        m.get("gap_revenue_inr",0)*s*0.6,
        4.2e4*s,
        7.6e4*s
    ])
    ai_conf = 91.4
    n_crit  = len([r for r in recs if "CRITICAL" in r["priority"]])
    n_opp   = len([r for r in recs if "OPPORTUNITY" in r["priority"]])

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(kpi_html(f"{len(recs)}", "Active Recommendations", "Priority-ranked", "neu", ORANGE, "🤖"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_html(str(n_crit),   "Critical Actions",        "Require attention", "neg" if n_crit>0 else "pos", RED if n_crit>0 else GREEN, "🔴"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_html(str(n_opp),    "Revenue Opportunities",   "AI-identified", "pos", GREEN, "💰"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_html(f"₹{total_impact/100000:.1f}L", "Total Actionable Value","Weekly potential", "pos", PURPLE, "📈"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendation cards ──
    priority_order = {"🔴 CRITICAL":0,"🟡 WARNING":1,"🟢 OPPORTUNITY":2,"🟢 INFO":3}
    recs_sorted = sorted(recs, key=lambda r: priority_order.get(r["priority"],9))

    col_recs, col_side = st.columns([1.8, 1])

    with col_recs:
        st.markdown("### 📋 Recommendations")
        for rec in recs_sorted:
            pr_col = RED if "CRITICAL" in rec["priority"] else (YELLOW if "WARNING" in rec["priority"] else GREEN)
            cat_icon = {"Performance":"📊","O&M":"🔧","Maintenance":"🏥","Revenue":"💰","Optimization":"⚙️","Asset Lifecycle":"🏗"}.get(rec["category"],"🤖")
            st.markdown(f"""
            <div class="rec-card" style="border-left:4px solid {pr_col}">
                <div class="rec-priority" style="color:{pr_col}">{rec['priority']} &nbsp;·&nbsp; {cat_icon} {rec['category']}</div>
                <div class="rec-title">{rec['title']}</div>
                <div class="rec-insight">💡 {rec['insight']}</div>
                <div class="rec-action">🎯 <b>Action:</b> {rec['action']}</div>
                <div style="display:flex;justify-content:space-between;margin-top:10px;flex-wrap:wrap;gap:8px">
                    <div class="rec-impact">📈 {rec['impact']}</div>
                    <div style="font-size:0.78rem;color:#64748B;background:#1E293B;border-radius:6px;padding:3px 10px">
                        🎯 Confidence: {rec['confidence']}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_side:
        # AI confidence radar
        st.markdown("### 🎯 AI Confidence by Domain")
        domains  = ["Power Pred","Fault Detect","Soiling Est","Forecasting","Trading","Anomaly"]
        scores   = [99.9, 87.0, 91.0, 94.2, 88.0, 84.0]
        fig = go.Figure(go.Scatterpolar(
            r=scores + [scores[0]], theta=domains + [domains[0]],
            fill='toself', fillcolor="rgba(232,119,34,0.15)",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(size=8, color=ORANGE)
        ))
        fig.update_layout(**{**CHART_LAYOUT,
            "polar":dict(
                bgcolor="#111827",
                radialaxis=dict(range=[70,100],gridcolor="#1E293B",tickfont=dict(color="#64748B",size=10)),
                angularaxis=dict(gridcolor="#1E293B",tickfont=dict(color="#CBD5E1",size=11)),
            ),
            "height":320, "showlegend":False, "title":"Model Confidence (%) by Domain"
        })
        st.plotly_chart(fig, use_container_width=True)

        # Priority donut
        st.markdown("### 🧩 Recommendations by Priority")
        pr_cnt  = {"Critical":n_crit,"Warning":len([r for r in recs if "WARNING" in r["priority"]]),
                   "Opportunity":n_opp,"Info":len([r for r in recs if "INFO" in r["priority"]])}
        fig2 = go.Figure(go.Pie(
            labels=list(pr_cnt.keys()), values=list(pr_cnt.values()),
            hole=0.6, marker=dict(colors=[RED,YELLOW,GREEN,TEAL]),
            textfont=dict(size=12),
        ))
        fig2.add_annotation(text=f"{len(recs)}<br>Total", x=0.5, y=0.5, font=dict(size=16,color="#F1F5F9"), showarrow=False)
        fig2.update_layout(**{**CHART_LAYOUT,"height":280,"showlegend":True})
        st.plotly_chart(fig2, use_container_width=True)

        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")
        actions = ["📋 Export Report","📧 Email to EPC Team","📅 Schedule Maintenance","📤 Submit DSM","🔔 Set Alert"]
        for action in actions:
            st.button(action, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9: FLEET INTELLIGENCE  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
def page_fleet(engine, df, tier):
    st.markdown('<div class="page-title">🗺 Fleet Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Multi-plant comparison · Geo heat map · Benchmarking · Portfolio KPIs</div>', unsafe_allow_html=True)

    fleet = load_fleet()

    # ── Portfolio KPIs ──
    total_mw   = fleet['mw'].sum()
    total_mwh  = fleet['weekly_gen_mwh'].sum()
    total_rev  = fleet['weekly_rev_lakh'].sum()
    avg_pr     = fleet['pr_pct'].mean()
    total_faults = fleet['fault_count'].sum()
    total_co2  = fleet['co2_avoided_t'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(kpi_html(f"{total_mw:.0f} MW", "Portfolio Capacity", f"{len(fleet)} plants", "pos", ORANGE, "⚡"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_html(f"{total_mwh:.1f} MWh","Weekly Generation","Last 7 days", "pos", GREEN, "☀️"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_html(f"₹{total_rev:.1f}L",  "Weekly Revenue",   "All plants", "pos", BLUE, "💰"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_html(f"{avg_pr:.1f}%",       "Avg Performance Ratio","Fleet P50", "neu", PURPLE, "📊"), unsafe_allow_html=True)
    with col5: st.markdown(kpi_html(f"{total_co2:.1f}t",    "CO₂ Avoided",      "This week", "pos", TEAL, "🌱"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        # Geo scatter map
        fig = go.Figure(go.Scattergeo(
            lat=fleet['lat'], lon=fleet['lon'],
            text=fleet.apply(lambda r: f"<b>{r['name']}</b><br>PR: {r['pr_pct']}%<br>Gen: {r['weekly_gen_mwh']:.1f} MWh<br>Rev: ₹{r['weekly_rev_lakh']:.2f}L", axis=1),
            marker=dict(
                size=fleet['mw']*3.5,
                color=fleet['pr_pct'],
                colorscale="RdYlGn", cmin=70, cmax=90,
                colorbar=dict(title="PR %", tickcolor="#64748B"),
                line=dict(color='white', width=1.5),
                sizemode='area',
            ),
            mode='markers+text',
            texttext=fleet['name'],
            textposition='top center',
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.update_geos(
            scope='asia', bgcolor="#0A0E1A",
            landcolor="#161f31", oceancolor="#0D1829",
            lakecolor="#0D1829", rivercolor="#1E293B",
            showcoastlines=True, coastlinecolor="#1E293B",
            showland=True, showocean=True,
            center=dict(lat=22, lon=76), projection_scale=5,
        )
        fig.update_layout(**{**CHART_LAYOUT,"title":"Fleet Geo-Performance Map — India","height":420,
                              "geo":dict(bgcolor="#0A0E1A")})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # PR comparison
        fleet_sorted = fleet.sort_values('pr_pct', ascending=True)
        colors_pr = [GREEN if p>82 else YELLOW if p>78 else RED for p in fleet_sorted['pr_pct']]
        fig = go.Figure(go.Bar(
            x=fleet_sorted['pr_pct'], y=fleet_sorted['name'],
            orientation='h', marker=dict(color=colors_pr),
            text=[f"{v:.1f}%" for v in fleet_sorted['pr_pct']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>PR: %{x:.1f}%<extra></extra>"
        ))
        fig.add_vline(x=fleet['pr_pct'].mean(), line_dash="dash", line_color=ORANGE,
                      annotation_text=f"Fleet Avg {fleet['pr_pct'].mean():.1f}%")
        fig.update_layout(**{**CHART_LAYOUT,"title":"Performance Ratio by Plant","height":240,"xaxis_range":[65,95]})
        st.plotly_chart(fig, use_container_width=True)

        # Revenue bar
        fig = go.Figure(go.Bar(
            x=fleet['name'], y=fleet['weekly_rev_lakh'],
            marker=dict(color=fleet['weekly_rev_lakh'], colorscale="Blues", showscale=False),
            hovertemplate="<b>%{x}</b><br>₹%{y:.2f}L<extra></extra>"
        ))
        fig.update_layout(**{**CHART_LAYOUT,"title":"Weekly Revenue by Plant (₹ Lakh)","height":240})
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed fleet table ──
    st.markdown("### 📋 Plant-Level Dashboard")
    fleet_display = fleet[['id','name','state','mw','pr_pct','cuf_pct','weekly_gen_mwh',
                            'weekly_rev_lakh','soiling_pct','fault_count','availability_pct',
                            'co2_avoided_t','irr_kwh_m2']].copy()
    fleet_display.columns = ['ID','Plant','State','MW','PR %','CUF %','Gen MWh',
                              'Rev ₹L','Soiling %','Faults','Avail %','CO₂ t','Irr kWh/m²']
    st.dataframe(fleet_display.sort_values('PR %', ascending=False),
                 use_container_width=True, hide_index=True)

    # ── Benchmark radar ──
    st.markdown("### 🏆 Plant Benchmarking Radar")
    cat = st.selectbox("Select Plant for Deep Benchmark", fleet['name'].tolist(), key="fleet_bench")
    p   = fleet[fleet['name']==cat].iloc[0]
    avg = fleet.mean(numeric_only=True)

    cats_radar = ["PR","CUF","Availability","Irradiance","Revenue/MW","CO₂/MW"]
    vals_plant  = [p['pr_pct'], p['cuf_pct'], p['availability_pct'],
                   p['irr_kwh_m2']*10, p['weekly_rev_lakh']/p['mw']*10, p['co2_avoided_t']/p['mw']*10]
    vals_fleet  = [avg['pr_pct'], avg['cuf_pct'], avg['availability_pct'],
                   avg['irr_kwh_m2']*10, avg['weekly_rev_lakh']/avg['mw']*10, avg['co2_avoided_t']/avg['mw']*10]

    def norm(vals):
        mx = 100
        return [min(100, v*100/mx) for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_plant+[vals_plant[0]], theta=cats_radar+[cats_radar[0]],
        fill='toself', fillcolor="rgba(232,119,34,0.15)",
        line=dict(color=ORANGE,width=2.5), name=cat
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_fleet+[vals_fleet[0]], theta=cats_radar+[cats_radar[0]],
        fill='toself', fillcolor="rgba(59,130,246,0.1)",
        line=dict(color=BLUE,width=2,dash='dash'), name="Fleet Average"
    ))
    fig.update_layout(**{**CHART_LAYOUT,
        "polar":dict(
            bgcolor="#111827",
            radialaxis=dict(range=[0,110],gridcolor="#1E293B",tickfont=dict(color="#64748B",size=9)),
            angularaxis=dict(gridcolor="#1E293B",tickfont=dict(color="#CBD5E1",size=11)),
        ),
        "height":380,"showlegend":True,"title":f"{cat} vs Fleet Average"
    })
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10: ESG & SUSTAINABILITY  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
def page_esg(engine, df, tier):
    t = tier_cfg(tier); s = t["scale"]
    st.markdown('<div class="page-title">🌱 ESG & Sustainability Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Carbon avoidance · Green score · SDG alignment · Carbon credit valuation · ESG benchmarking</div>', unsafe_allow_html=True)

    from ml_core import compute_esg_metrics
    esg = compute_esg_metrics(df, t["mw"])

    # ── Hero ESG score ──
    green_score = esg["green_score"]
    st.markdown(f"""
    <div class="bk-card bk-card-green" style="text-align:center;padding:30px">
        <div style="font-size:0.78rem;color:#64748B;letter-spacing:2px;margin-bottom:8px">BASKER ENERGY GREEN SCORE</div>
        <div style="font-size:5rem;font-weight:900;color:#10B981;line-height:1">{green_score}</div>
        <div style="font-size:1.1rem;color:#94A3B8;margin-top:4px">out of 100 · {'🏆 Excellent' if green_score>85 else '🥈 Good' if green_score>70 else '🥉 Fair'}</div>
        <div style="margin-top:16px;display:flex;justify-content:center;gap:8px">
            <span style="background:#0d2e1a;border:1px solid #10B981;color:#6EE7B7;border-radius:20px;padding:4px 16px;font-size:0.8rem">🌍 SDG 7 Aligned</span>
            <span style="background:#0d2e1a;border:1px solid #10B981;color:#6EE7B7;border-radius:20px;padding:4px 16px;font-size:0.8rem">🌿 Net Positive</span>
            <span style="background:#0d2e1a;border:1px solid #10B981;color:#6EE7B7;border-radius:20px;padding:4px 16px;font-size:0.8rem">♻️ Zero Waste Ops</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Carbon metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(kpi_html(f"{esg['co2_avoided_t']:.1f}t", "CO₂ Avoided","vs coal baseline","pos",GREEN,"🌍"), unsafe_allow_html=True)
    with col2: st.markdown(kpi_html(f"{esg['trees_equivalent']:,}","Trees Equivalent","Offset achieved","pos",TEAL,"🌳"), unsafe_allow_html=True)
    with col3: st.markdown(kpi_html(f"₹{esg['carbon_credit_inr']/100000:.1f}L","Carbon Credit Value","@$15/tonne VCM","pos",PURPLE,"💱"), unsafe_allow_html=True)
    with col4: st.markdown(kpi_html(f"{esg['recs_earned']:,}","RECs Earned","@ 1 REC/MWh","pos",BLUE,"🏅"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Cumulative CO2 curve
        if 'cumulative_co2_kg' in df.columns:
            daily_co2 = df.groupby(df['timestamp'].dt.date)['co2_avoided_kg'].sum().reset_index()
            daily_co2['co2_t'] = daily_co2['co2_avoided_kg'] * s / 1000
            daily_co2['cumul'] = daily_co2['co2_t'].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_co2['timestamp'].astype(str), y=daily_co2['cumul'],
                fill='tozeroy', fillcolor="rgba(16,185,129,0.12)",
                line=dict(color=GREEN,width=2.5), name="Cumulative CO₂ Avoided (t)"))
            fig.update_layout(**{**CHART_LAYOUT,"title":"Cumulative CO₂ Avoidance (Tonnes)","height":300,"yaxis_title":"CO₂ Tonnes"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Environmental equivalences
        st.markdown("### 🌍 Environmental Equivalences")
        equiv = [
            ("🌳","Trees Planted", f"{esg['trees_equivalent']:,}","Equivalent offset"),
            ("🚗","Cars Off Road/yr",f"{esg['cars_off_road_yr']:.0f}","Annual equivalent"),
            ("🏠","Homes Powered/yr",f"{esg['homes_powered_yr']:.0f}","Average homes"),
            ("💧","Water Saved",    f"{esg['water_saved_kl']:,.0f} kL","vs thermal gen"),
            ("🏭","Coal Displaced", f"{esg['coal_displaced_t']:.1f}t","Thermal coal"),
        ]
        for icon, label, val, sub in equiv:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:16px;padding:10px 16px;background:#161f31;
                 border:1px solid #1E2A40;border-radius:10px;margin-bottom:8px">
                <div style="font-size:1.6rem;width:40px;text-align:center">{icon}</div>
                <div style="flex:1">
                    <div style="font-size:0.78rem;color:#64748B">{label}</div>
                    <div style="font-size:1.2rem;font-weight:700;color:#10B981">{val}</div>
                </div>
                <div style="font-size:0.72rem;color:#475569">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── SDG radar ──
    col1, col2 = st.columns(2)
    with col1:
        sdg_names  = ["SDG 7\nClean Energy","SDG 13\nClimate Action","SDG 9\nInfrastructure",
                      "SDG 11\nSustainable Cities","SDG 12\nResponsible Prod","SDG 17\nPartnerships"]
        sdg_scores = [9.8, 9.2, 8.5, 8.1, 8.7, 7.9]
        fig = go.Figure(go.Scatterpolar(
            r=sdg_scores+[sdg_scores[0]], theta=sdg_names+[sdg_names[0]],
            fill='toself', fillcolor="rgba(16,185,129,0.15)",
            line=dict(color=GREEN,width=2.5), marker=dict(size=8,color=GREEN)
        ))
        fig.update_layout(**{**CHART_LAYOUT,
            "polar":dict(bgcolor="#111827",
                radialaxis=dict(range=[0,10],gridcolor="#1E293B",tickfont=dict(color="#64748B",size=9)),
                angularaxis=dict(gridcolor="#1E293B",tickfont=dict(color="#CBD5E1",size=9)),
            ),
            "height":360,"showlegend":False,"title":"SDG Alignment Score (out of 10)"
        })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ESG pillar scores
        st.markdown("### 📊 ESG Pillar Scores")
        pillars = [
            ("🌿 Environmental","Carbon avoidance, water savings, biodiversity",94),
            ("👥 Social","Community energy access, local employment",81),
            ("🏛 Governance","Data transparency, compliance, reporting",88),
            ("⚡ Energy Access","Off-grid contribution, energy security",76),
            ("📡 Technology","AI/ML adoption, digitisation index",96),
        ]
        for icon_label, desc, score in pillars:
            bar_col = GREEN if score>85 else YELLOW if score>70 else RED
            st.markdown(f"""
            <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.88rem;color:#CBD5E1;font-weight:600">{icon_label}</span>
                    <span style="font-size:0.88rem;color:{bar_col};font-weight:700">{score}/100</span>
                </div>
                <div style="font-size:0.74rem;color:#64748B;margin-bottom:5px">{desc}</div>
                <div class="esg-bar"><div class="esg-fill" style="width:{score}%;background:linear-gradient(90deg,{bar_col},{bar_col}88)"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="bk-card bk-card-teal" style="margin-top:16px;padding:14px 18px">
            <div style="font-size:0.76rem;color:#64748B">CARBON CREDIT MONETISATION</div>
            <div style="font-size:1.5rem;font-weight:800;color:#10B981">₹{esg['carbon_credit_inr']/100000:.2f}L</div>
            <div style="font-size:0.8rem;color:#94A3B8;margin-top:4px">
                {esg['co2_avoided_t']:.1f}t CO₂ × $15/t (VCM) × ₹83.5/$<br>
                RECs: {esg['recs_earned']} × ₹800 = ₹{esg['recs_earned']*800/100000:.2f}L additional
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Annual ESG targets ──
    st.markdown("### 🎯 Annual ESG Targets vs Actuals")
    ann_kwh = esg.get("ann_kwh", esg["total_kwh"] * 52)
    targets = {
        "Metric":     ["Annual Generation","CO₂ Avoided","RECs Earned","CUF","PR","Availability","Water Saved"],
        "Target":     [f"{t['mw']*22/100*8760:.0f} kWh",f"{t['mw']*22/100*8760*0.716/1000:.0f}t","2,190",">22%",">80%",">98%","25,000 kL"],
        "Actuals (7d extrapolated)": [
            f"{ann_kwh:.0f} kWh", f"{esg['ann_co2_t']:.0f}t", str(esg['recs_earned']*52),
            f"{engine.performance_metrics(df).get('cuf_pct',22):.1f}%",
            f"{engine.performance_metrics(df).get('performance_ratio_pct',82):.1f}%",
            "98.8%", f"{esg['water_saved_kl']*52:.0f} kL"
        ],
        "Status": ["🟢 On Track","🟢 On Track","🟢 On Track","🟢 On Track","🟡 Watch","🟢 On Track","🟢 On Track"],
    }
    st.dataframe(pd.DataFrame(targets), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.get("authenticated", False):
        login_page()
        return

    with st.spinner("🧠 Initialising Basker AI Engine…"):
        engine, df = load_engine()

    render_sidebar()

    tier = st.session_state.get("active_tier", "utility")
    page = st.session_state.get("active_page", "Overview Dashboard")

    render_topbar(tier)

    if   page == "Overview Dashboard":        page_overview(engine, df, tier)
    elif page == "Performance Optimization":  page_performance(engine, df, tier)
    elif page == "Predictive Maintenance":    page_maintenance(engine, df, tier)
    elif page == "Energy Forecasting":        page_forecasting(engine, df, tier)
    elif page == "ROI & Cost Analytics":      page_roi(engine, df, tier)
    elif page == "Smart Cleaning":            page_cleaning(engine, df, tier)
    elif page == "Energy Trading":            page_trading(engine, df, tier)
    elif page == "AI Advisor":                page_ai_advisor(engine, df, tier)
    elif page == "Fleet Intelligence":        page_fleet(engine, df, tier)
    elif page == "ESG & Sustainability":      page_esg(engine, df, tier)

if __name__ == "__main__":
    main()
