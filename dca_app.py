import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime

import time
from typing import Tuple

# --- Page Config ---
st.set_page_config(page_title="DCA Simulator", layout="wide")

# --- Constants ---

GITHUB_PROFILE_URL = "https://github.com/enesdesahin"
LINKEDIN_PROFILE_URL = "https://linkedin.com/in/sahinenes42/"

_SOCIAL_SVGS = {
    "github": (
        "GitHub",
        """<svg viewBox="0 0 24 24" role="img" aria-label="GitHub" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.089-.745.083-.73.083-.73 1.205.085 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762-1.605-2.665-.304-5.466-1.332-5.466-5.932 0-1.31.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23a11.5 11.5 0 0 1 3.003-.404 c1.018.005 2.045.138 3.003.404 2.291-1.552 3.297-1.23 3.297-1.23.655 1.653.244 2.874.12 3.176.77.84 1.235 1.911 1.235 3.221 0 4.61-2.807 5.625-5.479 5.921.43.372.823 1.102.823 2.222 0 1.604-.015 2.896-.015 3.286 0 .319.216.694.825.576 C20.565 22.092 24 17.592 24 12.297 24 5.67 18.627.297 12 .297z"/></svg>""",
    ),
    "linkedin": (
        "LinkedIn",
        """<svg viewBox="0 0 448 512" role="img" aria-label="LinkedIn" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M416 32H32A32 32 0 0 0 0 64v384a32 32 0 0 0 32 32h384a32 32 0 0 0 32-32V64a32 32 0 0 0-32-32zM135.4 416H69.1V202.2h66.3zm-33.1-243a38.4 38.4 0 1 1 38.4-38.4 38.4 38.4 0 0 1-38.4 38.4zM384 416h-66.2V312c0-24.8-.5-56.7-34.5-56.7-34.5 0-39.8 27-39.8 54.9V416h-66.2V202.2h63.6v29.2h.9c8.9-16.8 30.6-34.5 63-34.5 67.3 0 79.7 44.4 79.7 102.1z"/></svg>""",
    ),
}

CURRENCY_MAP = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "CHF": "CHF "
}

FX_TICKERS = {
    "EUR": "EUR=X", # EUR per USD (e.g. 0.92 EUR = 1 USD)
    "GBP": "GBP=X",
    "CHF": "CHF=X"
}

def get_social_links_html(
    *,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    clean_layout: bool = False,
    show_attribution: bool = True,
) -> str | None:
    """Generate the HTML for social profile badges."""
    github_target = github_url or GITHUB_PROFILE_URL
    linkedin_target = linkedin_url or LINKEDIN_PROFILE_URL

    entries = []
    if github_target:
        entries.append(("github", github_target))
    if linkedin_target:
        entries.append(("linkedin", linkedin_target))

    if not entries:
        return None

    badges = []
    for key, url in entries:
        label, svg_markup = _SOCIAL_SVGS.get(key, ("", ""))
        if not svg_markup:
            continue
        extra_style = "margin-left:-8px;" if key == "linkedin" else ""
        badge = (
            "<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
            "style=\"display:inline-flex;width:38px;height:38px;border-radius:999px;"
            "align-items:center;justify-content:center;color:#f8fafc;text-decoration:none;{extra_style}\" "
            "title=\"{label}\">"
            "<span style=\"display:inline-flex;width:22px;height:22px;color:#f8fafc;\">{svg}</span>"
            "</a>"
        ).format(url=url, label=label, svg=svg_markup, extra_style=extra_style)
        badges.append(badge)

    if not badges:
        return None

    # Construct the inner content
    attribution_html = ""
    if show_attribution:
        attribution_html = "<div style=\"margin-top:0.45rem;font-size:0.8rem;color:rgba(248,250,252,0.65);\">Developed by Enes SAHIN</div>"
    
    inner_html = (
        "<div style=\"display:flex;flex-direction:column;align-items:flex-start;\">"
        "<div style=\"font-size:0.85rem;font-weight:400;color:#e2e8f0;margin-bottom:0.4rem;\">Connect with me</div>"
        f"<div style=\"display:flex;gap:0.3rem;margin-left:-6px;\">{''.join(badges)}</div>"
        f"{attribution_html}"
        "</div>"
    )
    return inner_html

def render_social_links(
    *,
    github_url: str | None = None,
    linkedin_url: str | None = None,
    clean_layout: bool = False,
) -> None:
    """Render social profile badges at the bottom of the sidebar."""
    content_html = get_social_links_html(github_url=github_url, linkedin_url=linkedin_url, clean_layout=clean_layout)
    # Valid content check
    if not content_html:
        return

    # Add divider only for standard layout (Builder/Analytics)
    if not clean_layout:
        st.sidebar.divider()

    # Render as a styled card for ALL pages (matching Home page style)
    # Background: rgba(255, 255, 255, 0.04), Border: 1px solid rgba(255, 255, 255, 0.1), Radius: 8px
    card_style = (
        "background-color: rgba(255, 255, 255, 0.04);"
        "border: 1px solid rgba(255, 255, 255, 0.1);"
        "border-radius: 8px;"
        "padding: 16px;"
        "margin-bottom: 20px;"
    )
    
    card_html = f'<div style="{card_style}">{content_html}</div>'
    st.sidebar.markdown(card_html, unsafe_allow_html=True)

def render_metrics_html(
    total_inv: float,
    port_val: float,
    port_gain: float,
    port_ret: float,
    monthly_invest: float,
    currency_label: str
) -> str:
    """Generates HTML for the 4 metrics in styled cards."""
    
    ret_color = "#00e396" if port_ret >= 0 else "#ff4560"
    
    # Style adapted exactly from the user's snippet
    card_style = (
        "border: 1px solid #7c7c7c;"
        "padding: 15px;"
        "border-radius: 0px;"
        "display: flex;"
        "flex-direction: column;"
        "align-items: flex-start;"
        "justify-content: flex-start;"
        "background-color: transparent;"
    )
    
    
    # Styles from: <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
    label_style = "font-size: 16px; font-weight: 600; margin-bottom: 2px; color: #ffffff;"
    
    # Styles from: <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{subtitle}</div>
    desc_style = "font-size: 12px; opacity: 0.6; margin-bottom: 8px; color: #ffffff;"
    
    # Styles from: <div style="font-size: 28px; font-weight: 500;">{value}</div>
    value_style = "font-size: 28px; font-weight: 500; color: #ffffff;"
    
    # Pill style for percentage
    if port_ret >= 0:
        pill_bg = "rgba(46, 189, 133, 0.2)"
        pill_color = "#4ade80"
        arrow = "↑"
    else:
        pill_bg = "rgba(255, 69, 96, 0.2)"
        pill_color = "#ff4560"
        arrow = "↓"
        
    sub_style = (
        f"font-size: 14px; font-weight: 600; color: {pill_color}; "
        f"background-color: {pill_bg}; padding: 2px 8px; border-radius: 12px; "
        f"margin-left: 8px; display: inline-flex; align-items: center;"
    )
    
    def make_card(label, description, value, sub_value=None):
        if sub_value:
             # Flex row for Value + Percentage
             value_html = (
                 f'<div style="display:flex; align-items:baseline;">'
                 f'<span style="{value_style}">{value}</span>'
                 f'<span style="{sub_style}">{sub_value}</span>'
                 f'</div>'
             )
        else:
            value_html = f'<div style="{value_style}">{value}</div>'
            
        return (
            f'<div style="{card_style}">'
            f'<div style="{label_style}">{label}</div>'
            f'<div style="{desc_style}">{description}</div>'
            f'{value_html}'
            '</div>'
        )

    # Format values
    inv_str = f"{currency_label}{total_inv:,.0f}"
    val_str = f"{currency_label}{port_val:,.0f}"
    gain_str = f"{currency_label}{port_gain:,.0f}"
    input_str = f"{currency_label}{monthly_invest:,.0f}"
    input_str = f"{currency_label}{monthly_invest:,.0f}"
    ret_str = f"{arrow} {abs(port_ret):.1f}%"

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
        {make_card("Cumulative invested", "Total capital contributed", inv_str)}
        {make_card("Portfolio value", "Current market value", val_str, ret_str)}
        {make_card("Total gain", "Net profit or loss", gain_str)}
        {make_card("Monthly investment", "Current monthly contribution", input_str)}
    </div>
    """
    return html

# --- Data Fetching Functions ---

@st.cache_data(ttl=3600*24)
def fetch_market_data(ticker):
    """Fetches max available history for the ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="max", interval="1mo", auto_adjust=False)
        if data.index.tz is not None:
             data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching market data for {ticker}: {e}")
        return pd.DataFrame()

# --- Simulation Logic ---

def run_simulation(ticker, start_date, end_date, monthly_amount, market_full, fx_full=None, currency_code="USD"):
    # 1. Filter Market Data
    start_ts = pd.Timestamp(start_date).replace(day=1)
    end_ts = pd.Timestamp(end_date).replace(day=1)
    
    df = market_full[(market_full.index >= start_ts) & (market_full.index <= end_ts)].copy()
    
    if df.empty:
        return pd.DataFrame()

    # Normalize Columns (Asset)
    if 'Close' in df.columns: price_col = 'Close' 
    elif 'Adj Close' in df.columns: price_col = 'Adj Close'
    else: return pd.DataFrame()
        
    df = df[[price_col]].copy()
    df = df.rename(columns={price_col: 'Price'})
    df = df.resample('MS').first()
    
    # 2. Handle FX Data if needed
    if currency_code != "USD" and fx_full is not None:
         fx_df = fx_full[(fx_full.index >= start_ts) & (fx_full.index <= end_ts)].copy()
         # Normalize FX
         if 'Close' in fx_df.columns: fx_col = 'Close'
         else: fx_col = 'Adj Close' # fallback
         
         fx_df = fx_df[[fx_col]].copy()
         fx_df = fx_df.rename(columns={fx_col: 'FX_Rate'})
         fx_df = fx_df.resample('MS').first()
         
         # Align Data (Inner Join usually best to avoid missing rates/prices)
         df = df.join(fx_df, how='inner')
    else:
         df['FX_Rate'] = 1.0 # USD case

    # 3. Simulation Steps
    # We invest 'monthly_amount' in LOCAL currency.
    # Convert input to USD to buy shares.
    # Rate is Local/USD. So USD = Local / Rate.
    df['Input_Local'] = monthly_amount
    df['Input_USD'] = df['Input_Local'] / df['FX_Rate']
    
    df['Shares'] = df['Input_USD'] / df['Price']
    df['Total_Shares'] = df['Shares'].cumsum()
    
    # Portfolio Value in USD
    df['Portfolio_Value_USD'] = df['Total_Shares'] * df['Price']
    
    # Portfolio Value in Local Currency
    # Local = USD * Rate
    df['Portfolio_Value'] = df['Portfolio_Value_USD'] * df['FX_Rate']
    
    # Cumulative Invested (in nominal local currency)
    df['Invested_Total'] = df['Input_Local'].cumsum()
    
    # For compatibility, ensure 'Input' exists as what was invested (for returning ret calc)
    df['Input'] = df['Input_Local']
    
    return df

# Sidebar
st.sidebar.header("Configuration")

POPULAR_ASSETS = ["SPY", "QQQ", "VTI", "VOO", "SPUS", "GLD", "BTC-USD", "ETH-USD", "AAPL", "MSFT", "NVDA", "Other"]
ticker_choice = st.sidebar.selectbox("Select asset", options=POPULAR_ASSETS)

if ticker_choice == "Other":
    ticker_input = st.sidebar.text_input("Enter custom ticker", value="SPY")
else:
    ticker_input = ticker_choice




today = datetime.now().date()
default_start = datetime(2001, 1, 1).date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, today),
    min_value=datetime(1980, 1, 1).date(),
    max_value=today
)

if len(date_range) != 2:
    st.warning("Please select both a start and end date.")
    st.stop()

start_date_input, end_date_input = date_range
monthly_amt = st.sidebar.number_input("Monthly investment", min_value=10, value=100, step=10)

currency_code = st.sidebar.selectbox("Currency", options=["USD", "EUR", "GBP", "CHF"])




render_social_links()

# Run
with st.spinner("Fetching data..."):
    market_data = fetch_market_data(ticker_input)
    
    fx_data = None
    if currency_code != "USD":
        fx_ticker = FX_TICKERS.get(currency_code)
        if fx_ticker:
            fx_data = fetch_market_data(fx_ticker)
            if fx_data.empty:
                 st.warning(f"Could not load FX data for {currency_code}. Defaulting to USD.")
                 currency_code = "USD"
    
    if market_data.empty:
        st.error(f"Could not load data for {ticker_input}.")
        st.stop()

    results = run_simulation(ticker_input, start_date_input, end_date_input, monthly_amt, market_data, fx_data, currency_code)

if results.empty:
    st.warning("No data overlap found.")
    st.stop()

val_col = 'Portfolio_Value'
inv_col = 'Invested_Total'
input_col = 'Input'
currency_label = CURRENCY_MAP.get(currency_code, "$")

# --- Plot & Metrics (External) ---

# Metrics Placeholders
metric_cols = st.columns(4)
# Metrics Placeholders
metrics_placeholder = st.empty()


# Chart Placeholder
with st.container(border=True):
    chart_placeholder = st.empty()


def get_fig(data_subset, full_index, max_val_y):
    fig = go.Figure()
    
    # Trace 0: Portfolio Value (Line + Area)
    fig.add_trace(go.Scatter(
        x=data_subset.index, 
        y=data_subset[val_col],
        mode="lines", 
        name=f"Portfolio Value",
        line=dict(color="#1ed760", width=2),
        fill='tozeroy',
        fillcolor='rgba(30, 215, 96, 0.2)'
    ))

    # Trace 1: Current Value Marker
    if not data_subset.empty:
        last_pt = data_subset.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_pt.name],
            y=[last_pt[val_col]],
            mode="markers",
            name="Current Value",
            marker=dict(color="#1ed760", size=15, line=dict(color="white", width=3)),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Trace 2: Total Invested (Dashed Line)
    fig.add_trace(go.Scatter(
        x=data_subset.index, 
        y=data_subset[inv_col],
        mode="lines", 
        name=f"Total Invested",
        line=dict(color="#008FFB", width=2, dash='dash')
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[full_index[0], full_index[-1]], title="Year", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[0, max_val_y*1.1], title=f"Value ({currency_label})", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


start_anim_btn = st.button("Play animation", type="secondary")

if start_anim_btn:
    # Animation Loop
    step = max(1, len(results) // 100) 
    max_val = results[val_col].max()
    full_idx = results.index
    
    for i in range(1, len(results), step):
        subset = results.iloc[:i+1]
        curr = subset.iloc[-1]
        
        curr_val = curr[val_col]
        curr_inv = curr[inv_col]
        curr_gain = curr_val - curr_inv
        curr_ret = (curr_gain / curr_inv) * 100 if curr_inv > 0 else 0
        curr_input = curr[input_col]
        
        metrics_html = render_metrics_html(
            curr_inv, curr_val, curr_gain, curr_ret, curr_input, currency_label
        )
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
        
        fig = get_fig(subset, full_idx, max_val)
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"anim_{i}")
        
        time.sleep(0.1)
        
    # Final Frame
    curr = results.iloc[-1]
    curr_val = curr[val_col]
    curr_inv = curr[inv_col]
    curr_gain = curr_val - curr_inv
    curr_ret = (curr_gain / curr_inv) * 100 if curr_inv > 0 else 0
    curr_input = curr[input_col]

    metrics_html = render_metrics_html(
        curr_inv, curr_val, curr_gain, curr_ret, curr_input, currency_label
    )
    metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
    
    fig = get_fig(results, full_idx, max_val)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key="portfolio_chart")

else:
    # Static View
    curr = results.iloc[-1]
    curr_val = curr[val_col]
    curr_inv = curr[inv_col]
    curr_gain = curr_val - curr_inv
    curr_ret = (curr_gain / curr_inv) * 100 if curr_inv > 0 else 0
    curr_input = curr[input_col]
    
    metrics_html = render_metrics_html(
        curr_inv, curr_val, curr_gain, curr_ret, curr_input, currency_label
    )
    metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
    
    fig = get_fig(results, results.index, results[val_col].max())
    chart_placeholder.plotly_chart(fig, use_container_width=True, key="portfolio_chart")

with st.expander("See raw data"):
    st.dataframe(results.style.format("{:,.2f}"))
    
    csv = results.to_csv().encode('utf-8')
    st.download_button(
        label="Download data",
        data=csv,
        file_name=f'dca_simulation_{ticker_input}.csv',
        mime='text/csv',
    )
