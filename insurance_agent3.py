import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Dict, List
import random

# Page config
st.set_page_config(
    page_title="Parametric Insurance Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2980b9 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    
    .policy-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .trigger-alert {
        background: linear-gradient(135deg, #fff3cd 0%, #fef5e7 100%);
        border: 2px solid #f39c12;
        padding: 1.5rem;
        border-radius: 12px;
        color: #8b6914;
        box-shadow: 0 4px 15px rgba(243,156,18,0.2);
    }
    
    .success-alert {
        background: linear-gradient(135deg, #d1edff 0%, #e8f4fd 100%);
        border: 2px solid #3498db;
        padding: 1.5rem;
        border-radius: 12px;
        color: #2471a3;
        box-shadow: 0 4px 15px rgba(52,152,219,0.2);
    }
    
    .danger-alert {
        background: linear-gradient(135deg, #fadbd8 0%, #f8d7da 100%);
        border: 2px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 12px;
        color: #a93226;
        box-shadow: 0 4px 15px rgba(231,76,60,0.2);
    }
    
    .info-card {
        background: linear-gradient(145deg, #ffffff 0%, #f7f9fc 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e3e6ea;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    
    .sidebar {
        background: #f8f9fa;
    }
    
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-active {
        background: #d4edda;
        color: #155724;
    }
    
    .status-triggered {
        background: #f8d7da;
        color: #721c24;
    }
    
    .status-processing {
        background: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with expanded categories
if 'policies' not in st.session_state:
    st.session_state.policies = [
        {
            'id': 'CROP001',
            'type': 'Agricultural Insurance',
            'category': 'Agriculture',
            'customer': 'Green Valley Farms',
            'parameter': 'rainfall',
            'threshold': 50,
            'premium': 5000,
            'coverage': 100000,
            'status': 'active',
            'current_value': 45,
            'location': 'Maharashtra',
            'unit': 'mm/month'
        },
        {
            'id': 'FLIGHT002',
            'type': 'Travel Delay Insurance',
            'category': 'Travel',
            'customer': 'Corporate Traveler',
            'parameter': 'delay_minutes',
            'threshold': 120,
            'premium': 200,
            'coverage': 5000,
            'status': 'active',
            'current_value': 180,
            'location': 'Mumbai Airport',
            'unit': 'minutes'
        },
        {
            'id': 'CAT003',
            'type': 'Catastrophe Bond',
            'category': 'Natural Disaster',
            'customer': 'City Infrastructure',
            'parameter': 'earthquake_magnitude',
            'threshold': 6.0,
            'premium': 50000,
            'coverage': 10000000,
            'status': 'active',
            'current_value': 4.2,
            'location': 'Delhi',
            'unit': 'richter scale'
        },
        {
            'id': 'WIND004',
            'type': 'Wind Speed Insurance',
            'category': 'Natural Disaster',
            'customer': 'Renewable Energy Corp',
            'parameter': 'wind_speed',
            'threshold': 80,
            'premium': 15000,
            'coverage': 500000,
            'status': 'active',
            'current_value': 65,
            'location': 'Gujarat Coast',
            'unit': 'km/h'
        },
        {
            'id': 'TEMP005',
            'type': 'Temperature Insurance',
            'category': 'Agriculture',
            'customer': 'Dairy Cooperative',
            'parameter': 'temperature',
            'threshold': 42,
            'premium': 8000,
            'coverage': 300000,
            'status': 'active',
            'current_value': 39,
            'location': 'Punjab',
            'unit': '°C'
        },
        {
            'id': 'CYBER006',
            'type': 'Cyber Attack Insurance',
            'category': 'Technology',
            'customer': 'FinTech Solutions',
            'parameter': 'attack_severity',
            'threshold': 7,
            'premium': 25000,
            'coverage': 2000000,
            'status': 'active',
            'current_value': 3,
            'location': 'Bangalore',
            'unit': 'threat level'
        },
        {
            'id': 'ENERGY007',
            'type': 'Solar Irradiance Insurance',
            'category': 'Energy',
            'customer': 'Solar Power Plant',
            'parameter': 'solar_irradiance',
            'threshold': 4.5,
            'premium': 12000,
            'coverage': 800000,
            'status': 'active',
            'current_value': 5.2,
            'location': 'Rajasthan',
            'unit': 'kWh/m²/day'
        },
        {
            'id': 'MARINE008',
            'type': 'Wave Height Insurance',
            'category': 'Marine',
            'customer': 'Shipping Corporation',
            'parameter': 'wave_height',
            'threshold': 8,
            'premium': 18000,
            'coverage': 1500000,
            'status': 'active',
            'current_value': 6.5,
            'location': 'Arabian Sea',
            'unit': 'meters'
        },
        {
            'id': 'AUTO009',
            'type': 'Vehicle Telematics Insurance',
            'category': 'Automotive',
            'customer': 'Fleet Management Co.',
            'parameter': 'accident_risk_score',
            'threshold': 85,
            'premium': 3000,
            'coverage': 150000,
            'status': 'active',
            'current_value': 78,
            'location': 'Chennai',
            'unit': 'risk score'
        },
        {
            'id': 'HEALTH010',
            'type': 'Pandemic Response Insurance',
            'category': 'Healthcare',
            'customer': 'Hospital Network',
            'parameter': 'infection_rate',
            'threshold': 5,
            'premium': 35000,
            'coverage': 5000000,
            'status': 'active',
            'current_value': 2.1,
            'location': 'Karnataka',
            'unit': '% population'
        }
    ]

if 'claim_history' not in st.session_state:
    st.session_state.claim_history = [
        {'date': '2024-06-27', 'policy': 'FLIGHT002', 'amount': 1500, 'trigger': 'Flight Delay 180min', 'status': 'Processing'},
        {'date': '2024-06-25', 'policy': 'CROP001', 'amount': 25000, 'trigger': 'Low Rainfall 35mm', 'status': 'Paid'},
        {'date': '2024-06-22', 'policy': 'WIND004', 'amount': 45000, 'trigger': 'High Wind Speed 95km/h', 'status': 'Paid'},
        {'date': '2024-06-20', 'policy': 'TEMP005', 'amount': 18000, 'trigger': 'High Temperature 44°C', 'status': 'Paid'},
        {'date': '2024-06-18', 'policy': 'MARINE008', 'amount': 85000, 'trigger': 'Wave Height 9.2m', 'status': 'Paid'},
        {'date': '2024-06-15', 'policy': 'CYBER006', 'amount': 120000, 'trigger': 'Cyber Attack Level 8', 'status': 'Paid'}
    ]

if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = ""

# Helper functions
def get_ollama_response(prompt: str) -> str:
    """Get AI response from Ollama local model with better error handling"""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "phi3.5:3.8b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'Analysis completed.')
        else:
            return "**AI Analysis (Demo Mode)**: Local AI model unavailable. Using rule-based assessment. Start Ollama with: `ollama serve`"
    
    except Exception as e:
        return "**AI Analysis (Demo Mode)**: Local AI model unavailable. Using rule-based assessment. Start Ollama with: `ollama serve`"

def calculate_payout(policy: Dict) -> int:
    """Calculate payout based on parameter severity"""
    current = policy['current_value']
    threshold = policy['threshold']
    coverage = policy['coverage']
    
    # Different trigger logic for different parameters
    if policy['parameter'] in ['rainfall', 'solar_irradiance']:
        # Lower values trigger payout
        if current < threshold:
            severity = (threshold - current) / threshold
            return int(coverage * severity * 0.6)
    else:
        # Higher values trigger payout
        if current > threshold:
            severity = min(1.0, (current - threshold) / threshold)
            payout_ratio = {
                'delay_minutes': 0.3,
                'earthquake_magnitude': 0.8,
                'wind_speed': 0.5,
                'temperature': 0.4,
                'attack_severity': 0.7,
                'wave_height': 0.6,
                'accident_risk_score': 0.4,
                'infection_rate': 0.9
            }.get(policy['parameter'], 0.5)
            return int(coverage * severity * payout_ratio)
    return 0

def update_parameters():
    """Simulate real-time parameter updates"""
    parameter_ranges = {
        'rainfall': (20, 80),
        'delay_minutes': (0, 300),
        'earthquake_magnitude': (2.0, 7.5),
        'wind_speed': (40, 120),
        'temperature': (25, 50),
        'attack_severity': (1, 10),
        'solar_irradiance': (3.0, 7.0),
        'wave_height': (2, 12),
        'accident_risk_score': (30, 100),
        'infection_rate': (0.5, 8.0)
    }
    
    for i, policy in enumerate(st.session_state.policies):
        param = policy['parameter']
        if param in parameter_ranges:
            min_val, max_val = parameter_ranges[param]
            st.session_state.policies[i]['current_value'] = random.uniform(min_val, max_val)

def check_trigger_conditions(policy: Dict):
    """Check if trigger conditions are met and process claims"""
    payout = calculate_payout(policy)
    
    if payout > 0:
        # Add to claim history
        new_claim = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'policy': policy['id'],
            'amount': payout,
            'trigger': f"{policy['parameter'].replace('_', ' ').title()}: {policy['current_value']:.1f} {policy['unit']}",
            'status': 'Processing'
        }
        st.session_state.claim_history.insert(0, new_claim)
        
        # Get AI analysis
        prompt = f"""Analyze this parametric insurance trigger:
        Policy: {policy['type']}
        Category: {policy['category']}
        Parameter: {policy['parameter']} 
        Current Value: {policy['current_value']:.1f} {policy['unit']}
        Threshold: {policy['threshold']} {policy['unit']}
        Customer: {policy['customer']}
        Calculated Payout: ₹{payout:,}
        
        Provide risk analysis and payout decision with reasoning."""
        
        st.session_state.ai_analysis = get_ollama_response(prompt)
        return True, payout
    else:
        st.session_state.ai_analysis = f"**Risk Assessment Complete**\n\nCurrent {policy['parameter'].replace('_', ' ').title()} value ({policy['current_value']:.1f} {policy['unit']}) is within acceptable parameters. Threshold: {policy['threshold']} {policy['unit']}.\n\nContinuing real-time monitoring for policy {policy['id']}."
        return False, 0

# Main App Layout
st.markdown("""
<div class="main-header">
    <h1>Parametric Insurance Platform</h1>
    <p>AI-Powered Automated Claims Processing & Real-time Risk Monitoring</p>
    <small>Enterprise Grade Insurance Technology Solution</small>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    
    # Category Filter
    categories = list(set([p['category'] for p in st.session_state.policies]))
    selected_category = st.selectbox("Filter by Category", ["All"] + sorted(categories))
    
    # Filter policies based on category
    if selected_category == "All":
        filtered_policies = st.session_state.policies
    else:
        filtered_policies = [p for p in st.session_state.policies if p['category'] == selected_category]
    
    # Policy Selection
    if filtered_policies:
        policy_options = [f"{p['id']} - {p['type']}" for p in filtered_policies]
        selected_idx = st.selectbox("Select Policy", range(len(policy_options)), 
                                   format_func=lambda x: policy_options[x])
        st.session_state.selected_policy = filtered_policies[selected_idx]
    else:
        st.warning("No policies match the selected criteria")
        st.session_state.selected_policy = st.session_state.policies[0]

    
    st.divider()
    
    # Real-time Controls
    st.subheader("Real-time Monitoring")
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

    auto_update = st.checkbox("Auto-update Parameters", value=False)
    
    if auto_update:
        current_time = time.time()
        if current_time - st.session_state.last_update > 10:
            update_parameters()
            st.session_state.last_update = current_time
            st.rerun()
        
        next_update = 10 - int(current_time - st.session_state.last_update)
        st.info(f"Next update in: {max(0, next_update)}s")
    
    if st.button("Update All Parameters Now"):
        update_parameters()
        st.session_state.last_update = time.time()
        st.rerun()
    
    # Manual parameter adjustment
    st.subheader("Manual Parameter Control")
    policy = st.session_state.selected_policy
    
    parameter_configs = {
        'rainfall': {'min': 0.0, 'max': 100.0, 'step': 0.1},
        'delay_minutes': {'min': 0.0, 'max': 400.0, 'step': 1.0},
        'earthquake_magnitude': {'min': 0.0, 'max': 10.0, 'step': 0.1},
        'wind_speed': {'min': 0.0, 'max': 150.0, 'step': 1.0},
        'temperature': {'min': 0.0, 'max': 60.0, 'step': 0.1},
        'attack_severity': {'min': 1, 'max': 10, 'step': 1},
        'solar_irradiance': {'min': 0.0, 'max': 10.0, 'step': 0.1},
        'wave_height': {'min': 0.0, 'max': 15.0, 'step': 0.1},
        'accident_risk_score': {'min': 0, 'max': 100, 'step': 1},
        'infection_rate': {'min': 0.0, 'max': 10.0, 'step': 0.1}
    }
    
    param = policy['parameter']
if param in parameter_configs:
    config = parameter_configs[param]
    # Ensure current_value is a number, not a list
    current_val = policy['current_value']
    if isinstance(current_val, (list, tuple)):
        current_val = float(current_val[0]) if current_val else config['min']
    else:
        current_val = float(current_val)
    
    new_value = st.slider(
        f"{param.replace('_', ' ').title()} ({policy['unit']})",
        config['min'], config['max'], current_val, config['step']
    )

if st.button("Update Parameter"):
    if 'new_value' in locals() and new_value != current_val:
            for i, p in enumerate(st.session_state.policies):
                if p['id'] == policy['id']:
                    st.session_state.policies[i]['current_value'] = new_value
                    break
            st.success(f"Parameter updated to {new_value}")
            st.rerun()

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # Policy Dashboard
    st.header("Policy Dashboard")
    
    # Selected Policy Details
    policy = st.session_state.selected_policy
    
    st.markdown(f"""
    <div class="policy-card">
        <h3>{policy['id']} - {policy['type']}</h3>
        <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
            <div>
                <p><strong>Customer:</strong> {policy['customer']}</p>
                <p><strong>Location:</strong> {policy['location']}</p>
                <p><strong>Category:</strong> {policy['category']}</p>
            </div>
            <div>
                <p><strong>Coverage:</strong> ₹{policy['coverage']:,}</p>
                <p><strong>Premium:</strong> ₹{policy['premium']:,}</p>
                <p><strong>Parameter:</strong> {policy['parameter'].replace('_', ' ').title()}</p>
            </div>
        </div>
        <p><strong>Trigger Threshold:</strong> {policy['threshold']} {policy['unit']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Status
    current_val = policy['current_value']
    threshold = policy['threshold']
    
    # Determine status based on parameter type
    if policy['parameter'] in ['rainfall', 'solar_irradiance']:
        is_triggered = current_val < threshold
    else:
        is_triggered = current_val > threshold
    
    status_text = "TRIGGER CONDITION MET" if is_triggered else "NORMAL RANGE"
    status_class = "danger-alert" if is_triggered else "success-alert"
    
    # Status Display
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric(
            label=f"Current {policy['parameter'].replace('_', ' ').title()}",
            value=f"{current_val:.1f} {policy['unit']}",
            delta=f"Threshold: {threshold} {policy['unit']}"
        )
    
    with col_b:
        st.metric(
            label="Status",
            value=status_text,
            delta="Real-time"
        )
    
    with col_c:
        payout_amount = calculate_payout(policy)
        st.metric(
            label="Potential Payout",
            value=f"₹{payout_amount:,}",
            delta="Auto-calculated"
        )
    
    # Trigger Analysis Button
    st.divider()
    
    col_x, col_y, col_z = st.columns([1, 2, 1])
    with col_y:
        if st.button("ANALYZE TRIGGER CONDITIONS", type="primary", use_container_width=True):
            with st.spinner("AI Agent analyzing parametric conditions..."):
                time.sleep(2)
                triggered, payout = check_trigger_conditions(policy)
                st.rerun()
    
    # Parameter Visualization
    st.subheader("Parameter Trend Analysis")
    
    # Generate sample time series data
    dates = pd.date_range(start='2024-06-01', end='2024-06-28', freq='D')
    
    # Generate realistic data based on parameter type
    if policy['parameter'] == 'rainfall':
        values = np.random.normal(45, 15, len(dates))
        values = np.maximum(0, values)
    elif policy['parameter'] == 'delay_minutes':
        values = np.random.exponential(60, len(dates))
    elif policy['parameter'] == 'earthquake_magnitude':
        values = np.random.normal(4.0, 1.0, len(dates))
        values = np.maximum(0, values)
    elif policy['parameter'] == 'wind_speed':
        values = np.random.normal(60, 20, len(dates))
        values = np.maximum(0, values)
    elif policy['parameter'] == 'temperature':
        values = np.random.normal(35, 8, len(dates))
    elif policy['parameter'] == 'attack_severity':
        values = np.random.choice(range(1, 11), len(dates))
    elif policy['parameter'] == 'solar_irradiance':
        values = np.random.normal(5.5, 1.2, len(dates))
        values = np.maximum(0, values)
    elif policy['parameter'] == 'wave_height':
        values = np.random.normal(5, 2.5, len(dates))
        values = np.maximum(0, values)
    elif policy['parameter'] == 'accident_risk_score':
        values = np.random.normal(60, 20, len(dates))
        values = np.clip(values, 0, 100)
    else:  # infection_rate
        values = np.random.exponential(2, len(dates))
    
    # Add current value as latest point
    values[-1] = current_val
    
    df = pd.DataFrame({
        'Date': dates,
        'Value': values,
        'Threshold': threshold
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', 
                            name=f'{policy["parameter"].replace("_", " ").title()}', 
                            line=dict(color='#3498db', width=3)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", 
                  annotation_text=f"Trigger Threshold: {threshold}")
    
    # Highlight trigger zones
    if policy['parameter'] in ['rainfall', 'solar_irradiance']:
        fig.add_hrect(y0=0, y1=threshold, fillcolor="#e74c3c", opacity=0.1, 
                      annotation_text="Trigger Zone", annotation_position="top left")
    else:
        fig.add_hrect(y0=threshold, y1=df['Value'].max()*1.2, fillcolor="#e74c3c", opacity=0.1,
                      annotation_text="Trigger Zone", annotation_position="bottom left")
    
    fig.update_layout(
        title=f"{policy['parameter'].replace('_', ' ').title()} Monitoring - {policy['location']}",
        xaxis_title="Date",
        yaxis_title=f"{policy['parameter'].replace('_', ' ').title()} ({policy['unit']})",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # AI Analysis Panel
    st.header("AI Risk Analysis")
    
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = ""

    if st.session_state.ai_analysis:
        if any(keyword in st.session_state.ai_analysis.upper() for keyword in ["PAYOUT", "TRIGGER", "AUTHORIZED"]):
            alert_class = "danger-alert"
        else:
            alert_class = "success-alert"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <h4>Latest AI Analysis</h4>
            {st.session_state.ai_analysis.replace('\n', '<br>')}
            <br><small>Updated: {datetime.now().strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Analyze Trigger Conditions' to get AI-powered risk assessment")
    
    # Claims History
    st.divider()
    st.subheader("Recent Claims")
    
    for claim in st.session_state.claim_history[:6]:
        status_color = {
            'Paid': '#28a745',
            'Processing': '#ffc107',
            'Rejected': '#dc3545'
        }.get(claim['status'], '#6c757d')
        
        st.markdown(f"""
        <div class="info-card">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <strong>{claim['policy']}</strong><br>
                    <small style="color: #666;">{claim['trigger']}</small>
                </div>
                <div style="text-align: right;">
                    <div style="color: {status_color}; font-weight: bold; font-size: 1.1rem;">
                        ₹{claim['amount']:,}
                    </div>
                    <small style="color: #666;">{claim['date']}</small><br>
                    <span class="status-badge" style="background: {status_color}20; color: {status_color};">
                        {claim['status']}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary Statistics
    st.divider()
    st.subheader("Portfolio Overview")
    
    total_coverage = sum(p['coverage'] for p in st.session_state.policies)
    total_payouts = sum(c['amount'] for c in st.session_state.claim_history)
    active_policies = len([p for p in st.session_state.policies if p['status'] == 'active'])
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.metric("Active Policies", active_policies)
        st.metric("Total Coverage", f"₹{total_coverage:,}")
    
    with col_stat2:
        st.metric("Claims Processed", len(st.session_state.claim_history))
        st.metric("Total Payouts", f"₹{total_payouts:,}")

# Portfolio Summary Section
st.divider()
st.header("Portfolio Summary by Category")

# Create category summary
category_summary = {}
for policy in st.session_state.policies:
    cat = policy['category']
    if cat not in category_summary:
        category_summary[cat] = {'count': 0, 'coverage': 0, 'premium': 0}
    category_summary[cat]['count'] += 1
    category_summary[cat]['coverage'] += policy['coverage']
    category_summary[cat]['premium'] += policy['premium']

# Display category cards
cols = st.columns(min(len(category_summary), 4))
for i, (category, data) in enumerate(category_summary.items()):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{category}</h4>
            <p><strong>Policies:</strong> {data['count']}</p>
            <p><strong>Coverage:</strong> ₹{data['coverage']:,}</p>
            <p><strong>Premiums:</strong> ₹{data['premium']:,}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem; background: #f8f9fa; border-radius: 8px;">
    <strong>Parametric Insurance Platform</strong> | Powered by AI Risk Analytics | Real-time Monitoring System
    <br><small>Enterprise Insurance Technology Solution - Built for BFSI Innovation</small>
</div>
""", unsafe_allow_html=True)
