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
    page_title="Parametric Insurance Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .policy-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .trigger-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        color: #856404;
    }
    .success-alert {
        background: #d1edff;
        border: 1px solid #74b9ff;
        padding: 1rem;
        border-radius: 8px;
        color: #0984e3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'policies' not in st.session_state:
    st.session_state.policies = [
        {
            'id': 'CROP001',
            'type': 'Crop Insurance',
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
            'type': 'Flight Delay Insurance',
            'customer': 'Business Traveler',
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
            'customer': 'City Infrastructure',
            'parameter': 'earthquake_magnitude',
            'threshold': 6.0,
            'premium': 50000,
            'coverage': 10000000,
            'status': 'active',
            'current_value': 4.2,
            'location': 'Delhi',
            'unit': 'richter scale'
        }
    ]

if 'claim_history' not in st.session_state:
    st.session_state.claim_history = [
        {'date': '2024-06-20', 'policy': 'CROP001', 'amount': 25000, 'trigger': 'Low Rainfall (35mm)', 'status': 'Paid'},
        {'date': '2024-06-15', 'policy': 'FLIGHT002', 'amount': 1500, 'trigger': 'Flight Delay 150min', 'status': 'Paid'},
        {'date': '2024-05-30', 'policy': 'CROP001', 'amount': 15000, 'trigger': 'Drought Conditions', 'status': 'Paid'}
    ]

if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = ""

if 'triggers' not in st.session_state:
    st.session_state.triggers = []

# Helper functions
def get_ollama_response(prompt: str) -> str:
    """Get AI response from Ollama local model"""
    try:
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        data = {
            "model": "llama3.1:8b",  # Change to your downloaded model
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'Analysis completed.')
        else:
            # Fallback response if Ollama is not running
            return f"⚠️ **Ollama Model Analysis**\n\nLocal AI model detected parametric condition. Current parameter value requires immediate attention. Please ensure Ollama is running with: `ollama serve`"
    
    except Exception as e:
        # Fallback for demo
        policy = st.session_state.selected_policy
        if policy['parameter'] == 'rainfall' and policy['current_value'] < policy['threshold']:
            return f"🌧️ **Llama 3.1 Analysis**: Drought trigger detected. Rainfall {policy['current_value']:.1f}mm is below {policy['threshold']}mm threshold. **PAYOUT APPROVED** ✅"
        elif policy['parameter'] == 'delay_minutes' and policy['current_value'] > policy['threshold']:
            return f"✈️ **Llama 3.1 Analysis**: Flight delay {policy['current_value']:.0f} minutes exceeds {policy['threshold']} minute threshold. **PAYOUT APPROVED** ✅"
        elif policy['parameter'] == 'earthquake_magnitude' and policy['current_value'] > policy['threshold']:
            return f"🏗️ **Llama 3.1 Analysis**: Seismic activity {policy['current_value']:.1f} exceeds threshold {policy['threshold']}. **CATASTROPHE PAYOUT TRIGGERED** ⚠️"
        else:
            return f"✅ **Llama 3.1 Analysis**: Parameter {policy['current_value']:.1f} within normal range. No trigger conditions met."

def calculate_payout(policy: Dict) -> int:
    """Calculate payout based on parameter severity"""
    if policy['parameter'] == 'rainfall':
        if policy['current_value'] < policy['threshold']:
            severity = (policy['threshold'] - policy['current_value']) / policy['threshold']
            return int(policy['coverage'] * severity * 0.5)
    elif policy['parameter'] == 'delay_minutes':
        if policy['current_value'] > policy['threshold']:
            severity = min(1.0, (policy['current_value'] - policy['threshold']) / policy['threshold'])
            return int(policy['coverage'] * severity * 0.3)
    elif policy['parameter'] == 'earthquake_magnitude':
        if policy['current_value'] > policy['threshold']:
            severity = min(1.0, (policy['current_value'] - policy['threshold']) / 2.0)
            return int(policy['coverage'] * severity * 0.8)
    return 0

def update_parameters():
    """Simulate real-time parameter updates"""
    for i, policy in enumerate(st.session_state.policies):
        if policy['parameter'] == 'rainfall':
            st.session_state.policies[i]['current_value'] = random.uniform(20, 80)
        elif policy['parameter'] == 'delay_minutes':
            st.session_state.policies[i]['current_value'] = random.uniform(0, 300)
        elif policy['parameter'] == 'earthquake_magnitude':
            st.session_state.policies[i]['current_value'] = random.uniform(2.0, 7.5)

def check_trigger_conditions(policy: Dict):
    """Check if trigger conditions are met and process claims"""
    payout = calculate_payout(policy)
    
    if payout > 0:
        # Add to claim history
        new_claim = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'policy': policy['id'],
            'amount': payout,
            'trigger': f"{policy['parameter']}: {policy['current_value']:.1f} {policy['unit']}",
            'status': 'Processing'
        }
        st.session_state.claim_history.insert(0, new_claim)
        
        # Get AI analysis
        prompt = f"""Analyze this parametric insurance trigger:
        Policy: {policy['type']}
        Parameter: {policy['parameter']} 
        Current Value: {policy['current_value']:.1f} {policy['unit']}
        Threshold: {policy['threshold']} {policy['unit']}
        Customer: {policy['customer']}
        Calculated Payout: ₹{payout:,}
        
        Provide analysis and payout decision."""
        
        st.session_state.ai_analysis = get_ollama_response(prompt)
        return True, payout
    else:
        st.session_state.ai_analysis = f"✅ **No Trigger Detected**\n\nCurrent {policy['parameter']} value ({policy['current_value']:.1f} {policy['unit']}) is within acceptable parameters. Threshold: {policy['threshold']} {policy['unit']}.\n\nContinuing real-time monitoring..."
        return False, 0

# Main App Layout
st.markdown("""
<div class="main-header">
    <h1>🛡️ Parametric Insurance Agent</h1>
    <p>AI-Powered Instant Claims & Real-time Risk Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Sidebar

# Fixed Policy Selection with proper error handling
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Policy Selection - Fixed
    if 'policies' in st.session_state and st.session_state.policies:
        policy_options = [f"{p['id']} - {p['type']}" for p in st.session_state.policies]
        
        # Use direct selectbox with options instead of indices
        selected_policy_str = st.selectbox(
            "Select Policy", 
            options=policy_options,
            index=0  # Default to first policy
        )
        
        # Find the selected policy by matching the string
        selected_idx = policy_options.index(selected_policy_str)
        st.session_state.selected_policy = st.session_state.policies[selected_idx]
        
        st.divider()
        
        # Real-time Controls
        st.subheader("📡 Real-time Monitoring")
        
        auto_update = st.checkbox("Auto-update Parameters", value=False)
        if auto_update:
            if st.button("🔄 Update Now"):
                update_parameters()
                st.rerun()
        
        # Manual parameter adjustment
        st.subheader("🎚️ Manual Parameter Control")
        
        # Get current policy safely
        if 'selected_policy' in st.session_state:
            policy = st.session_state.selected_policy
            
            # Parameter-specific sliders with proper key management
            if policy['parameter'] == 'rainfall':
                new_value = st.slider(
                    "Rainfall (mm/month)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=float(policy['current_value']), 
                    step=0.1,
                    key=f"rainfall_slider_{policy['id']}"
                )
            elif policy['parameter'] == 'delay_minutes':
                new_value = st.slider(
                    "Delay (minutes)", 
                    min_value=0.0, 
                    max_value=400.0, 
                    value=float(policy['current_value']), 
                    step=1.0,
                    key=f"delay_slider_{policy['id']}"
                )
            elif policy['parameter'] == 'earthquake_magnitude':
                new_value = st.slider(
                    "Earthquake Magnitude", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=float(policy['current_value']), 
                    step=0.1,
                    key=f"earthquake_slider_{policy['id']}"
                )
            else:
                st.error(f"Unknown parameter type: {policy['parameter']}")
                new_value = policy['current_value']
            
            # Update button with proper state management
            if st.button("Update Parameter", key=f"update_btn_{policy['id']}"):
                # Update the specific policy in session state
                for i, p in enumerate(st.session_state.policies):
                    if p['id'] == policy['id']:
                        st.session_state.policies[i]['current_value'] = new_value
                        break
                
                # Force refresh to show updated values
                st.rerun()
        
        else:
            st.error("No policy selected. Please check your session state.")
    
    else:
        st.error("No policies found. Please initialize policies in session state.")

# Alternative approach - more robust policy selection
def get_policy_selection_robust():
    """More robust policy selection with better error handling"""
    
    if 'policies' not in st.session_state or not st.session_state.policies:
        st.error("No policies available")
        return None
    
    # Create policy display options
    policy_display = {}
    for i, policy in enumerate(st.session_state.policies):
        display_name = f"{policy['id']} - {policy['type']}"
        policy_display[display_name] = i
    
    # Selectbox with string options
    selected_display = st.selectbox(
        "Select Policy",
        options=list(policy_display.keys()),
        key="policy_selector"
    )
    
    if selected_display:
        policy_idx = policy_display[selected_display]
        return st.session_state.policies[policy_idx]
    
    return None

# Usage of the robust function
def sidebar_with_robust_selection():
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Get selected policy
        selected_policy = get_policy_selection_robust()
        
        if selected_policy:
            st.session_state.selected_policy = selected_policy
            
            st.divider()
            
            # Rest of your sidebar code here...
            st.subheader("📡 Real-time Monitoring")
            
            auto_update = st.checkbox("Auto-update Parameters", value=False)
            
            # Manual parameter control
            st.subheader("🎚️ Manual Parameter Control")
            
            # Use unique keys for each slider to prevent conflicts
            param_type = selected_policy['parameter']
            policy_id = selected_policy['id']
            current_val = selected_policy['current_value']
            
            if param_type == 'rainfall':
                new_value = st.slider(
                    "Rainfall (mm/month)",
                    0.0, 100.0, 
                    float(current_val),
                    0.1,
                    key=f"slider_rainfall_{policy_id}"
                )
            elif param_type == 'delay_minutes':
                new_value = st.slider(
                    "Delay (minutes)",
                    0.0, 400.0,
                    float(current_val),
                    1.0,
                    key=f"slider_delay_{policy_id}"
                )
            elif param_type == 'earthquake_magnitude':
                new_value = st.slider(
                    "Earthquake Magnitude",
                    0.0, 10.0,
                    float(current_val),
                    0.1,
                    key=f"slider_earthquake_{policy_id}"
                )
            
            if st.button("Update Parameter", key=f"update_{policy_id}"):
                # Update the policy
                for i, policy in enumerate(st.session_state.policies):
                    if policy['id'] == policy_id:
                        st.session_state.policies[i]['current_value'] = new_value
                        break
                st.rerun()
# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # Policy Dashboard
    st.header("📊 Policy Dashboard")
    
    # Selected Policy Details
    policy = st.session_state.selected_policy
    
    st.markdown(f"""
    <div class="policy-card">
        <h3>🏷️ {policy['id']} - {policy['type']}</h3>
        <p><strong>Customer:</strong> {policy['customer']}</p>
        <p><strong>Location:</strong> {policy['location']}</p>
        <p><strong>Coverage:</strong> ₹{policy['coverage']:,}</p>
        <p><strong>Premium:</strong> ₹{policy['premium']:,}</p>
        <p><strong>Parameter:</strong> {policy['parameter']} (Threshold: {policy['threshold']} {policy['unit']})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Status
    current_val = policy['current_value']
    threshold = policy['threshold']
    
    # Determine status
    if policy['parameter'] == 'rainfall':
        is_triggered = current_val < threshold
        status_color = "🔴" if is_triggered else "🟢"
        status_text = "TRIGGER CONDITION MET" if is_triggered else "NORMAL RANGE"
    else:
        is_triggered = current_val > threshold
        status_color = "🔴" if is_triggered else "🟢"
        status_text = "TRIGGER CONDITION MET" if is_triggered else "NORMAL RANGE"
    
    # Status Display
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric(
            label=f"Current {policy['parameter'].title()}",
            value=f"{current_val:.1f} {policy['unit']}",
            delta=f"Threshold: {threshold} {policy['unit']}"
        )
    
    with col_b:
        st.metric(
            label="Status",
            value=status_text,
            delta=status_color
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
    
    col_x, col_y, col_z = st.columns([1, 1, 1])
    with col_y:
        if st.button("🔍 **ANALYZE TRIGGER CONDITIONS**", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Agent analyzing parameters..."):
                time.sleep(2)  # Simulate processing
                triggered, payout = check_trigger_conditions(policy)
                st.rerun()
    
    # Parameter Visualization
    st.subheader("📈 Parameter Trend Analysis")
    
    # Generate sample time series data
    dates = pd.date_range(start='2024-06-01', end='2024-06-28', freq='D')
    if policy['parameter'] == 'rainfall':
        values = np.random.normal(45, 15, len(dates))
        values = np.maximum(0, values)  # No negative rainfall
    elif policy['parameter'] == 'delay_minutes':
        values = np.random.exponential(60, len(dates))
    else:
        values = np.random.normal(4.0, 1.0, len(dates))
        values = np.maximum(0, values)
    
    # Add current value as latest point
    values[-1] = current_val
    
    df = pd.DataFrame({
        'Date': dates,
        'Value': values,
        'Threshold': threshold
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', 
                            name=f'{policy["parameter"].title()}', line=dict(color='blue', width=3)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Trigger Threshold: {threshold}")
    
    # Highlight trigger zones
    if policy['parameter'] == 'rainfall':
        fig.add_hrect(y0=0, y1=threshold, fillcolor="red", opacity=0.1, 
                      annotation_text="Trigger Zone", annotation_position="top left")
    else:
        fig.add_hrect(y0=threshold, y1=df['Value'].max()*1.2, fillcolor="red", opacity=0.1,
                      annotation_text="Trigger Zone", annotation_position="bottom left")
    
    fig.update_layout(
        title=f"{policy['parameter'].title()} Monitoring - {policy['location']}",
        xaxis_title="Date",
        yaxis_title=f"{policy['parameter'].title()} ({policy['unit']})",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # AI Analysis Panel
    st.header("🤖 AI Analysis")
    
    if st.session_state.ai_analysis:
        st.markdown(f"""
        <div class="success-alert">
            <h4>Latest Analysis</h4>
            {st.session_state.ai_analysis}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Click 'Analyze Trigger Conditions' to get AI-powered risk assessment")
    
    # Claims History
    st.divider()
    st.subheader("💰 Recent Payouts")
    
    for claim in st.session_state.claim_history[:5]:
        status_emoji = "✅" if claim['status'] == 'Paid' else "⏳"
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #28a745;">
            <strong>{claim['policy']}</strong> {status_emoji}<br>
            <small>{claim['trigger']}</small><br>
            <span style="color: #28a745; font-weight: bold;">₹{claim['amount']:,}</span> | {claim['date']}
        </div>
        """, unsafe_allow_html=True)
    
    # Summary Statistics
    st.divider()
    st.subheader("📈 Today's Summary")
    
    total_coverage = sum(p['coverage'] for p in st.session_state.policies)
    total_payouts = sum(c['amount'] for c in st.session_state.claim_history)
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.metric("Active Policies", len(st.session_state.policies))
        st.metric("Total Coverage", f"₹{total_coverage:,}")
    
    with col_stat2:
        st.metric("Claims Processed", len(st.session_state.claim_history))
        st.metric("Total Payouts", f"₹{total_payouts:,}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🛡️ <strong>Parametric Insurance Agent</strong> | Powered by AI | Real-time Risk Monitoring
    <br><small>Built for BFSI Innovation Hackathon 2024</small>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time effect
if auto_update:
    time.sleep(3)
    st.rerun()