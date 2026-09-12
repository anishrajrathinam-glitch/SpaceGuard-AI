"""
SpaceGuard AI - Frontend UI Components Module
Reusable presentation cards, gauges, risk badges, and alert banners
for the SpaceGuard AI mission control dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


def render_mission_header(title: str, subtitle: str, tag: str = "DEMO MODE — Sample/Historical Data"):
    """Renders the standard SpaceGuard mission control header."""
    st.markdown(f"""
        <div class="spaceguard-header">
            <h1 class="spaceguard-title">{title}</h1>
            <p class="spaceguard-subtitle">{subtitle}</p>
            <span class="demo-mode-tag">{tag}</span>
        </div>
    """, unsafe_allow_html=True)


def render_top_kpi_cards(
    active_zones: int,
    rain_prob: float,
    rain_risk_badge: str,
    rain_risk_color: str,
    flood_score: float,
    flood_badge: str,
    flood_color: str,
    threat_level: str,
    threat_badge: str,
    threat_color: str,
    model_status: str = "Random Forest Ensemble • 92.8% Acc • Active"
):
    """Renders the 5 top command center KPI cards requested by SIH judges."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
            <div class="sg-kpi-card">
                <div class="sg-kpi-label">📍 Active Monitoring Zones</div>
                <div class="sg-kpi-value" style="color: #38BDF8;">{active_zones} <span style="font-size: 14px;">Basins</span></div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Synchronized GIS Feeds</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="sg-kpi-card">
                <div class="sg-kpi-label">🌧️ Heavy Rain Probability</div>
                <div class="sg-kpi-value" style="color: {rain_risk_color};">{rain_prob}%</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{rain_risk_badge}</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="sg-kpi-card">
                <div class="sg-kpi-label">🌊 Flood Inundation Risk</div>
                <div class="sg-kpi-value" style="color: {flood_color};">{flood_score} <span style="font-size: 14px; color: #94A3B8;">/100</span></div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{flood_badge}</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="sg-kpi-card">
                <div class="sg-kpi-label">⚠️ Current Threat Level</div>
                <div class="sg-kpi-value" style="color: {threat_color}; font-size: 20px;">{threat_level}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{threat_badge}</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="sg-kpi-card">
                <div class="sg-kpi-label">🤖 Model Status</div>
                <div class="sg-kpi-value" style="color: #10B981; font-size: 18px;">OPERATIONAL</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{model_status}</div>
            </div>
        """, unsafe_allow_html=True)


def render_prediction_summary_card(
    location: str,
    probability_pct: float,
    risk_level: str,
    expected_intensity: str,
    risk_color: str,
    risk_badge: str,
    confidence: float
):
    """Renders the clean, high-impact prediction card."""
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 58, 138, 0.3) 100%); 
                    border: 2px solid {risk_color}; border-radius: 10px; padding: 22px; margin: 16px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 16px;">
                <h3 style="margin: 0; color: #F8FAFC; font-size: 22px;">🎯 Target Location: <span style="color: #38BDF8;">{location}</span></h3>
                <span style="background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}; padding: 6px 14px; border-radius: 20px; font-weight: bold;">{risk_badge}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; text-align: center;">
                <div style="background: #0B1120; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase;">Heavy Rainfall Probability</div>
                    <div style="font-size: 32px; font-weight: 800; color: {risk_color}; margin-top: 4px;">{probability_pct}%</div>
                </div>
                <div style="background: #0B1120; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase;">Risk Classification</div>
                    <div style="font-size: 26px; font-weight: 800; color: {risk_color}; margin-top: 8px;">{risk_level}</div>
                </div>
                <div style="background: #0B1120; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase;">Expected Intensity</div>
                    <div style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-top: 10px;">{expected_intensity}</div>
                </div>
            </div>
            <div style="font-size: 12px; color: #94A3B8; text-align: right; margin-top: 12px;">
                Model Prediction Confidence: <strong>{confidence}%</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_early_warning_card(
    location: str,
    risk_level: str,
    probability_pct: float,
    expected_intensity: str,
    recommended_actions: List[str]
):
    """Renders the clean early warning card format requested by the user."""
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%);
                    border-left: 6px solid #EF4444; border-top: 1px solid rgba(239, 68, 68, 0.4);
                    border-right: 1px solid rgba(239, 68, 68, 0.4); border-bottom: 1px solid rgba(239, 68, 68, 0.4);
                    border-radius: 8px; padding: 22px; margin: 18px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h2 style="color: #EF4444; margin: 0; font-size: 22px;">🚨 EARLY WARNING ADVISORY</h2>
                <span class="badge-extreme">CRITICAL DISASTER ALERT</span>
            </div>
            <hr style="margin: 12px 0; border-color: rgba(239, 68, 68, 0.3);">
            <div style="font-size: 15px; color: #F1F5F9; line-height: 1.8;">
                • <strong>Location:</strong> <span style="color: #38BDF8; font-weight: bold;">{location}</span><br>
                • <strong>Risk Level:</strong> <span style="color: #EF4444; font-weight: bold;">{risk_level}</span><br>
                • <strong>Heavy Rain Probability:</strong> <span style="color: #EF4444; font-weight: bold;">{probability_pct}%</span><br>
                • <strong>Expected Intensity:</strong> <strong>{expected_intensity}</strong>
            </div>
            <div style="margin-top: 14px; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 14px;">
                <h4 style="color: #FBBF24; margin: 0 0 8px 0;">Recommended Action Protocols:</h4>
                <ul style="margin: 0; padding-left: 20px; color: #CBD5E1; font-size: 13px; line-height: 1.6;">
                    {"".join(f"<li>{act}</li>" for act in recommended_actions)}
                </ul>
            </div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 12px;">
                Dispatched by SpaceGuard AI Prototype Early Warning Module (Demo Sandbox Mode).
            </div>
        </div>
    """, unsafe_allow_html=True)
