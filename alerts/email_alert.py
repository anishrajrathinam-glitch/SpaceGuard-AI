"""
SpaceGuard AI - Multi-Disaster Early Warning & Alert Dispatch Module
Dispatches high-priority heavy rainfall & flood inundation warnings with 4-tier severity.
Supports live SMTP sending via environment variables with safe sandbox simulation mode.
"""

import os
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional


class AlertDispatcher:
    """
    Manages multi-hazard early warnings and disaster mitigation alert notifications.
    Never exposes credentials; pulls dynamically from environment variables.
    """

    SEVERITY_TIERS = {
        "GREEN": {"name": "LOW WATCH", "color": "#10B981", "badge": "🟢 GREEN — WATCH"},
        "YELLOW": {"name": "MEDIUM ADVISORY", "color": "#F59E0B", "badge": "🟡 YELLOW — ADVISORY"},
        "ORANGE": {"name": "HIGH WARNING", "color": "#F97316", "badge": "🟠 ORANGE — WARNING"},
        "RED": {"name": "CRITICAL EMERGENCY", "color": "#EF4444", "badge": "🔴 RED — CRITICAL ALERT"}
    }

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(self.log_dir, "alert_dispatch_log.json")
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves log of recently dispatched early warnings.
        """
        try:
            with open(self.log_file, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def log_alert(self, alert_entry: Dict[str, Any]):
        """
        Appends alert record to history.
        """
        history = self.get_alert_history()
        history.insert(0, alert_entry)
        history = history[:50]
        with open(self.log_file, "w") as f:
            json.dump(history, f, indent=2)

    def send_early_warning_alert(
        self,
        recipient_email: str,
        location: str,
        probability_pct: float,
        flood_risk_score: float,
        overall_disaster_risk: float,
        intensity: str,
        severity_tier: str = "RED",
        top_factors: List[str] = None,
        recommended_actions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Dispatches multi-hazard early warning notification.
        If SMTP environment variables are configured, dispatches real email;
        otherwise dispatches via verified prototype sandbox mode.
        """
        top_factors = top_factors or ["High Atmospheric Moisture (>90%)", "Freezing Cloud Top Core (< -65°C)", "Saturated Drainage Catchment"]
        recommended_actions = recommended_actions or [
            "Monitor low-lying catchment basins and urban canals.",
            "Pre-position District Emergency Response Teams (DDMA/SDRF).",
            "Operate mechanical drainage pump stations and sluices.",
            "Issue localized public safety advisories for road transit.",
            "Prepare pre-identified evacuation routes if water levels rise."
        ]
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

        tier_info = self.SEVERITY_TIERS.get(severity_tier, self.SEVERITY_TIERS["RED"])

        smtp_server = os.getenv("SPACEGUARD_SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SPACEGUARD_SMTP_PORT", "587"))
        sender_email = os.getenv("SPACEGUARD_EMAIL_SENDER")
        sender_password = os.getenv("SPACEGUARD_EMAIL_PASSWORD")

        subject = f"🚨 SPACEGUARD {severity_tier} ALERT: {location} — Rainfall {probability_pct}% | Flood Risk {flood_risk_score}/100"

        plain_text = f"""
======================================================================
SPACEGUARD AI — MULTI-HAZARD DISASTER EARLY WARNING ADVISORY
======================================================================
Alert Level: {tier_info['badge']}
Location: {location}
Issuance Timestamp: {timestamp_str}

HAZARD ASSESSMENT:
- Heavy Rainfall Probability: {probability_pct}%
- Forecast Storm Intensity: {intensity}
- Hydrological Flood Inundation Score: {flood_risk_score}/100
- Composite Disaster Risk Score: {overall_disaster_risk}/100

METEOROLOGICAL DRIVERS (XAI ATTRIBUTION):
{chr(10).join(f"- {fac}" for fac in top_factors)}

DECISION-SUPPORT ACTION PROTOCOLS:
{chr(10).join(f"- {act}" for act in recommended_actions)}

DISCLAIMER:
Prototype early-warning simulation. Live SMTP enabled if credentials configured.
======================================================================
"""

        html_content = f"""
        <div style="font-family: Arial, sans-serif; background-color: #070B14; color: #f8fafc; padding: 24px; border-radius: 8px; border: 1px solid #1e3a8a;">
            <div style="background-color: {tier_info['color']}; color: white; padding: 12px 18px; border-radius: 6px; font-weight: bold; font-size: 18px;">
                {tier_info['badge']} — MULTI-HAZARD EMERGENCY ADVISORY
            </div>
            <h2 style="color: #38bdf8; margin-top: 18px;">Disaster Threat Imminent: {location}</h2>
            <table style="width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px;">
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b;"><strong>Heavy Rain Probability:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b; color: {tier_info['color']}; font-weight: bold; font-size: 18px;">{probability_pct}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b;"><strong>Flood Inundation Score:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b; color: {tier_info['color']}; font-weight: bold; font-size: 18px;">{flood_risk_score} / 100</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b;"><strong>Overall Disaster Risk:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b; font-weight: bold;">{overall_disaster_risk} / 100</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b;"><strong>Expected Storm Intensity:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #1e293b;">{intensity}</td>
                </tr>
            </table>

            <div style="background-color: #0f172a; padding: 14px; border-radius: 6px; margin: 16px 0; border: 1px solid #334155;">
                <h4 style="color: #60a5fa; margin-top: 0;">Identified Contributing Meteorological Factors (SHAP XAI):</h4>
                <ul>
                    {"".join(f"<li>{fac}</li>" for fac in top_factors)}
                </ul>
            </div>

            <div style="background-color: #1e1b4b; padding: 14px; border-radius: 6px; margin: 16px 0; border-left: 4px solid #f59e0b;">
                <h4 style="color: #fbbf24; margin-top: 0;">Decision-Support Recommended Action Protocols:</h4>
                <ul>
                    {"".join(f"<li>{act}</li>" for act in recommended_actions)}
                </ul>
            </div>

            <p style="font-size: 11px; color: #94a3af; margin-top: 20px;">
                Dispatched by SpaceGuard AI Prototype Early Warning Module. (Demo Mode Sandbox).
            </p>
        </div>
        """

        is_live_sent = False
        status_message = ""

        if sender_email and sender_password:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = sender_email
                msg["To"] = recipient_email
                msg.attach(MIMEText(plain_text, "plain"))
                msg.attach(MIMEText(html_content, "html"))

                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, recipient_email, msg.as_string())

                is_live_sent = True
                status_message = f"Live alert email dispatched successfully to {recipient_email} via SMTP."
            except Exception as e:
                status_message = f"SMTP live dispatch error ({e}). Switched to verified prototype simulation mode."
        else:
            status_message = f"Prototype Sandbox Mode: Alert advisory generated and logged for {recipient_email}. (To enable live SMTP dispatch, configure SPACEGUARD_EMAIL_SENDER & SPACEGUARD_EMAIL_PASSWORD env variables)."

        alert_entry = {
            "timestamp": timestamp_str,
            "recipient": recipient_email,
            "location": location,
            "probability_pct": probability_pct,
            "flood_risk_score": flood_risk_score,
            "overall_disaster_risk": overall_disaster_risk,
            "severity_tier": severity_tier,
            "intensity": intensity,
            "live_smtp_sent": is_live_sent,
            "status_message": status_message,
            "recommended_actions": recommended_actions,
        }

        self.log_alert(alert_entry)
        return alert_entry
