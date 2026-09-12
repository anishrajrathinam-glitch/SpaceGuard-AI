"""
SpaceGuard AI - Real-Time Integrated Inference Engine
Predicts heavy rainfall probability, estimates flood/inundation risk scores,
classifies multi-disaster severity tiers, and triggers early warning criteria.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.preprocess import DataPreprocessor, FEATURE_COLUMNS
from flood.inundation import FloodInundationEngine


class RainfallPredictor:
    """
    Integrated prediction engine for heavy rainfall and flood inundation risk.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        low_threshold: float = 0.40,
        high_threshold: float = 0.70
    ):
        self.model_dir = model_dir or os.path.join(BASE_DIR, "model", "saved_model")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.model = None
        self.preprocessor = DataPreprocessor(model_dir=self.model_dir)
        self.flood_engine = FloodInundationEngine()
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """
        Loads the trained model or triggers training pipeline if not present.
        """
        model_path = os.path.join(self.model_dir, "rainfall_model.joblib")
        if not os.path.exists(model_path):
            print("Notice: Saved model not found. Initializing automatic model training...")
            from model.train_model import train_rainfall_model
            self.model, _ = train_rainfall_model()
        else:
            self.model = joblib.load(model_path)

    def determine_risk_level(self, probability: float) -> Dict[str, str]:
        """
        Maps heavy rainfall probability to risk tiers.
        """
        if probability < self.low_threshold:
            return {
                "level": "LOW",
                "color": "#10B981", # Green
                "badge": "🟢 LOW RISK",
                "status": "NORMAL CONDITIONS — Heavy Rainfall Unlikely"
            }
        elif probability <= self.high_threshold:
            return {
                "level": "MEDIUM",
                "color": "#F59E0B", # Amber
                "badge": "🟡 MEDIUM RISK",
                "status": "ADVISORY — Moderate to Heavy Showers Possible"
            }
        else:
            return {
                "level": "HIGH",
                "color": "#EF4444", # Red
                "badge": "🔴 HIGH RISK",
                "status": "CRITICAL ALERT — Heavy / Severe Rainfall Likely"
            }

    def estimate_rainfall_intensity(self, probability: float, input_data: Dict[str, Any]) -> str:
        """
        Estimates expected rainfall intensity category based on storm probability
        and convective atmospheric factors.
        """
        if probability < 0.20:
            return "NONE_TO_LIGHT (< 15.5 mm)"
        elif probability < 0.40:
            return "MODERATE (15.6 - 64.4 mm)"

        ctt = input_data.get("cloud_top_temp_c", -40.0)
        cape = input_data.get("convective_available_potential_energy", 1500.0)
        humidity = input_data.get("relative_humidity_pct", 85.0)

        if probability > 0.85 and ctt < -65.0 and (cape > 2800 or humidity > 94.0):
            return "EXTREMELY HEAVY (> 204.4 mm)"
        elif probability > 0.70 or ctt < -55.0:
            return "VERY HEAVY (115.6 - 204.4 mm)"
        else:
            return "HEAVY (64.5 - 115.5 mm)"

    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs dual-stage inference:
        1. ML heavy rainfall classification & probability
        2. Hydrological flood & inundation risk estimation
        """
        self._ensure_model_loaded()

        # Transform and scale input for ML rainfall model
        X_scaled, df_features = self.preprocessor.transform_input(input_features)

        # Compute probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            heavy_rain_prob = float(probs[1])
        else:
            pred = int(self.model.predict(X_scaled)[0])
            heavy_rain_prob = float(pred)

        risk_info = self.determine_risk_level(heavy_rain_prob)
        intensity = self.estimate_rainfall_intensity(heavy_rain_prob, input_features)

        # Estimate quantitative rainfall amount (in mm) if not explicitly provided
        pred_rain_mm = float(input_features.get("predicted_rainfall_mm", 0.0))
        if pred_rain_mm <= 0.0:
            # Physics-based proxy for estimated precipitation from probability and antecedents
            prev_rain = float(input_features.get("previous_24h_rainfall_mm", 20.0))
            if heavy_rain_prob > 0.80:
                pred_rain_mm = round(max(70.0, prev_rain * 1.25 + 30.0), 1)
            elif heavy_rain_prob > 0.50:
                pred_rain_mm = round(max(35.0, prev_rain * 0.90 + 15.0), 1)
            else:
                pred_rain_mm = round(max(0.0, prev_rain * 0.40), 1)

        # Hydrological Flood & Inundation Evaluation
        elevation = float(input_features.get("elevation_m", 25.0))
        slope = float(input_features.get("slope_deg", 2.0))
        soil_sat = float(input_features.get("soil_saturation_pct", 75.0))
        drainage = float(input_features.get("drainage_density_km_km2", 1.4))
        runoff_coeff = float(input_features.get("runoff_coefficient_pct", 75.0))
        hist_flood = float(input_features.get("historical_flood_proneness", 0.70))

        location = input_features.get("location", "Selected Region")

        flood_risk_data = FloodInundationEngine.calculate_inundation_risk(
            predicted_rainfall_mm=pred_rain_mm,
            rainfall_probability_pct=round(heavy_rain_prob * 100, 1),
            elevation_m=elevation,
            slope_deg=slope,
            soil_saturation_pct=soil_sat,
            drainage_density_km_km2=drainage,
            runoff_coefficient_pct=runoff_coeff,
            historical_flood_proneness=hist_flood,
            location_name=location
        )

        # 4-tier Combined Overall Disaster Alert Level (GREEN, YELLOW, ORANGE, RED)
        overall_score = flood_risk_data["overall_disaster_risk"]
        if overall_score <= 30.0:
            alert_tier = "GREEN"
            alert_badge = "🟢 GREEN — WATCH"
            alert_color = "#10B981"
        elif overall_score <= 60.0:
            alert_tier = "YELLOW"
            alert_badge = "🟡 YELLOW — ADVISORY"
            alert_color = "#F59E0B"
        elif overall_score <= 80.0:
            alert_tier = "ORANGE"
            alert_badge = "🟠 ORANGE — WARNING"
            alert_color = "#F97316"
        else:
            alert_tier = "RED"
            alert_badge = "🔴 RED — CRITICAL ALERT"
            alert_color = "#EF4444"

        # Confidence metric
        confidence = float(np.round((abs(heavy_rain_prob - 0.5) * 2 * 0.45 + 0.55) * 100, 1))

        result = {
            "location": location,
            "heavy_rainfall_probability": round(heavy_rain_prob, 4),
            "probability_pct": round(heavy_rain_prob * 100, 1),
            "predicted_rainfall_mm": pred_rain_mm,
            "prediction_class": 1 if heavy_rain_prob >= 0.50 else 0,
            "is_heavy_rainfall": heavy_rain_prob >= self.low_threshold,
            "rainfall_risk_level": risk_info["level"],
            "rainfall_risk_color": risk_info["color"],
            "rainfall_risk_badge": risk_info["badge"],
            "rainfall_status": risk_info["status"],
            "intensity": intensity,
            "model_confidence": confidence,
            "warning_triggered": alert_tier in ["ORANGE", "RED"],
            # Flood Inundation Metrics
            "flood_inundation_score": flood_risk_data["inundation_risk_score"],
            "flood_probability_pct": flood_risk_data["flood_probability_pct"],
            "flood_severity": flood_risk_data["flood_severity"],
            "flood_severity_badge": flood_risk_data["severity_badge"],
            "flood_severity_color": flood_risk_data["severity_color"],
            "affected_area_km2": flood_risk_data["affected_area_km2"],
            "water_accumulation_risk": flood_risk_data["water_accumulation_risk"],
            "accumulation_rate": flood_risk_data["accumulation_rate"],
            "vulnerable_zones": flood_risk_data["vulnerable_zones"],
            "road_lowlying_risk": flood_risk_data["road_lowlying_risk"],
            "flood_components": flood_risk_data["components"],
            # Unified Multi-Hazard Metrics
            "overall_disaster_risk": overall_score,
            "alert_tier": alert_tier,
            "alert_badge": alert_badge,
            "alert_color": alert_color,
            "feature_values": df_features.iloc[0].to_dict(),
            "raw_scaled_array": X_scaled,
            "recommended_action": input_features.get(
                "recommended_action",
                "Monitor district catchment levels and check drainage infrastructure."
            )
        }

        return result
