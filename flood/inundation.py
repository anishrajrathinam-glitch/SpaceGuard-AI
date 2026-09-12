"""
SpaceGuard AI - Flood & Inundation Risk Module
Hydrological runoff, terrain elevation, slope, soil moisture saturation,
vulnerable zone identification, and drainage susceptibility modeling.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class FloodInundationEngine:
    """
    Estimates flood and inundation risk scores (0-100), severity categories,
    affected waterlogged area, road/underpass inundation risk, and vulnerable zones.
    
    NOTE: DEMO / SIMULATION — Prototype Hydrological Catchment Model.
    """

    SEVERITY_TIERS = {
        "LOW": {"range": (0, 30), "color": "#10B981", "badge": "🟢 LOW FLOOD RISK", "status": "Stable Catchment — No Significant Inundation"},
        "MEDIUM": {"range": (31, 60), "color": "#F59E0B", "badge": "🟡 MEDIUM FLOOD RISK", "status": "Advisory — Localized Waterlogging in Low-Lying Drains"},
        "HIGH": {"range": (61, 80), "color": "#F97316", "badge": "🟠 HIGH FLOOD RISK", "status": "Warning — Severe Flash Flood / Inundation Imminent"},
        "EXTREME": {"range": (81, 100), "color": "#EF4444", "badge": "🔴 EXTREME FLOOD RISK", "status": "CRITICAL EMERGENCY — Widespread Catastrophic Submersion"}
    }

    def __init__(self):
        pass

    @staticmethod
    def calculate_inundation_risk(
        predicted_rainfall_mm: float,
        rainfall_probability_pct: float,
        elevation_m: float,
        slope_deg: float,
        soil_saturation_pct: float,
        drainage_density_km_km2: float = 1.8,
        runoff_coefficient_pct: float = 75.0,
        historical_flood_proneness: float = 0.70,
        location_name: str = "Target Basin"
    ) -> Dict[str, Any]:
        """
        Calculates multi-criteria flood inundation metrics:
        - Inundation Risk Score (0-100)
        - Flood Probability (%)
        - Estimated Inundated Area (km^2)
        - Vulnerable Zones
        - Road / Subway Underpass Inundation Risk
        - Water Accumulation Depth & Rate
        """
        # 1. Rainfall Forcing (0 to 1)
        rain_effective = (predicted_rainfall_mm * (rainfall_probability_pct / 100.0))
        rain_score = np.clip(rain_effective / 180.0, 0.0, 1.0)

        # 2. Soil Saturation Term (0 to 1)
        soil_score = np.clip(soil_saturation_pct / 100.0, 0.0, 1.0)
        if soil_saturation_pct > 85.0:
            soil_score = min(1.0, soil_score * 1.15)

        # 3. Topographic Susceptibility (0 to 1)
        elev_factor = np.clip(1.0 - (elevation_m / 400.0), 0.1, 1.0)
        slope_factor = np.clip(1.0 - (slope_deg / 18.0), 0.1, 1.0)
        topo_score = 0.6 * slope_factor + 0.4 * elev_factor

        # 4. Drainage & Runoff Coefficient Term (0 to 1)
        runoff_score = np.clip((runoff_coefficient_pct / 100.0) * (2.5 / max(0.5, drainage_density_km_km2)), 0.0, 1.0)

        # Composite Inundation Score (0 to 100)
        raw_score = (
            (0.40 * rain_score) +
            (0.25 * soil_score) +
            (0.20 * topo_score) +
            (0.15 * runoff_score)
        ) * 100.0

        inundation_score = float(np.clip(round(raw_score * (0.90 + 0.20 * historical_flood_proneness), 1), 0.0, 100.0))

        # Severity Classification
        if inundation_score <= 30.0:
            severity = "LOW"
            accumulation_desc = "Low Runoff Absorption (0 - 2 cm ponding)"
            accum_rate = "0.5 cm/hr"
        elif inundation_score <= 60.0:
            severity = "MEDIUM"
            accumulation_desc = "Moderate Puddling & Gutter Overflow (5 - 15 cm)"
            accum_rate = "2.5 cm/hr"
        elif inundation_score <= 80.0:
            severity = "HIGH"
            accumulation_desc = "Severe Street Inundation (20 - 50 cm waterlogging)"
            accum_rate = "6.0 cm/hr"
        else:
            severity = "EXTREME"
            accumulation_desc = "Catastrophic Sheet Submersion (> 60 cm water depth)"
            accum_rate = "14.5 cm/hr"

        tier_info = FloodInundationEngine.SEVERITY_TIERS[severity]

        # Estimated Flood Probability (%)
        flood_prob = float(np.clip(round(inundation_score * 0.95 + 4.0, 1), 5.0, 99.0))

        # Estimated Affected Waterlogged Area (km^2 proxy)
        base_urban_area_km2 = 280.0
        affected_area_km2 = round((inundation_score / 100.0) ** 1.6 * (base_urban_area_km2 * 0.45), 1)

        # Location-specific Vulnerable Zones & Road Infrastructure Risk
        vulnerable_zones = [
            f"Low-lying riparian flood plains along {location_name} waterways",
            f"Dense urban catchment basins with impervious coverage > {int(runoff_coefficient_pct)}%",
            "Natural estuarine drainage outfalls & canal convergence junctions",
            "Hillside debris flow cones in slopes > 15°"
        ]

        road_risks = [
            {"facility": "Subway Underpasses", "risk": "CRITICAL" if inundation_score > 65 else "MODERATE", "action": "Automatic barricade closure required"},
            {"facility": "Arterial Highway Sinks", "risk": "HIGH" if inundation_score > 55 else "LOW", "action": "Divert heavy traffic to elevated expressways"},
            {"facility": "Railway Cuttings", "risk": "CRITICAL" if inundation_score > 75 else "LOW", "action": "Deploy high-volume de-watering pumps"},
            {"facility": "Hospital Basement Power Stations", "risk": "HIGH" if inundation_score > 60 else "LOW", "action": "Erect temporary flood barrier berms"}
        ]

        # Overall Multi-Disaster Risk Score
        overall_disaster_risk = float(np.clip(
            round(0.45 * rainfall_probability_pct + 0.55 * inundation_score, 1),
            0.0, 100.0
        ))

        return {
            "inundation_risk_score": inundation_score,
            "flood_probability_pct": flood_prob,
            "flood_severity": severity,
            "severity_color": tier_info["color"],
            "severity_badge": tier_info["badge"],
            "status_description": tier_info["status"],
            "affected_area_km2": affected_area_km2,
            "water_accumulation_risk": accumulation_desc,
            "accumulation_rate": accum_rate,
            "vulnerable_zones": vulnerable_zones,
            "road_lowlying_risk": road_risks,
            "rainfall_risk_score": round(rainfall_probability_pct, 1),
            "overall_disaster_risk": overall_disaster_risk,
            "components": {
                "rainfall_forcing_pct": round(rain_score * 100, 1),
                "soil_saturation_pct": round(soil_score * 100, 1),
                "topographic_sink_pct": round(topo_score * 100, 1),
                "drainage_impedance_pct": round(runoff_score * 100, 1),
            },
            "is_demo_simulation": True,
            "disclaimer": "DEMO / SIMULATION — Prototype Hydrological Inundation Estimation"
        }
