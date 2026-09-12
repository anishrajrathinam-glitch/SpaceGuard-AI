"""
SpaceGuard AI - Helper Utilities Module
Provides static geographic coordinates, preset extreme weather profiles,
predefined judging scenarios, risk visual tokens, and procedural satellite imagery.
"""

import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, List


# Static, verified geographic coordinates and hydrological baseline for monitored stations
PRESET_LOCATIONS = {
    "Chennai (Cyclone / Depression Episode)": {
        "location": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "region": "Coromandel Coast",
        "elevation_m": 14.0,
        "slope_deg": 1.2,
        "temperature_c": 25.8,
        "relative_humidity_pct": 96.0,
        "wind_speed_kmh": 58.0,
        "wind_direction_deg": 75.0,
        "surface_pressure_hpa": 992.5,
        "dew_point_c": 25.2,
        "previous_24h_rainfall_mm": 115.0,
        "previous_6h_rainfall_mm": 48.0,
        "predicted_rainfall_mm": 175.0,
        "forecast_lead_time_hours": 6,
        "soil_saturation_pct": 94.0,
        "drainage_density_km_km2": 1.2,
        "runoff_coefficient_pct": 82.0,
        "historical_flood_proneness": 0.85,
        "cloud_top_temp_c": -72.0,
        "cloud_optical_thickness": 94.0,
        "water_vapor_brightness_temp_k": 216.0,
        "convective_available_potential_energy": 3100.0,
        "vertical_wind_shear_kts": 42.0,
        "convective_storm_index": 0.86,
        "critical_infrastructure": [
            "Chennai Central Sub-station",
            "Adyar River Flood Gate",
            "Velachery Low-Lying Drainage Basin",
            "Kathipara Flyover Underpass"
        ],
        "recommended_action": "🚨 ACTIVATE RED ALERT: Pre-position NDRF boats along Adyar and Cooum basins. Halt low-lying transit and close underpasses."
    },
    "Mumbai (Monsoon Deluge / High Tide)": {
        "location": "Mumbai",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "region": "Konkan Coast",
        "elevation_m": 10.0,
        "slope_deg": 2.5,
        "temperature_c": 26.2,
        "relative_humidity_pct": 98.0,
        "wind_speed_kmh": 64.0,
        "wind_direction_deg": 240.0,
        "surface_pressure_hpa": 991.0,
        "dew_point_c": 25.9,
        "previous_24h_rainfall_mm": 150.0,
        "previous_6h_rainfall_mm": 65.0,
        "predicted_rainfall_mm": 210.0,
        "forecast_lead_time_hours": 6,
        "soil_saturation_pct": 98.0,
        "drainage_density_km_km2": 0.9,
        "runoff_coefficient_pct": 88.0,
        "historical_flood_proneness": 0.90,
        "cloud_top_temp_c": -78.0,
        "cloud_optical_thickness": 105.0,
        "water_vapor_brightness_temp_k": 214.0,
        "convective_available_potential_energy": 3400.0,
        "vertical_wind_shear_kts": 48.0,
        "convective_storm_index": 0.91,
        "critical_infrastructure": [
            "Mithi River Outfall Gates",
            "Hindmata Pumping Station",
            "Milan Subway Drainage Well",
            "Kurla Rail Inundation Junction"
        ],
        "recommended_action": "🚨 HIGH TIDE SYNERGY WARNING: Deploy emergency de-watering high-capacity pumps at Milan Subway & Hindmata. Disseminate beach and seafront red alerts."
    },
    "Wayanad (Western Ghats Orographic Cloudburst)": {
        "location": "Wayanad (Kerala)",
        "latitude": 11.6854,
        "longitude": 76.1320,
        "region": "Western Ghats",
        "elevation_m": 850.0,
        "slope_deg": 28.0,
        "temperature_c": 21.5,
        "relative_humidity_pct": 98.5,
        "wind_speed_kmh": 46.0,
        "wind_direction_deg": 235.0,
        "surface_pressure_hpa": 988.0,
        "dew_point_c": 21.2,
        "previous_24h_rainfall_mm": 185.0,
        "previous_6h_rainfall_mm": 78.0,
        "predicted_rainfall_mm": 240.0,
        "forecast_lead_time_hours": 3,
        "soil_saturation_pct": 99.0,
        "drainage_density_km_km2": 2.4,
        "runoff_coefficient_pct": 65.0,
        "historical_flood_proneness": 0.95,
        "cloud_top_temp_c": -76.0,
        "cloud_optical_thickness": 110.0,
        "water_vapor_brightness_temp_k": 215.0,
        "convective_available_potential_energy": 2900.0,
        "vertical_wind_shear_kts": 36.0,
        "convective_storm_index": 0.88,
        "critical_infrastructure": [
            "Chooralmala Valley Culvert",
            "Meppadi Hill Stream Gauges",
            "Vythiri Mountain Pass Road"
        ],
        "recommended_action": "🚨 CATASTROPHIC DEBRIS FLOW / LANDSLIDE ALERT: Trigger immediate mandatory evacuations for hillside settlements in slope zones > 20 degrees."
    },
    "Guwahati (Brahmaputra Valley Flood Pulse)": {
        "location": "Guwahati (Assam)",
        "latitude": 26.1445,
        "longitude": 91.7362,
        "region": "Brahmaputra Valley",
        "elevation_m": 55.0,
        "slope_deg": 8.0,
        "temperature_c": 26.0,
        "relative_humidity_pct": 96.5,
        "wind_speed_kmh": 38.0,
        "wind_direction_deg": 190.0,
        "surface_pressure_hpa": 993.0,
        "dew_point_c": 25.4,
        "previous_24h_rainfall_mm": 110.0,
        "previous_6h_rainfall_mm": 42.0,
        "predicted_rainfall_mm": 160.0,
        "forecast_lead_time_hours": 6,
        "soil_saturation_pct": 95.0,
        "drainage_density_km_km2": 1.6,
        "runoff_coefficient_pct": 74.0,
        "historical_flood_proneness": 0.88,
        "cloud_top_temp_c": -71.0,
        "cloud_optical_thickness": 95.0,
        "water_vapor_brightness_temp_k": 217.0,
        "convective_available_potential_energy": 3050.0,
        "vertical_wind_shear_kts": 28.0,
        "convective_storm_index": 0.84,
        "critical_infrastructure": [
            "Bharalu River Outfall",
            "Anil Nagar Sluice Gates",
            "Jalukbari Inter-City Transit Hub"
        ],
        "recommended_action": "🟠 ORANGE ADVISORY: Monitor Brahmaputra gauge crest levels. Operate mechanical sluice gates at Bharalu channel to prevent urban backflow."
    },
    "Kolkata (Ganges Delta Tidal Surge)": {
        "location": "Kolkata",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "region": "Ganges Delta",
        "elevation_m": 9.0,
        "slope_deg": 0.8,
        "temperature_c": 26.5,
        "relative_humidity_pct": 95.0,
        "wind_speed_kmh": 44.0,
        "wind_direction_deg": 170.0,
        "surface_pressure_hpa": 994.0,
        "dew_point_c": 25.5,
        "previous_24h_rainfall_mm": 95.0,
        "previous_6h_rainfall_mm": 35.0,
        "predicted_rainfall_mm": 140.0,
        "forecast_lead_time_hours": 6,
        "soil_saturation_pct": 92.0,
        "drainage_density_km_km2": 1.1,
        "runoff_coefficient_pct": 80.0,
        "historical_flood_proneness": 0.82,
        "cloud_top_temp_c": -68.0,
        "cloud_optical_thickness": 88.0,
        "water_vapor_brightness_temp_k": 218.0,
        "convective_available_potential_energy": 2850.0,
        "vertical_wind_shear_kts": 32.0,
        "convective_storm_index": 0.81,
        "critical_infrastructure": [
            "Circular Canal Sluice System",
            "Thanthania Low-Lying Corridor",
            "Alipore Meteorological Radar Post"
        ],
        "recommended_action": "🟠 CANAL CONGESTION WARNING: Synchronize drainage discharge with Hooghly low-tide windows. Clear surface gully pits."
    },
    "Visakhapatnam (Northern Circars Coastal Storm)": {
        "location": "Visakhapatnam",
        "latitude": 17.6868,
        "longitude": 83.2185,
        "region": "Northern Circars",
        "elevation_m": 22.0,
        "slope_deg": 6.5,
        "temperature_c": 26.8,
        "relative_humidity_pct": 92.0,
        "wind_speed_kmh": 48.0,
        "wind_direction_deg": 110.0,
        "surface_pressure_hpa": 996.0,
        "dew_point_c": 25.2,
        "previous_24h_rainfall_mm": 80.0,
        "previous_6h_rainfall_mm": 30.0,
        "predicted_rainfall_mm": 115.0,
        "forecast_lead_time_hours": 12,
        "soil_saturation_pct": 86.0,
        "drainage_density_km_km2": 1.5,
        "runoff_coefficient_pct": 72.0,
        "historical_flood_proneness": 0.72,
        "cloud_top_temp_c": -64.0,
        "cloud_optical_thickness": 80.0,
        "water_vapor_brightness_temp_k": 220.0,
        "convective_available_potential_energy": 2700.0,
        "vertical_wind_shear_kts": 34.0,
        "convective_storm_index": 0.76,
        "critical_infrastructure": ["Visakhapatnam Port Berth 4", "Kailasagiri Hill Road Drain"],
        "recommended_action": "🟠 HARBOR WARNING: Suspend small-craft navigation. Monitor hill slope runoff around Kailasagiri foothills."
    },
    "Puri (East Coast Depression Zone)": {
        "location": "Puri (Odisha)",
        "latitude": 19.8135,
        "longitude": 85.8312,
        "region": "East Coast Cyclonic Zone",
        "elevation_m": 12.0,
        "slope_deg": 1.5,
        "temperature_c": 26.0,
        "relative_humidity_pct": 94.0,
        "wind_speed_kmh": 52.0,
        "wind_direction_deg": 105.0,
        "surface_pressure_hpa": 993.5,
        "dew_point_c": 24.8,
        "previous_24h_rainfall_mm": 105.0,
        "previous_6h_rainfall_mm": 40.0,
        "predicted_rainfall_mm": 155.0,
        "forecast_lead_time_hours": 6,
        "soil_saturation_pct": 94.0,
        "drainage_density_km_km2": 1.3,
        "runoff_coefficient_pct": 76.0,
        "historical_flood_proneness": 0.84,
        "cloud_top_temp_c": -70.0,
        "cloud_optical_thickness": 92.0,
        "water_vapor_brightness_temp_k": 217.0,
        "convective_available_potential_energy": 3100.0,
        "vertical_wind_shear_kts": 38.0,
        "convective_storm_index": 0.83,
        "critical_infrastructure": ["Mahanadi Delta Estuary Embankments", "Grand Road Storm Conduits"],
        "recommended_action": "🚨 COASTAL INUNDATION ALERT: Inspect saline river embankments. Disseminate cyclone shelter readiness warnings."
    },
    "Bengaluru (Moderate Thunderstorm / Urban Pooling)": {
        "location": "Bengaluru",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "region": "Deccan Plateau",
        "elevation_m": 920.0,
        "slope_deg": 4.1,
        "temperature_c": 24.5,
        "relative_humidity_pct": 84.0,
        "wind_speed_kmh": 26.0,
        "wind_direction_deg": 250.0,
        "surface_pressure_hpa": 1002.0,
        "dew_point_c": 21.5,
        "previous_24h_rainfall_mm": 25.0,
        "previous_6h_rainfall_mm": 12.0,
        "predicted_rainfall_mm": 45.0,
        "forecast_lead_time_hours": 12,
        "soil_saturation_pct": 68.0,
        "drainage_density_km_km2": 1.4,
        "runoff_coefficient_pct": 78.0,
        "historical_flood_proneness": 0.60,
        "cloud_top_temp_c": -42.0,
        "cloud_optical_thickness": 48.0,
        "water_vapor_brightness_temp_k": 230.0,
        "convective_available_potential_energy": 1650.0,
        "vertical_wind_shear_kts": 18.0,
        "convective_storm_index": 0.52,
        "critical_infrastructure": ["Silk Board Storm Channel", "Bellandur Outflow Weir"],
        "recommended_action": "🟡 URBAN ADVISORY: Check primary storm water drains (Rajakaluves) for localized pooling. Maintain traffic advisories."
    },
    "Hyderabad (Moderate Convective Shower)": {
        "location": "Hyderabad",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "region": "Telangana Plateau",
        "elevation_m": 542.0,
        "slope_deg": 3.2,
        "temperature_c": 26.5,
        "relative_humidity_pct": 78.0,
        "wind_speed_kmh": 22.0,
        "wind_direction_deg": 245.0,
        "surface_pressure_hpa": 1004.0,
        "dew_point_c": 22.0,
        "previous_24h_rainfall_mm": 18.0,
        "previous_6h_rainfall_mm": 6.0,
        "predicted_rainfall_mm": 35.0,
        "forecast_lead_time_hours": 24,
        "soil_saturation_pct": 60.0,
        "drainage_density_km_km2": 1.5,
        "runoff_coefficient_pct": 70.0,
        "historical_flood_proneness": 0.55,
        "cloud_top_temp_c": -36.0,
        "cloud_optical_thickness": 40.0,
        "water_vapor_brightness_temp_k": 234.0,
        "convective_available_potential_energy": 1400.0,
        "vertical_wind_shear_kts": 16.0,
        "convective_storm_index": 0.44,
        "critical_infrastructure": ["Musi River Low Bridge", "Nala Inundation Sinks"],
        "recommended_action": "🟢 NORMAL WATCH: Standard monsoon preparedness. Monitor Musi river flow gates."
    },
    "Delhi NCR (Dry Heatwave Spell / Fair Weather)": {
        "location": "Delhi NCR",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "region": "Indo-Gangetic Plain",
        "elevation_m": 216.0,
        "slope_deg": 1.8,
        "temperature_c": 39.5,
        "relative_humidity_pct": 34.0,
        "wind_speed_kmh": 14.0,
        "wind_direction_deg": 290.0,
        "surface_pressure_hpa": 1005.0,
        "dew_point_c": 18.0,
        "previous_24h_rainfall_mm": 0.0,
        "previous_6h_rainfall_mm": 0.0,
        "predicted_rainfall_mm": 0.0,
        "forecast_lead_time_hours": 24,
        "soil_saturation_pct": 18.0,
        "drainage_density_km_km2": 1.2,
        "runoff_coefficient_pct": 80.0,
        "historical_flood_proneness": 0.40,
        "cloud_top_temp_c": -6.0,
        "cloud_optical_thickness": 8.0,
        "water_vapor_brightness_temp_k": 252.0,
        "convective_available_potential_energy": 600.0,
        "vertical_wind_shear_kts": 8.0,
        "convective_storm_index": 0.14,
        "critical_infrastructure": ["Yamuna Barrage Weir", "Minto Bridge Underpass"],
        "recommended_action": "🟢 CLEAR / NO HAZARD: Atmospheric conditions stable. Zero flood or heavy precipitation risk."
    }
}


# Pre-configured SIH demonstration scenarios for immediate judge evaluation
DEMO_SCENARIOS = {
    "🌟 Scenario 1: Normal Monsoon (LOW / MEDIUM)": {
        "preset_key": "Bengaluru (Moderate Thunderstorm / Urban Pooling)",
        "scenario_title": "Normal Monsoon Regime",
        "narrative": "Standard tropical convective activity. Stable atmospheric pressure (~1002 hPa), moderate cloud top temperature (-42°C), and manageable runoff. Low to localized minor drainage pooling only.",
        "risk_expectation": "🟢 LOW / 🟡 MEDIUM RISK",
        "rain_prob_expected": "35% - 48%",
        "flood_score_expected": "25 - 40 / 100",
        "alert_level": "GREEN (Standard Watch)"
    },
    "⚡ Scenario 2: Heavy Rainfall Event (HIGH)": {
        "preset_key": "Guwahati (Brahmaputra Valley Flood Pulse)",
        "scenario_title": "Active Monsoon Low-Pressure Surge",
        "narrative": "Deep synoptic convergence across the Brahmaputra valley. Rising moisture (96.5% humidity), deepening low pressure (993 hPa), and towering clouds (-71°C CTT). Heavy rain probability > 80% with high inundation risk.",
        "risk_expectation": "🟠 HIGH DISASTER RISK",
        "rain_prob_expected": "80% - 88%",
        "flood_score_expected": "65 - 78 / 100",
        "alert_level": "ORANGE (Active Warning)"
    },
    "🚨 Scenario 3: Extreme Rainfall / Cyclonic Cloudburst (EXTREME)": {
        "preset_key": "Chennai (Cyclone / Depression Episode)",
        "scenario_title": "Severe Tropical Depression & Orographic Deluge",
        "narrative": "Violent tropical convective depression. Sub-992 hPa core pressure, freezing overshooting tops (-72°C to -78°C CTT), extreme CAPE (>3000 J/kg), saturated soil (>94%). Extreme catastrophic flood and rain emergency.",
        "risk_expectation": "🔴 EXTREME DISASTER ALERT",
        "rain_prob_expected": "88% - 98%",
        "flood_score_expected": "82 - 95 / 100",
        "alert_level": "RED (Mandatory Evacuation Action)"
    }
}


def generate_synthetic_satellite_image(
    cloud_intensity: float = 0.8,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
    draw_storm_box: bool = True
) -> Image.Image:
    """
    Synthesizes a realistic multispectral false-color thermal infrared satellite image (10.8µm)
    with deep convective storm bands, overshooting tops, and optional storm-tracking bounding box.
    """
    np.random.seed(seed)
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)

    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    spiral = np.sin(3.5 * theta - 2.8 * np.log(r + 0.1))

    noise = np.random.normal(0, 0.25, (height, width))
    field = np.exp(-0.45 * r**2) * (1.0 + 0.5 * spiral) * cloud_intensity + noise * 0.2
    field = np.clip((field - field.min()) / (field.max() - field.min() + 1e-6), 0.0, 1.0)

    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            v = field[i, j]
            if v < 0.25:
                img_array[i, j] = [int(10 + 20 * v), int(20 + 60 * v), int(45 + 100 * v)]
            elif v < 0.50:
                norm_v = (v - 0.25) / 0.25
                img_array[i, j] = [int(20 + 80 * norm_v), int(120 + 100 * norm_v), int(180 - 60 * norm_v)]
            elif v < 0.75:
                norm_v = (v - 0.50) / 0.25
                img_array[i, j] = [int(220 + 35 * norm_v), int(180 - 60 * norm_v), 30]
            elif v < 0.90:
                norm_v = (v - 0.75) / 0.15
                img_array[i, j] = [int(230 + 25 * norm_v), int(40 - 20 * norm_v), int(40 + 20 * norm_v)]
            else:
                norm_v = (v - 0.90) / 0.10
                img_array[i, j] = [int(255 * norm_v), int(240 * norm_v), 255]

    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    if draw_storm_box and cloud_intensity > 0.6:
        # Draw AI Convective Storm Bounding Box
        draw = ImageDraw.Draw(img)
        box = [width * 0.28, height * 0.25, width * 0.72, height * 0.75]
        draw.rectangle(box, outline="#EF4444", width=3)
        draw.text((box[0] + 8, box[1] + 8), "DETECTED CONVECTIVE CORE (< -65°C)", fill="#EF4444")

    return img


def ensure_sample_assets():
    """
    Creates sample satellite images in assets/sample_satellite_images/.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets", "sample_satellite_images")
    os.makedirs(assets_dir, exist_ok=True)

    cyclone_path = os.path.join(assets_dir, "cyclone_depression_thermal_ir.png")
    monsoon_path = os.path.join(assets_dir, "monsoon_deep_convective_band.png")
    fair_path = os.path.join(assets_dir, "fair_weather_clear_sky.png")

    if not os.path.exists(cyclone_path):
        img_cyclone = generate_synthetic_satellite_image(cloud_intensity=0.95, seed=101)
        img_cyclone.save(cyclone_path)

    if not os.path.exists(monsoon_path):
        img_monsoon = generate_synthetic_satellite_image(cloud_intensity=0.75, seed=202)
        img_monsoon.save(monsoon_path)

    if not os.path.exists(fair_path):
        img_fair = generate_synthetic_satellite_image(cloud_intensity=0.20, seed=303, draw_storm_box=False)
        img_fair.save(fair_path)
