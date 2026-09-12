"""
SpaceGuard AI - Realistic Sample & Historical Dataset Generator
Generates multi-year meteorological and satellite observation records
modeled on real-world tropical storm physics and monsoon dynamics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from satellite.features import SatelliteFeatureExtractor


LOCATIONS = [
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "elevation": 14.0, "slope": 1.2, "region": "Coromandel Coast"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "elevation": 10.0, "slope": 2.5, "region": "Konkan Coast"},
    {"name": "Wayanad (Kerala)", "lat": 11.6854, "lon": 76.1320, "elevation": 850.0, "slope": 28.0, "region": "Western Ghats"},
    {"name": "Guwahati (Assam)", "lat": 26.1445, "lon": 91.7362, "elevation": 55.0, "slope": 8.0, "region": "Brahmaputra Valley"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "elevation": 9.0, "slope": 0.8, "region": "Ganges Delta"},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "elevation": 920.0, "slope": 4.1, "region": "Deccan Plateau"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elevation": 542.0, "slope": 3.2, "region": "Telangana Plateau"},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "elevation": 22.0, "slope": 6.5, "region": "Northern Circars"},
    {"name": "Puri (Odisha)", "lat": 19.8135, "lon": 85.8312, "elevation": 12.0, "slope": 1.5, "region": "East Coast Cyclonic Zone"},
    {"name": "Delhi NCR", "lat": 28.6139, "lon": 77.2090, "elevation": 216.0, "slope": 1.8, "region": "Indo-Gangetic Plain"},
]


def generate_dataset(num_records_per_location: int = 550, seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes meteorologically consistent atmospheric & satellite measurements.
    """
    np.random.seed(seed)
    records = []
    start_date = datetime(2021, 1, 1)

    for loc in LOCATIONS:
        for i in range(num_records_per_location):
            dt = start_date + timedelta(days=i * 2.5)
            month = dt.month

            # Monsoon seasonality weight (June-Sept Southwest monsoon, Oct-Dec Northeast monsoon for Coromandel)
            is_sw_monsoon = month in [6, 7, 8, 9]
            is_ne_monsoon = month in [10, 11, 12] and loc["name"] in ["Chennai", "Visakhapatnam", "Puri (Odisha)"]
            monsoon_active = is_sw_monsoon or is_ne_monsoon

            # Determine whether this day has a synoptic depression / tropical convective disturbance
            storm_trigger_prob = 0.35 if monsoon_active else 0.08
            is_storm_system = np.random.random() < storm_trigger_prob

            if is_storm_system:
                # Deep tropical depression / cyclonic disturbance
                pressure = np.random.normal(996.0, 4.5)  # low pressure core
                humidity = np.random.uniform(84.0, 99.0)
                temp = np.random.normal(26.5, 2.0)
                dew_point = temp - np.random.uniform(0.5, 2.5)
                wind_speed = np.random.normal(48.0, 15.0)
                wind_dir = (np.random.normal(220.0, 40.0)) % 360  # south-westerly / cyclonic
                prev_rain = np.random.exponential(45.0)
                soil_sat = np.random.uniform(65.0, 98.0)

                # Satellite channels during severe convective development
                # Cloud Top Temperatures drop dramatically during deep cumulonimbus (-50 to -82 C)
                cloud_top_temp = np.random.normal(-62.0, 8.0)
                cloud_optical_thickness = np.random.uniform(45.0, 110.0)
                wv_bt = np.random.normal(222.0, 6.0)
                cape = np.random.normal(2600.0, 600.0)
                wind_shear = np.random.uniform(25.0, 55.0)

                # Precipitation formula with orographic and convective enhancement
                rain_base = np.random.gamma(shape=3.5, scale=24.0)
                # Orographic lift multiplier
                if loc["elevation"] > 500 or loc["slope"] > 10.0:
                    rain_base *= 1.35

                rainfall_mm = rain_base
            else:
                # Fair weather, dry spell, or marginal showers
                pressure = np.random.normal(1013.0, 3.5)
                humidity = np.random.uniform(35.0, 78.0)
                temp = np.random.normal(32.0, 4.0)
                dew_point = temp - np.random.uniform(4.0, 12.0)
                wind_speed = np.random.normal(16.0, 6.0)
                wind_dir = np.random.uniform(0.0, 360.0)
                prev_rain = np.random.exponential(4.0) if monsoon_active else 0.0
                soil_sat = np.random.uniform(15.0, 55.0)

                # Satellite channels in normal regime
                cloud_top_temp = np.random.normal(-15.0, 15.0)
                cloud_optical_thickness = np.random.uniform(2.0, 30.0)
                wv_bt = np.random.normal(245.0, 8.0)
                cape = np.random.normal(850.0, 350.0)
                wind_shear = np.random.uniform(5.0, 25.0)

                rainfall_mm = np.random.exponential(3.0) if np.random.random() < 0.25 else 0.0

            # Compute Convective Storm Index via SatelliteFeatureExtractor
            csi = SatelliteFeatureExtractor.compute_convective_signature(
                cloud_top_temp, wv_bt, cape, cloud_optical_thickness
            )

            # Heavy rainfall definition (IMD Standard: >= 64.5 mm / 24 hr)
            heavy_rainfall = 1 if rainfall_mm >= 64.5 else 0

            # Rainfall intensity categorization
            if rainfall_mm < 15.6:
                intensity = "NONE_TO_LIGHT"
            elif rainfall_mm < 64.5:
                intensity = "MODERATE"
            elif rainfall_mm < 115.5:
                intensity = "HEAVY"
            elif rainfall_mm < 204.4:
                intensity = "VERY_HEAVY"
            else:
                intensity = "EXTREMELY_HEAVY"

            # Create entry
            record = {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "location": loc["name"],
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "elevation_m": loc["elevation"],
                "slope_deg": loc["slope"],
                "region": loc["region"],
                # Weather variables
                "temperature_c": round(temp, 2),
                "relative_humidity_pct": round(humidity, 1),
                "wind_speed_kmh": round(max(0.0, wind_speed), 1),
                "wind_direction_deg": round(wind_dir, 1),
                "surface_pressure_hpa": round(pressure, 1),
                "dew_point_c": round(dew_point, 2),
                "previous_24h_rainfall_mm": round(prev_rain, 1),
                "soil_saturation_pct": round(soil_sat, 1),
                # Satellite spectral variables
                "cloud_top_temp_c": round(cloud_top_temp, 1),
                "cloud_optical_thickness": round(cloud_optical_thickness, 1),
                "water_vapor_brightness_temp_k": round(wv_bt, 1),
                "convective_available_potential_energy": round(max(0.0, cape), 1),
                "vertical_wind_shear_kts": round(max(0.0, wind_shear), 1),
                "convective_storm_index": csi,
                # Ground truth targets
                "recorded_rainfall_mm": round(rainfall_mm, 1),
                "rainfall_intensity": intensity,
                "heavy_rainfall_event": heavy_rainfall,
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Inject realistic missing values (~0.5%) to test cleaning pipeline
    mask_null_humidity = np.random.random(len(df)) < 0.005
    df.loc[mask_null_humidity, "relative_humidity_pct"] = np.nan

    mask_null_pressure = np.random.random(len(df)) < 0.005
    df.loc[mask_null_pressure, "surface_pressure_hpa"] = np.nan

    mask_null_satellite = np.random.random(len(df)) < 0.005
    df.loc[mask_null_satellite, "cloud_top_temp_c"] = np.nan

    return df


def main():
    print("🛰️ Generating SpaceGuard AI Meteorological & Satellite Dataset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hist_dir = os.path.join(base_dir, "data", "historical")
    sat_dir = os.path.join(base_dir, "data", "satellite")
    proc_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(sat_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    df = generate_dataset(num_records_per_location=550)
    print(f"Total simulated multi-year records: {len(df)}")
    print(f"Heavy Rainfall Events (Target=1): {df['heavy_rainfall_event'].sum()} ({df['heavy_rainfall_event'].mean():.1%})")

    # Split into historical weather records and satellite observations
    weather_cols = [
        "timestamp", "location", "latitude", "longitude", "region",
        "temperature_c", "relative_humidity_pct", "wind_speed_kmh",
        "wind_direction_deg", "surface_pressure_hpa", "dew_point_c",
        "previous_24h_rainfall_mm", "elevation_m", "slope_deg",
        "soil_saturation_pct", "recorded_rainfall_mm", "rainfall_intensity",
        "heavy_rainfall_event"
    ]
    sat_cols = [
        "timestamp", "location",
        "cloud_top_temp_c", "cloud_optical_thickness",
        "water_vapor_brightness_temp_k", "convective_available_potential_energy",
        "vertical_wind_shear_kts", "convective_storm_index"
    ]

    df_weather = df[weather_cols]
    df_sat = df[sat_cols]

    hist_file = os.path.join(hist_dir, "weather_records.csv")
    sat_file = os.path.join(sat_dir, "satellite_features.csv")
    merged_file = os.path.join(proc_dir, "merged_dataset.csv")

    df_weather.to_csv(hist_file, index=False)
    df_sat.to_csv(sat_file, index=False)
    df.to_csv(merged_file, index=False)

    print(f"Saved: {hist_file}")
    print(f"Saved: {sat_file}")
    print(f"Saved: {merged_file}")
    print("Dataset generation completed successfully.")


if __name__ == "__main__":
    main()
