"""
SpaceGuard AI - Satellite Data Preprocessing Module
Normalizes raw spectral radiances, filters sensor noise,
and applies meteorological physical plausibility bounds.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class SatellitePreprocessor:
    """
    Cleans, validates, and normalizes satellite-derived observation features.
    """

    # Physical valid bounds for satellite meteorological sensors
    SENSOR_BOUNDS = {
        "cloud_top_temp_c": (-90.0, 50.0),       # Deep convective cores can reach -85C
        "cloud_optical_thickness": (0.0, 150.0), # Optical depth
        "water_vapor_brightness_temp_k": (180.0, 320.0), # Kelvin
        "convective_available_potential_energy": (0.0, 6500.0), # J/kg
        "vertical_wind_shear_kts": (0.0, 100.0),  # Knots
    }

    def __init__(self):
        pass

    def clean_and_bound_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clips extreme sensor glitches to physical meteorological boundaries
        and fills missing values with robust median estimates.
        """
        df_clean = df.copy()

        for col, (min_val, max_val) in self.SENSOR_BOUNDS.items():
            if col in df_clean.columns:
                # Replace invalid values or missing values with median
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                # Clip to physical bounds
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

        return df_clean

    def calculate_cloud_mask(self, cloud_top_temp_c: float, optical_depth: float) -> str:
        """
        Classifies cloud regime based on thermal IR brightness temperature and optical depth.
        """
        if optical_depth < 5.0 or cloud_top_temp_c > 15.0:
            return "Clear / Thin Cirrus"
        elif cloud_top_temp_c > 0.0:
            return "Low Stratus / Cumulus"
        elif cloud_top_temp_c > -40.0:
            return "Altocumulus / Moderate Convection"
        else:
            return "Deep Convective Cumulonimbus (Towering Storm)"
