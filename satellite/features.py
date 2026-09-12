"""
SpaceGuard AI - Satellite Feature Engineering Module
Calculates physical convective indicators, instability indices,
and thermal temperature anomalies from raw satellite spectral data.
"""

import pandas as pd
import numpy as np


class SatelliteFeatureExtractor:
    """
    Extracts specialized severe weather indicators from satellite spectral channels.
    """

    @staticmethod
    def compute_convective_signature(
        cloud_top_temp_c: float,
        water_vapor_bt_k: float,
        cape_j_kg: float,
        optical_depth: float
    ) -> float:
        """
        Computes a normalized Convective Storm Index (CSI) between 0.0 and 1.0.
        Deep convection is marked by:
        - Very cold cloud top temperatures (< -50 C)
        - High water vapor brightness temperature depression (approaching cloud top temp)
        - High CAPE (> 2000 J/kg)
        - High optical thickness (> 50)
        """
        # Score 1: Cloud Top Temperature anomaly (0 to 1)
        # -80C -> 1.0, 0C -> 0.0
        c_temp_score = np.clip((-cloud_top_temp_c) / 80.0, 0.0, 1.0)

        # Score 2: CAPE Instability (0 to 1)
        # 3500 J/kg -> 1.0, 0 -> 0.0
        cape_score = np.clip(cape_j_kg / 3500.0, 0.0, 1.0)

        # Score 3: Optical Thickness
        tau_score = np.clip(optical_depth / 80.0, 0.0, 1.0)

        # Convective Index is weighted blend
        csi = (0.45 * c_temp_score) + (0.35 * cape_score) + (0.20 * tau_score)
        return float(np.round(csi, 4))

    @staticmethod
    def add_satellite_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a dataframe containing satellite channels with derived meteorology features.
        """
        df_feat = df.copy()

        # Severe convection indicator: True if Cloud Top Temp < -52C (tropopause penetration)
        if "cloud_top_temp_c" in df_feat.columns:
            df_feat["severe_convective_core"] = (df_feat["cloud_top_temp_c"] < -52.0).astype(int)

        # Convective storm index
        if all(col in df_feat.columns for col in ["cloud_top_temp_c", "water_vapor_brightness_temp_k", "convective_available_potential_energy", "cloud_optical_thickness"]):
            df_feat["convective_storm_index"] = df_feat.apply(
                lambda row: SatelliteFeatureExtractor.compute_convective_signature(
                    row["cloud_top_temp_c"],
                    row["water_vapor_brightness_temp_k"],
                    row["convective_available_potential_energy"],
                    row["cloud_optical_thickness"]
                ),
                axis=1
            )

        return df_feat
