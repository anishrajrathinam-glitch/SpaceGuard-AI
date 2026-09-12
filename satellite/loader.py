"""
SpaceGuard AI - Satellite Data Loader Module
Handles loading of satellite spectral channels, atmospheric soundings,
and convective cloud metadata.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class SatelliteDataLoader:
    """
    Loads and standardizes satellite observational features and metadata.
    Designed to work with demo/sample tabular spectral bands as well as
    extensible for future API feeds (e.g., INSAT-3D/3DR, GOES, Meteosat).
    """

    EXPECTED_CHANNELS = [
        "cloud_top_temp_c",
        "cloud_optical_thickness",
        "water_vapor_brightness_temp_k",
        "convective_available_potential_energy",
        "vertical_wind_shear_kts",
    ]

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "satellite",
        )

    def load_satellite_dataset(self, filename: str = "satellite_features.csv") -> pd.DataFrame:
        """
        Loads the satellite observations CSV dataset.
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Satellite dataset not found at {file_path}. "
                "Please run `python generate_sample_data.py` to create the initial dataset."
            )
        df = pd.read_csv(file_path)
        return df

    def get_satellite_metadata(self) -> Dict[str, Any]:
        """
        Returns observational sensor metadata and channel configuration.
        """
        return {
            "source": "SpaceGuard Synthetic & Historical Satellite Band Fusion",
            "reference_sensor": "INSAT-3D/3DR & Sentinel-3 Multispectral Proxy",
            "spectral_bands": [
                {"band": "IR_10.8um", "name": "Thermal Infrared", "target": "Cloud Top Temperature"},
                {"band": "WV_6.7um", "name": "Water Vapor", "target": "Mid-Tropospheric Moisture"},
                {"band": "VIS_0.6um", "name": "Visible", "target": "Cloud Optical Thickness"},
                {"band": "SOUNDER_CAPE", "name": "Atmospheric Sounder", "target": "CAPE Instability"},
            ],
            "spatial_resolution": "4km x 4km at Nadir",
            "temporal_cadence": "15-minute intervals",
            "mode": "DEMO MODE — Using sample/historical satellite dataset",
        }
