"""
SpaceGuard AI - Backend Services Module
Coordinates multi-sensor data fusion, prediction orchestration,
flood inundation services, and early warning dispatch.
"""

import os
import sys
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.predict import RainfallPredictor
from flood.inundation import FloodInundationEngine
from xai.explain import ModelExplainer
from alerts.email_alert import AlertDispatcher
from utils.helpers import PRESET_LOCATIONS, DEMO_SCENARIOS


class DisasterIntelligenceService:
    """
    Unified disaster intelligence coordination service.
    """

    def __init__(self):
        self.predictor = RainfallPredictor()
        self.flood_engine = FloodInundationEngine()
        self.explainer = ModelExplainer()
        self.dispatcher = AlertDispatcher()

    def run_full_disaster_assessment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete multi-hazard pipeline:
        1. ML heavy rainfall prediction
        2. Hydrological flood & inundation risk estimation
        3. Explainable AI factor decomposition
        """
        prediction_result = self.predictor.predict(input_data)
        
        # Compute local SHAP explanation
        xai_result = self.explainer.explain_prediction(
            X_scaled=prediction_result["raw_scaled_array"],
            feature_values_dict=prediction_result["feature_values"],
            top_n=6
        )

        return {
            "prediction": prediction_result,
            "xai": xai_result,
            "disclaimer": "DEMO MODE — Sample/Historical Data"
        }

    def load_scenario(self, scenario_key: str) -> Dict[str, Any]:
        """
        Loads pre-configured SIH judging demo scenario.
        """
        scenario = DEMO_SCENARIOS.get(scenario_key)
        if not scenario:
            scenario = list(DEMO_SCENARIOS.values())[0]
        preset_key = scenario["preset_key"]
        preset_data = PRESET_LOCATIONS[preset_key]
        return self.run_full_disaster_assessment(preset_data)
