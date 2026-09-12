"""
SpaceGuard AI - Machine Learning Pipeline Module
Encapsulates data ingestion, training, cross-validation, and evaluation.
"""

import os
import sys
from typing import Dict, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.train_model import train_rainfall_model
from model.evaluate_model import ModelEvaluator
from model.predict import RainfallPredictor


class MLPipeline:
    """
    Unified ML pipeline interface for SpaceGuard AI.
    """

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.predictor = RainfallPredictor()

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Executes stratified train-test split, fits ensemble models,
        and saves genuine evaluation metrics to metrics.json.
        """
        model, metrics = train_rainfall_model(test_size=test_size, random_state=random_state)
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """
        Loads saved test-set metrics.
        """
        return self.evaluator.load_saved_metrics()
