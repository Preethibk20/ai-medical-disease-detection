import numpy as np
from typing import Dict, Any, Optional


class DemoDiseaseDetector:
    """Heuristic, fast, pre-trained-like demo detector that works without model files."""

    diseases = ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']

    def __init__(self):
        pass

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_all_diseases_from_raw(
        self,
        image_features: Optional[Dict[str, float]],
        lab: Optional[Dict[str, float]],
        demo: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Produce stable probabilities from raw inputs without trained models."""
        lab = lab or {}
        demo = demo or {}
        image_features = image_features or {}

        # Basic signals
        bmi = demo.get('bmi') or self._compute_bmi(demo)
        sbp = float(demo.get('systolic_bp') or 120)
        dbp = float(demo.get('diastolic_bp') or 80)
        hr = float(demo.get('heart_rate') or 72)
        temp = float(demo.get('temperature') or 37.0)

        glucose = float(lab.get('glucose') or 100)
        hba1c = float(lab.get('hba1c') or 5.5)
        ldl = float(lab.get('cholesterol_ldl') or 110)
        hdl = float(lab.get('cholesterol_hdl') or 45)
        triglycerides = float(lab.get('triglycerides') or 150)
        creatinine = float(lab.get('creatinine') or 1.0)
        alt = float(lab.get('alt') or 25)
        ast = float(lab.get('ast') or 25)

        # Image-derived simple features
        brightness = float(image_features.get('mean_intensity') or 0.5)
        edges = float(image_features.get('edge_density') or 0.1)

        # Heuristic logits
        diabetes_logit = 0.02 * (glucose - 100) + 0.8 * (hba1c - 5.5) + 0.03 * (bmi - 25)
        cardio_logit = 0.015 * (sbp - 120) + 0.02 * (dbp - 80) + 0.02 * (ldl - 100) - 0.02 * (hdl - 50) + 0.01 * (triglycerides - 150)
        cancer_logit = 0.5 * (edges - 0.1) + 0.4 * (brightness - 0.5) + 0.01 * (age_or(demo) - 50)
        kidney_logit = 0.6 * (creatinine - 1.0) + 0.01 * (sbp - 120)
        liver_logit = 0.015 * (alt - 25) + 0.015 * (ast - 25) + 0.02 * (bmi - 25)

        return {
            'diabetes': float(self._sigmoid(diabetes_logit)),
            'cardiovascular': float(self._sigmoid(cardio_logit)),
            'cancer': float(self._sigmoid(cancer_logit)),
            'kidney_disease': float(self._sigmoid(kidney_logit)),
            'liver_disease': float(self._sigmoid(liver_logit)),
        }

    def _compute_bmi(self, demo: Dict[str, Any]) -> float:
        try:
            w = float(demo.get('weight') or 70.0)
            h_cm = float(demo.get('height') or 170.0)
            return w / ((h_cm / 100.0) ** 2)
        except Exception:
            return 24.0


def age_or(demo: Dict[str, Any]) -> float:
    try:
        return float(demo.get('age') or 50)
    except Exception:
        return 50.0


