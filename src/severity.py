"""
severity.py
===========
Defines the severity estimation logic for road waterlogging.

IMPORTANT METHODOLOGICAL NOTE:
The FloodNet dataset does NOT provide Low/Moderate/Severe ground-truth severity labels.
Therefore, all severity levels are explicitly designated as:
"Project-defined rule-based severity tiers derived from the predicted flooded-road area percentage."

Severity tiers are computed from:
    Flooded Road Ratio = (Road-flooded pixels) / (Road-flooded + Road-non-flooded pixels) * 100%

Project-Defined Thresholds:
---------------------------
- <= 0.5%         = NONE     (Surface completely dry / normal conditions)
- >0.5% to 15.0%  = LOW      (Minor surface pooling, passable with caution)
- >15.0% to 40.0% = MODERATE (Substantial water accumulation, lane disruption, aquaplaning risk)
- >40.0%          = SEVERE   (Widespread roadway submergence, road impassable)
"""

from typing import Dict, Any, Tuple


SEVERITY_TYPE_LABEL = (
    "Project-defined rule-based severity tiers derived from the "
    "predicted flooded-road area percentage"
)


class SeverityEstimator:
    """
    Computes road waterlogging severity tiers using transparent,
    project-defined rule-based thresholds derived from the predicted flooded-road area percentage.

    NOTE: The FloodNet dataset does NOT provide ground-truth severity labels.
    All tier assignments are project-defined and must be described as such in all
    reports, UI output, and academic documentation.
    """

    DEFAULT_THRESHOLDS = {
        "NONE":     (0.0,  0.5),
        "LOW":      (0.5,  15.0),
        "MODERATE": (15.0, 40.0),
        "SEVERE":   (40.0, 100.0)
    }

    def __init__(self, thresholds: Dict[str, Tuple[float, float]] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def estimate(
        self,
        affected_area_pct: float = None,
        waterlogging_detected: bool = True,
        road_pixels: int = 1,
    ) -> Dict[str, Any]:
        """
        Estimate severity tier from flooded-road area percentage.

        Args:
            affected_area_pct: Float [0.0 - 100.0] representing
                               (Road-flooded pixels) / (Total road pixels) * 100%.
            waterlogging_detected: Boolean indicating if any waterlogging was detected.

        Returns:
            Dictionary containing:
            - severity_level: 'NONE', 'LOW', 'MODERATE', or 'SEVERE'
            - severity_type: Disclosure string confirming project-defined tier basis
            - is_project_defined: Always True for FloodNet (no ground-truth severity labels)
            - explanation: Human-readable rationale
            - color: Hex color code for UI visualization
            - affected_area_pct: Clamped float value
        """
        if affected_area_pct is None or road_pixels <= 0:
            return {
                "severity_level": "UNAVAILABLE",
                "severity_type": SEVERITY_TYPE_LABEL,
                "is_project_defined": True,
                "explanation": "No road pixels were predicted, so a road-area percentage cannot be calculated.",
                "color": "#64748B",
                "affected_area_pct": None,
            }

        area = max(0.0, min(100.0, float(affected_area_pct)))

        if not waterlogging_detected or area <= self.thresholds["NONE"][1]:
            return {
                "severity_level": "NONE",
                "severity_type": SEVERITY_TYPE_LABEL,
                "is_project_defined": True,
                "explanation": (
                    f"Project-defined flooded-road coverage is {area:.1f}%; "
                    "this is not a depth or safety assessment."
                ),
                "color": "#10B981",
                "affected_area_pct": round(area, 2)
            }

        if area <= self.thresholds["LOW"][1]:
            level = "LOW"
            explanation = (
                f"Minor road waterlogging ({area:.1f}% of road surface). "
                "This project-defined tier is not a depth or safety assessment."
            )
            color = "#FBBF24"
        elif area <= self.thresholds["MODERATE"][1]:
            level = "MODERATE"
            explanation = (
                f"Moderate road waterlogging ({area:.1f}% of road surface). "
                "This project-defined tier is not a depth or safety assessment."
            )
            color = "#F97316"
        else:
            level = "SEVERE"
            explanation = (
                f"Severe road waterlogging ({area:.1f}% of road surface). "
                "This project-defined tier is not a depth or safety assessment."
            )
            color = "#EF4444"

        return {
            "severity_level": level,
            "severity_type": SEVERITY_TYPE_LABEL,
            "is_project_defined": True,
            "explanation": explanation,
            "color": color,
            "affected_area_pct": round(area, 2)
        }

    @staticmethod
    def _get_color(level: str) -> str:
        colors = {
            "NONE": "#10B981",
            "LOW": "#FBBF24",
            "MODERATE": "#F97316",
            "SEVERE": "#EF4444"
        }
        return colors.get(level, "#6B7280")
