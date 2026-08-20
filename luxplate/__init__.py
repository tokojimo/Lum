"""Scientific core for LuxPlate Analyzer (independent from Streamlit)."""

from .experimental_design import biological_n
from .kinetics import calculate_auc, calculate_baseline_shifted_auc, calculate_peak, run_kinetics

__version__ = "0.1.0.dev0"
__all__ = [
    "biological_n", "calculate_auc", "calculate_baseline_shifted_auc", "calculate_peak",
    "run_kinetics",
]
