"""Shared project paths and small runtime helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RAW_DATA_DIR = DATA_DIR / "Raw"
PROCESSED_DATA_DIR = DATA_DIR / "Processed"
GCN_DATA_DIR = DATA_DIR / "GCN"
CEEMDAN_DATA_DIR = DATA_DIR / "CEEMDAN_Decomposed"

CHECKPOINT_DIR = RESULTS_DIR / "Checkpoints"
PREDICTIONS_DIR = RESULTS_DIR / "Predictions"
METRICS_DIR = RESULTS_DIR / "Metrics"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "AnalysisResults"
FORECAST_PLOTS_DIR = RESULTS_DIR / "ForecastPlots"
PREDICTIONS_2000_DIR = RESULTS_DIR / "PredictionsPlot2000"
DECOMPOSITION_PLOTS_DIR = RESULTS_DIR / "Decompositions"
VAE_FEATURES_DIR = RESULTS_DIR / "VAE_Features"


def ensure_directories(*paths):
    """Create project output directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def configure_matplotlib_backend():
    """Use a headless backend so plotting scripts work on servers."""
    import matplotlib

    matplotlib.use("Agg")
