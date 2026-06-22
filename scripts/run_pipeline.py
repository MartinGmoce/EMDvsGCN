"""Run project stages from one stable entry point."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STAGES = {
    "clean": [
        "Code/Preprocess/data_clean.py",
    ],
    "ceemdan": [
        "Code/CEEMDAN/ceemdan_decompose.py",
        "Code/CEEMDAN/ceemdan_lstm.py",
    ],
    "graph": [
        "Code/GCN/prepare_local_features.py",
        "Code/GCN/build_vae_adj.py",
        "Code/GCN/build_base_adj.py",
    ],
    "baselines": [
        "Code/Baseline/arima_model.py",
        "Code/Baseline/train_dl_baseline.py",
        "Code/Baseline/run_informer.py",
    ],
    "gcn": [
        "Code/GCN/train_st_trader.py",
        "Code/GCN/train_gcn_informer.py",
    ],
    "analysis": [
        "Code/Analysis/calculate_total_metrics.py",
        "Code/Analysis/calculate_phase_delay.py",
    ],
    "plots": [
        "Code/Analysis/plot_predictions.py",
        "Code/Analysis/plot_predictions_2000.py",
        "Code/CEEMDAN/plot_decomposition.py",
    ],
}

FULL_ORDER = ["clean", "ceemdan", "graph", "baselines", "gcn", "analysis", "plots"]
REPRO_ORDER = ["analysis", "plots"]


def run_script(script):
    script_path = PROJECT_ROOT / script
    print(f"\n>>> {script}")
    subprocess.check_call([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run EMDvsGCN project pipeline stages.")
    parser.add_argument(
        "stage",
        choices=sorted(list(STAGES) + ["full", "reproduce"]),
        help="'reproduce' refreshes tables/plots from existing predictions; 'full' retrains everything.",
    )
    args = parser.parse_args()

    if args.stage == "full":
        stages = FULL_ORDER
    elif args.stage == "reproduce":
        stages = REPRO_ORDER
    else:
        stages = [args.stage]

    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        for script in STAGES[stage]:
            run_script(script)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
