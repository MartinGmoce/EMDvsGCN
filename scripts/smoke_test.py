"""Fast health checks for the EMDvsGCN project.

This script intentionally avoids training models. It verifies paths, local data
assets, result schemas, and key Python dependencies so the project can be
checked before running expensive experiments.
"""

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import DATA_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


CORE_IMPORTS = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "scipy",
    "statsmodels",
    "seaborn",
]

TRAINING_IMPORTS = [
    "torch",
    "PyEMD",
]


def check_imports(module_names, required=True):
    ok = True
    for name in module_names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"[OK] import {name} {version}".rstrip())
        except Exception as exc:
            prefix = "[FAIL]" if required else "[WARN]"
            print(f"{prefix} import {name}: {exc}")
            ok = False if required else ok
    return ok


def check_file_group(label, path, pattern, required=True):
    files = sorted(Path(path).glob(pattern))
    if files:
        print(f"[OK] {label}: {len(files)} file(s)")
        return True
    prefix = "[FAIL]" if required else "[WARN]"
    print(f"{prefix} {label}: no files matched {Path(path) / pattern}")
    return not required


def check_csv_schema(label, path, required_columns):
    import pandas as pd

    path = Path(path)
    if not path.exists():
        print(f"[FAIL] {label}: missing {path}")
        return False
    df = pd.read_csv(path, nrows=5)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[FAIL] {label}: missing columns {missing}")
        return False
    print(f"[OK] {label}: columns present")
    return True


def main():
    print(f"Project root: {PROJECT_ROOT}")
    ok = True

    for directory in [DATA_DIR, RESULTS_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR]:
        if directory.exists():
            print(f"[OK] directory exists: {directory.relative_to(PROJECT_ROOT)}")
        else:
            print(f"[FAIL] missing directory: {directory.relative_to(PROJECT_ROOT)}")
            ok = False

    ok = check_imports(CORE_IMPORTS, required=True) and ok
    check_imports(TRAINING_IMPORTS, required=False)

    ok = check_file_group("processed datasets", PROCESSED_DATA_DIR, "Cleaned_*.csv") and ok
    ok = check_file_group("prediction results", PREDICTIONS_DIR, "*_predictions.csv") and ok

    processed_sample = next(PROCESSED_DATA_DIR.glob("Cleaned_*.csv"), None)
    if processed_sample:
        ok = check_csv_schema(
            f"processed sample {processed_sample.name}",
            processed_sample,
            ["trade_time", "close"],
        ) and ok

    prediction_sample = next(PREDICTIONS_DIR.glob("*_predictions.csv"), None)
    if prediction_sample:
        ok = check_csv_schema(
            f"prediction sample {prediction_sample.name}",
            prediction_sample,
            ["trade_time", "True_Price", "Predicted_Price"],
        ) and ok

    if ok:
        print("\nSmoke test passed. Analysis scripts can run with the current assets.")
        return 0

    print("\nSmoke test failed. Fix the failed items above before running experiments.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
