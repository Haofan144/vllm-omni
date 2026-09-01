#!/usr/bin/env python3
"""Run H1-H7 in order and print a one-line verdict table.

    python benchmarks/hbm_admission/hypothesis_validation/run_all.py [--skip-h1]

H1 needs a real CUDA device; --skip-h1 runs only the control-plane experiments
(H2-H7), which are deterministic and GPU-free.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SCRIPTS = [
    ("H1", "H1_measurement_accuracy.py"),
    ("H2", "H2_shared_device_identity.py"),
    ("H3", "H3_high_watermark_multiplicative_decrease.py"),
    ("H4", "H4_critical_watermark_halts_admission.py"),
    ("H5", "H5_running_requests_not_evicted.py"),
    ("H6", "H6_recovery_after_pressure_release.py"),
    ("H7", "H7_fail_closed_on_faults.py"),
    ("H10", "H10_resource_aware_admission.py"),
    ("H11", "H11_ar_profile_prediction.py"),
    ("H12", "H12_ewma_calibration_effectiveness.py"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-h1", action="store_true")
    ap.add_argument("--h1-args", default="--steps 8 --step-mib 1024 --idle-samples 6 --plateau-samples 3 --sample-interval-ms 300")
    args = ap.parse_args()

    rows = []
    for hyp, script in SCRIPTS:
        if hyp == "H1" and args.skip_h1:
            rows.append((hyp, "SKIPPED", ""))
            continue
        cmd = [sys.executable, str(HERE / script)]
        if hyp == "H1":
            cmd += args.h1_args.split()
        print(f"\n=== running {hyp} ({script}) ===")
        proc = subprocess.run(cmd, cwd=HERE.parents[2])
        result_path = RESULTS / f"{hyp}_result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            verdict = data["verdict"]
            n_fail = sum(1 for c in data["checks"] if not c["passed"])
            rows.append((hyp, verdict, f"{n_fail} failing check(s)" if n_fail else "all checks pass"))
        else:
            rows.append((hyp, f"NO RESULT (exit {proc.returncode})", ""))

    print("\n\n================ H1-H7, H10-H12 VERDICT TABLE ================")
    for hyp, verdict, note in rows:
        print(f"  {hyp}: {verdict:<22} {note}")
    print("====================================================")


if __name__ == "__main__":
    main()
