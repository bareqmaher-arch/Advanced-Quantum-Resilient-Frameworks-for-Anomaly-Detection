"""
Q-ZAP Master Experiment Runner
================================
Runs all three experiments from the paper and saves results to results/.

Usage
-----
  # Quick demo (no dataset needed — uses synthetic / canonical data):
  python run_all_experiments.py --demo

  # Full run with real CIC-IDS2017 dataset:
  python run_all_experiments.py --data-dir ./data/cic_ids2017 --epochs 100

  # Skip expensive HAE training (useful for fast re-runs):
  python run_all_experiments.py --demo --skip-table1

Output
------
  results/
    table1.json       — Experiment 1: Anomaly detection (Table 1)
    table2.json       — Experiment 2: PQC overhead (Table 2)
    case_study.json   — Experiment 3: Cross-tenant attack (Section 5.5)
    summary.json      — All results in one file
"""

import argparse
import json
import os
import sys
import io
import time
import logging

# Ensure UTF-8 on Windows consoles that default to cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiments", "results")


def main():
    parser = argparse.ArgumentParser(
        description="Q-ZAP: Run all paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic / canonical data (no dataset download required)"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to CIC-IDS2017 CSV folder (download from unb.ca/cic/datasets/ids-2017.html)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs for neural network models (default: 100)"
    )
    parser.add_argument(
        "--handshakes", type=int, default=1000,
        help="Number of TLS handshakes to simulate for Table 2 (default: 1000)"
    )
    parser.add_argument("--skip-table1",    action="store_true", help="Skip Experiment 1")
    parser.add_argument("--skip-table2",    action="store_true", help="Skip Experiment 2")
    parser.add_argument("--skip-casestudy", action="store_true", help="Skip Experiment 3")
    args = parser.parse_args()

    # If neither --demo nor --data-dir provided, default to demo
    if not args.demo and not args.data_dir:
        logger.info("No --data-dir provided. Running in --demo mode.")
        args.demo = True

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {}
    total_start = time.time()

    # -------------------------------------------------------------------
    # Experiment 1 — Table 1
    # -------------------------------------------------------------------
    if not args.skip_table1:
        logger.info("\n" + "="*60)
        logger.info("  RUNNING EXPERIMENT 1: Anomaly Detection (Table 1)")
        logger.info("="*60)
        from experiments.benchmarks.anomaly_detection_benchmark import run_benchmark as run_e1
        t1 = run_e1(
            data_dir=args.data_dir,
            epochs=args.epochs,
            output=os.path.join(RESULTS_DIR, "table1.json"),
            demo=args.demo,
        )
        summary["table1"] = t1
    else:
        logger.info("Skipping Experiment 1 (--skip-table1)")

    # -------------------------------------------------------------------
    # Experiment 2 — Table 2
    # -------------------------------------------------------------------
    if not args.skip_table2:
        logger.info("\n" + "="*60)
        logger.info("  RUNNING EXPERIMENT 2: PQC Performance (Table 2)")
        logger.info("="*60)
        from experiments.benchmarks.pqc_performance_benchmark import run_benchmark as run_e2
        t2 = run_e2(
            n_handshakes=args.handshakes,
            output=os.path.join(RESULTS_DIR, "table2.json"),
        )
        summary["table2"] = t2
    else:
        logger.info("Skipping Experiment 2 (--skip-table2)")

    # -------------------------------------------------------------------
    # Experiment 3 — Case Study
    # -------------------------------------------------------------------
    if not args.skip_casestudy:
        logger.info("\n" + "="*60)
        logger.info("  RUNNING EXPERIMENT 3: End-to-End Case Study (Section 5.5)")
        logger.info("="*60)
        from experiments.benchmarks.end_to_end_case_study import run_case_study as run_e3
        cs = run_e3(
            demo=args.demo,
            data_dir=args.data_dir,
            output=os.path.join(RESULTS_DIR, "case_study.json"),
        )
        summary["case_study"] = cs
    else:
        logger.info("Skipping Experiment 3 (--skip-casestudy)")

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    elapsed = time.time() - total_start
    summary["total_elapsed_seconds"] = round(elapsed, 1)

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*60)

    if "table1" in summary:
        hae = next(r for r in summary["table1"]["results"] if "HAE" in r["model"])
        ae  = next(r for r in summary["table1"]["results"] if "Classical" in r["model"])
        print(f"  HAE F1-Score : {hae['f1_score']:.2f}  (paper: 0.91)")
        print(f"  HAE AUC      : {hae['auc']:.2f}  (paper: 0.96)")
        imp = summary["table1"].get("hae_improvement_over_ae_pct")
        if imp is not None:
            print(f"  Improvement  : {imp:+.1f}%  (paper: +9.6%)")
        else:
            ae_f1 = ae['f1_score']; hae_f1 = hae['f1_score']
            if ae_f1 > 0:
                print(f"  Improvement  : {(hae_f1-ae_f1)/ae_f1*100:+.1f}%  (paper: +9.6%)")

    if "table2" in summary:
        ov = summary["table2"]["overhead"]
        print(f"  PQC Handshake overhead : {ov['handshake_pct']:+.1f}%  (paper: +73.9%)")
        print(f"  PQC Throughput impact  : {ov['throughput_pct']:+.1f}%  (paper: -2.8%)")

    if "case_study" in summary:
        print(f"  Cross-tenant attack blocked : {summary['case_study']['exfiltration_prevented']}")

    print(f"\n  Results saved in:  {RESULTS_DIR}/")
    print(f"  Total time       : {elapsed:.1f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
