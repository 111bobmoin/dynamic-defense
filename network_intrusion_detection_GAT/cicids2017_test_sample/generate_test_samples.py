from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cicids2017_test_sample import (
    DEFAULT_DATASET_DIR,
    DEFAULT_LARGE_NODE_COUNTS,
    SUPPORTED_SCENARIOS,
    generate_test_samples,
    get_scenario_display_name,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CICIDS2017-style workflow test samples from a MachineLearningCVE dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing MachineLearningCVE CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory for generated scenarios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional generation seed. Different seeds produce different test samples.",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=SUPPORTED_SCENARIOS,
        help="Scenario to generate. Repeat the flag to generate multiple scenarios.",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Generate larger samples with 50-100 nodes per scenario using built-in defaults.",
    )
    parser.add_argument("--no-anomaly-nodes", type=int, default=None)
    parser.add_argument("--single-anomaly-nodes", type=int, default=None)
    parser.add_argument("--multi-anomaly-nodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = args.scenarios or list(SUPPORTED_SCENARIOS)
    node_count_overrides: dict[str, int] = {}
    if args.large:
        node_count_overrides.update(DEFAULT_LARGE_NODE_COUNTS)
    if args.no_anomaly_nodes is not None:
        node_count_overrides["no_anomaly"] = args.no_anomaly_nodes
    if args.single_anomaly_nodes is not None:
        node_count_overrides["single_anomaly"] = args.single_anomaly_nodes
    if args.multi_anomaly_nodes is not None:
        node_count_overrides["multi_anomaly"] = args.multi_anomaly_nodes
    suite_manifest = generate_test_samples(
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        scenarios=scenarios,
        seed=args.seed,
        node_count_overrides=node_count_overrides or None,
    )
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Generation seed: {args.seed}")
    print(f"Scenarios generated: {len(suite_manifest['scenarios'])}")
    for scenario in suite_manifest["scenarios"]:
        print(
            f"- {scenario['scenario']} ({get_scenario_display_name(scenario['scenario'])}): "
            f"flows={scenario['total_flows']}, "
            f"nodes={scenario['total_nodes']}, sample={scenario['sample_csv']}, "
            f"seed={scenario['generation_seed']}"
        )


if __name__ == "__main__":
    main()
