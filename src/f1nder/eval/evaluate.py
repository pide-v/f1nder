# src/f1nder/eval/evaluate.py

from pathlib import Path
from typing import Dict
from argparse import ArgumentParser

import json
import pyterrier as pt

from f1nder.eval.measures import get_measures


def evaluate_run(
    run_file: Path,
    qrels_file: Path,
) -> Dict[str, float]:
    """
    Evaluate a single TREC run file against qrels.

    Parameters
    ----------
    run_file : Path
        Path to a TREC run file (e.g. results/runs/E00.trec.gz)
    qrels_file : Path
        Path to qrels file (JSON or TREC format supported by PyTerrier)

    Returns
    -------
    Dict[str, float]
        Mapping metric_name -> value
    """

    if not run_file.exists():
        raise FileNotFoundError(f"Run file not found: {run_file}")

    if not qrels_file.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    # Make sure PyTerrier is initialized (safe to call multiple times)
    if not pt.started():
        pt.init()

    measures = get_measures()

    # pt.evaluate returns a dict-like object
    results = pt.evaluate(
        run=run_file,
        qrels=qrels_file,
        metrics=measures
    )

    # Convert metric objects to readable strings
    metrics_dict = {}
    for m, v in results.items():
        metrics_dict[str(m)] = float(v)

    return metrics_dict


def save_metrics(
    metrics: Dict[str, float],
    output_path: Path
) -> None:
    """
    Save metrics dictionary to JSON.

    Parameters
    ----------
    metrics : Dict[str, float]
        Evaluation metrics
    output_path : Path
        Where to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--qrels_file")
    ap.add_argument("--run_file")
    ap.add_argument("--output_file")
    args = ap.parse_args()

    metrics = evaluate_run(args.run_file, args.qrels_file)
    save_metrics(metrics, args.output_file)