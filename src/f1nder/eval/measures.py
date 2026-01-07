# src/f1nder/eval/measures.py

"""
Definition of evaluation measures required by the assignment.

Why this file exists:
- centralizes the official metrics
- avoids copy-paste and inconsistencies across experiments
"""

from ir_measures import P, R, nDCG, MAP


def get_measures():
    """
    Returns the list of IR measures used for all experiments.

    Metrics required:
    - P@1, P@5, P@10
    - R@5, R@10
    - nDCG@5, nDCG@10
    - MAP
    """
    return [
        P@1,
        P@5,
        P@10,
        R@5,
        R@10,
        nDCG@5,
        nDCG@10,
        MAP
    ]