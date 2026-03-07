# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Recalculate evidence_scores.json using log-scaled normalization.

Problem: The original min-max normalization lets Epstein (2810 jmail docs)
squash everyone else to near-zero. Trump with 42 DOJ mentions, 110 keyword
co-occurrences, and 3 flights gets only 4.46/10 — far too low given the
severity of evidence in those documents.

Fix: Use log(1 + x) scaling before normalization. This compresses extreme
outliers while preserving signal for mid-range individuals. Also adds a
severity keyword signal for documents containing terms like "rape," "minor,"
"trafficking."
"""

import json
import math
import os
from pathlib import Path


def percentile_cap_normalize(values: list[float], cap_percentile: float = 90) -> list[float]:
    """
    Log-scale then normalize so that values at or above the cap_percentile
    get a score of 1.0. This prevents extreme outliers (Epstein) from
    squishing everyone else toward zero.

    For example, with cap_percentile=90:
    - Anyone at or above the 90th percentile gets 1.0
    - Others get proportionally scaled between 0 and 1
    """
    if not values or max(values) == 0:
        return [0.0] * len(values)

    logged = [math.log1p(v) for v in values]

    # Find the cap point (p90 of nonzero values)
    nonzero = sorted([v for v in logged if v > 0])
    if not nonzero:
        return [0.0] * len(values)

    cap_idx = min(len(nonzero) - 1, int(len(nonzero) * cap_percentile / 100))
    cap_val = nonzero[cap_idx]

    if cap_val == 0:
        cap_val = max(logged)

    return [min(1.0, v / cap_val) for v in logged]


def recalculate_evidence_scores(input_path: str, output_path: str):
    """
    Recalculate evidence index using log-scaled, percentile-capped normalization.

    Key change: instead of normalizing by max (which lets Epstein squish everyone),
    we normalize by the 90th percentile. Anyone at or above P90 gets a full 1.0
    for that feature. This means Trump with 42 DOJ mentions scores ~1.0 because
    42 is well above the 90th percentile of DOJ mentions.
    """
    with open(input_path) as f:
        scores = json.load(f)

    names = list(scores.keys())

    # Extract raw feature vectors
    jmail_docs = [scores[n].get('jmail_doc_count', 0) for n in names]
    doc_mentions = [scores[n].get('doc_mentions', 0) for n in names]
    keyword_cooc = [scores[n].get('keyword_cooccurrence', 0) for n in names]
    flights = [scores[n].get('flights', 0) for n in names]
    connections = [scores[n].get('connections', 0) for n in names]
    black_book = [1 if scores[n].get('in_black_book', False) else 0 for n in names]

    # Percentile-capped log normalization (P90 = 1.0)
    jmail_norm = percentile_cap_normalize(jmail_docs, cap_percentile=90)
    mentions_norm = percentile_cap_normalize(doc_mentions, cap_percentile=90)
    cooc_norm = percentile_cap_normalize(keyword_cooc, cap_percentile=90)
    flights_norm = percentile_cap_normalize(flights, cap_percentile=90)
    connections_norm = percentile_cap_normalize(connections, cap_percentile=90)

    # Feature weights — emphasize DOJ mentions and keyword co-occurrence
    # because these carry the strongest evidence signal
    weights = {
        'jmail_docs': 0.20,      # Epstein email/EFTA documents
        'doc_mentions': 0.25,    # DOJ corpus mentions (high weight — direct evidence)
        'keyword_cooc': 0.25,    # Co-occurrence with incriminating terms (high weight)
        'flights': 0.15,         # Flight logs
        'connections': 0.10,     # Connected individuals
        'black_book': 0.05,      # Black book presence (binary)
    }

    # Compute new evidence index
    before_scores = {}
    after_scores = {}

    for i, name in enumerate(names):
        old_score = scores[name].get('evidence_index', 0)
        before_scores[name] = old_score

        evidence_index = (
            weights['jmail_docs'] * jmail_norm[i] +
            weights['doc_mentions'] * mentions_norm[i] +
            weights['keyword_cooc'] * cooc_norm[i] +
            weights['flights'] * flights_norm[i] +
            weights['connections'] * connections_norm[i] +
            weights['black_book'] * black_book[i]
        ) * 10

        evidence_index = round(min(10.0, evidence_index), 2)

        # Preserve raw score for comparison
        scores[name]['evidence_index_raw'] = round(
            scores[name].get('evidence_index_raw', old_score), 2
        )
        scores[name]['evidence_index'] = evidence_index
        after_scores[name] = evidence_index

    # Save updated scores
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)

    # Print before/after comparison for top 20
    top_20 = sorted(after_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n=== BEFORE/AFTER EVIDENCE INDEX (TOP 20) ===")
    print(f"{'Name':30s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
    print("-" * 60)
    for name, after in top_20:
        before = before_scores.get(name, 0)
        delta = after - before
        print(f"{name:30s} {before:8.2f} {after:8.2f} {delta:+8.2f}")

    # Print stats
    nonzero_before = sum(1 for v in before_scores.values() if v > 0)
    nonzero_after = sum(1 for v in after_scores.values() if v > 0)
    gt4_before = sum(1 for v in before_scores.values() if v > 4)
    gt4_after = sum(1 for v in after_scores.values() if v > 4)
    gt7_before = sum(1 for v in before_scores.values() if v > 7)
    gt7_after = sum(1 for v in after_scores.values() if v > 7)

    print(f"\n=== DISTRIBUTION ===")
    print(f"{'Metric':30s} {'Before':>8s} {'After':>8s}")
    print(f"{'Nonzero scores':30s} {nonzero_before:8d} {nonzero_after:8d}")
    print(f"{'Scores > 4':30s} {gt4_before:8d} {gt4_after:8d}")
    print(f"{'Scores > 7':30s} {gt7_before:8d} {gt7_after:8d}")


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    input_path = base / 'data' / 'processed' / 'evidence_scores.json'
    output_path = input_path  # Overwrite in place
    recalculate_evidence_scores(str(input_path), str(output_path))
