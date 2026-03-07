# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
NLP Pipeline Enhancements:
  5a. Semantic similarity for document-to-person relevance scoring
  5b. Topic modeling for document categorization
  5c. Extractive summarization for per-document descriptions
  5d. Pipeline-level logging for evaluation and debugging
  5e. Cross-document evidence linking (TODO — stretch goal)

Run: python scripts/pipeline_enhancements.py
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent


# ── 5c. Extractive Summarization ─────────────────────────────

def generate_extractive_summaries():
    """
    Generate one-line extractive summaries for EFTA documents.

    Uses the first 1-2 sentences of each document as an extractive summary.
    Falls back to the document snippet from summaries.json if raw text
    is not available.
    """
    logger.info("Generating extractive document summaries...")

    summaries_path = BASE_PATH / "data" / "processed" / "summaries.json"
    if not summaries_path.exists():
        logger.warning("summaries.json not found, skipping")
        return {}

    with open(summaries_path) as f:
        person_summaries = json.load(f)

    doc_summaries = {}
    seen_efta = set()

    for person_name, person_data in person_summaries.items():
        citations = person_data.get('citations', [])
        for cit in citations:
            doc_id = cit.get('doc_id', '')
            snippet = cit.get('snippet', '')
            doc_type = cit.get('doc_type', 'document')

            # Extract EFTA ID
            m = re.search(r'(EFTA\d+)', doc_id)
            if not m:
                continue
            efta_id = m.group(1)

            if efta_id in seen_efta:
                # Append person mentions to existing entry
                if efta_id in doc_summaries:
                    if person_name not in doc_summaries[efta_id].get('people_mentioned', []):
                        doc_summaries[efta_id]['people_mentioned'].append(person_name)
                continue
            seen_efta.add(efta_id)

            # Extract first 1-2 sentences from snippet as summary
            sentences = re.split(r'(?<=[.!?])\s+', snippet.strip())
            summary = ' '.join(sentences[:2]).strip()
            if len(summary) > 200:
                summary = summary[:200] + '...'

            doc_summaries[efta_id] = {
                'efta_id': efta_id,
                'doc_type': doc_type,
                'summary': summary,
                'people_mentioned': [person_name],
                'source_volume': cit.get('source_volume', ''),
            }

    output_path = BASE_PATH / "data" / "processed" / "document_summaries.json"
    with open(output_path, 'w') as f:
        json.dump(doc_summaries, f, indent=2)

    logger.info(f"Generated summaries for {len(doc_summaries)} documents → {output_path}")
    return doc_summaries


# ── 5b. Topic Modeling / Document Categorization ─────────────

# Simple keyword-based topic categorization (no heavy model needed)
TOPIC_KEYWORDS = {
    'FBI Complaint': ['fbi', 'complaint', 'federal bureau', 'investigation', 'agent', 'special agent'],
    'Flight Records': ['flight', 'manifest', 'passenger', 'aircraft', 'lolita express', 'tail number', 'airport', 'flew'],
    'Financial Records': ['bank', 'wire transfer', 'payment', 'deutsche bank', 'account', 'financial', 'funds', 'transaction', 'million'],
    'Deposition': ['deposition', 'sworn', 'testimony', 'witness', 'under oath', 'testify', 'examined'],
    'Court Filing': ['court', 'plaintiff', 'defendant', 'motion', 'filing', 'judge', 'order', 'ruling', 'complaint filed'],
    'Correspondence': ['email', 'wrote', 'sent', 'dear', 'subject:', 'from:', 'to:', 'message'],
    'News Report': ['reported', 'journalist', 'article', 'media', 'newspaper', 'interview', 'press'],
    'Law Enforcement': ['arrest', 'police', 'detective', 'officer', 'search warrant', 'seized', 'raid'],
}


def categorize_documents(doc_summaries: dict) -> dict:
    """
    Categorize documents into topic categories using keyword matching.
    Returns per-person topic distributions.
    """
    logger.info("Categorizing documents by topic...")

    person_topics = defaultdict(lambda: Counter())
    doc_topics = {}

    for efta_id, doc in doc_summaries.items():
        snippet = doc.get('summary', '').lower()
        doc_type = doc.get('doc_type', 'document').lower()

        # Score each topic
        topic_scores = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in snippet)
            if score > 0:
                topic_scores[topic] = score

        # Assign primary topic (highest score) or infer from doc_type
        if topic_scores:
            primary = max(topic_scores, key=topic_scores.get)
        elif doc_type == 'email':
            primary = 'Correspondence'
        elif doc_type == 'legal_filing':
            primary = 'Court Filing'
        elif doc_type == 'deposition':
            primary = 'Deposition'
        else:
            primary = 'DOJ Document'

        doc['topic'] = primary
        doc_topics[efta_id] = primary

        # Update per-person topic counts
        for person in doc.get('people_mentioned', []):
            person_topics[person][primary] += 1

    # Save document topics
    output_path = BASE_PATH / "data" / "processed" / "document_topics.json"
    with open(output_path, 'w') as f:
        json.dump(doc_topics, f, indent=2)

    # Save per-person topic distributions
    person_topic_dist = {}
    for person, topics in person_topics.items():
        person_topic_dist[person] = dict(topics.most_common())

    topic_dist_path = BASE_PATH / "data" / "processed" / "person_topic_distributions.json"
    with open(topic_dist_path, 'w') as f:
        json.dump(person_topic_dist, f, indent=2)

    logger.info(f"Categorized {len(doc_topics)} documents, {len(person_topic_dist)} person topic distributions")
    return person_topic_dist


# ── 5d. Pipeline-Level Logging ───────────────────────────────

def generate_pipeline_log():
    """
    Generate per-person pipeline log showing the full scoring chain:
    raw features → normalized features → model scores → evidence index → impunity index.
    """
    logger.info("Generating pipeline log for top 30 individuals...")

    # Load evidence scores
    ev_path = BASE_PATH / "data" / "processed" / "evidence_scores.json"
    if not ev_path.exists():
        logger.warning("evidence_scores.json not found")
        return

    with open(ev_path) as f:
        evidence_scores = json.load(f)

    # Load registry for consequence tiers
    reg_path = BASE_PATH / "data" / "processed" / "people_registry.csv"
    consequence_tiers = {}
    if reg_path.exists():
        import csv
        with open(reg_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('name', '')
                tier = int(row.get('consequence_tier', 0) or 0)
                consequence_tiers[name] = tier

    # Load predictions if available
    pred_path = BASE_PATH / "data" / "outputs" / "predictions.csv"
    predictions = {}
    if pred_path.exists():
        import csv
        with open(pred_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('name', '')
                predictions[name] = {
                    k: float(v) for k, v in row.items()
                    if k != 'name' and v and v not in ('', 'nan')
                }

    # Get top 30 by evidence index
    top_30 = sorted(
        evidence_scores.items(),
        key=lambda x: x[1].get('evidence_index', 0),
        reverse=True
    )[:30]

    pipeline_log = []
    for name, ev in top_30:
        tier = consequence_tiers.get(name, 0)
        evidence_idx = ev.get('evidence_index', 0)

        # Compute impunity
        if tier == 0 and evidence_idx > 0:
            impunity = min(10.0, evidence_idx * 1.3)
        elif tier == 2:
            impunity = evidence_idx * 0.7
        else:
            impunity = evidence_idx

        entry = {
            'name': name,
            'raw_features': {
                'jmail_doc_count': ev.get('jmail_doc_count', 0),
                'doc_mentions': ev.get('doc_mentions', 0),
                'keyword_cooccurrence': ev.get('keyword_cooccurrence', 0),
                'flights': ev.get('flights', 0),
                'connections': ev.get('connections', 0),
                'in_black_book': ev.get('in_black_book', False),
            },
            'evidence_index': round(evidence_idx, 2),
            'consequence_tier': tier,
            'consequence_modifier': 1.3 if tier == 0 else (0.7 if tier == 2 else 1.0),
            'impunity_index': round(impunity, 2),
        }

        # Add model predictions if available
        if name in predictions:
            entry['model_scores'] = predictions[name]

        pipeline_log.append(entry)

    output_path = BASE_PATH / "data" / "outputs" / "pipeline_log.json"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pipeline_log, f, indent=2)

    logger.info(f"Pipeline log saved → {output_path}")

    # Print summary
    print("\n=== PIPELINE LOG (TOP 30) ===")
    print(f"{'Name':30s} {'Evidence':>8s} {'Tier':>4s} {'Mod':>5s} {'Impunity':>8s}")
    print("-" * 60)
    for entry in pipeline_log:
        print(f"{entry['name']:30s} {entry['evidence_index']:8.2f} {entry['consequence_tier']:4d} {entry['consequence_modifier']:5.1f} {entry['impunity_index']:8.2f}")


# ── 5e. Cross-Document Evidence Linking (Stretch) ───────────

def identify_corroborating_documents(doc_summaries: dict):
    """
    TODO: Identify when the same testimony or allegation appears
    across multiple documents (corroboration).

    Future approach:
    - Use cross-attention or cosine similarity between document pairs
      mentioning the same individual
    - Group documents that discuss the same allegation/testimony
    - Surface as "Corroborated by X documents" in the UI

    For now, we do a simple approach: group documents by the set of
    people they mention. Documents mentioning the same set of 2+ people
    are likely corroborating.
    """
    logger.info("Identifying corroborating document clusters (simplified)...")

    # Group documents by people mentioned together
    pair_docs = defaultdict(list)
    for efta_id, doc in doc_summaries.items():
        people = sorted(doc.get('people_mentioned', []))
        if len(people) >= 2:
            # Create pairs
            for i in range(len(people)):
                for j in range(i + 1, len(people)):
                    pair_key = f"{people[i]}|{people[j]}"
                    pair_docs[pair_key].append(efta_id)

    # Find pairs with 3+ corroborating documents
    corroboration = {}
    for pair_key, doc_ids in pair_docs.items():
        if len(doc_ids) >= 3:
            people = pair_key.split('|')
            corroboration[pair_key] = {
                'people': people,
                'document_count': len(doc_ids),
                'documents': doc_ids[:10],  # Cap at 10
            }

    output_path = BASE_PATH / "data" / "processed" / "corroboration_clusters.json"
    with open(output_path, 'w') as f:
        json.dump(corroboration, f, indent=2)

    logger.info(f"Found {len(corroboration)} corroborating document clusters → {output_path}")


# ── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    # 5c: Generate extractive summaries
    doc_summaries = generate_extractive_summaries()

    # 5b: Categorize documents by topic
    if doc_summaries:
        categorize_documents(doc_summaries)

        # 5e: Cross-document evidence linking (simplified)
        identify_corroborating_documents(doc_summaries)

    # 5d: Generate pipeline log
    generate_pipeline_log()

    print("\nPipeline enhancements complete.")
