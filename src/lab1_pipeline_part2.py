"""
Lab Session 1 — From the Unstructured Web to Structured Entities
=================================================================
Pipeline complet :
  Phase 1 : Chargement du crawler_output.jsonl existant
  Phase 2 : Information Extraction (NER + Dependency Parsing)

Place crawler_output.jsonl dans le même dossier que ce script.

Outputs :
  extracted_knowledge.csv   — entités NER (entity, label, context, source_url)
  extracted_relations.csv   — triples (subject, relation, object, sentence, source_url)

Requirements :
  pip install spacy pandas
  python -m spacy download en_core_web_trf
"""

import json
import re
import csv
from pathlib import Path

import spacy
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_JSONL    = Path("data/crawler_output.jsonl")
OUTPUT_ENT_CSV = Path("data/extracted_knowledge.csv")
OUTPUT_REL_CSV = Path("data/extracted_relations.csv")
MIN_WORD_COUNT = 500   # pages shorter than this are skipped


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    """
    Load pages from crawler_output.jsonl.
    - Supports both 'text' and 'content' field names
    - Deduplicates by URL
    - Filters pages with fewer than MIN_WORD_COUNT words
    """
    pages = []
    seen_urls = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            url = record.get("url", "")

            # Deduplicate
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Support both field names: 'text' and 'content'
            text = record.get("text") or record.get("content") or ""
            word_count = len(text.split())

            if word_count < MIN_WORD_COUNT:
                continue

            pages.append({
                "url":        url,
                "text":       text,
                "word_count": word_count,
            })

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2.1 — NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────

def run_ner(pages: list, nlp) -> list:
    """
    Process each page with spaCy NER.
    Extracts: PERSON, ORG, GPE, DATE, LOC
    Filters: stopwords, single chars, pure numbers.
    Deduplicates per page.
    """
    TARGET_LABELS = {"PERSON", "ORG", "GPE", "DATE", "LOC"}

    STOPWORDS = {
        "google", "facebook", "twitter", "linkedin", "whatsapp",
        "accueil", "consulter", "voir", "détail", "tarif", "prev", "next",
    }

    all_entities = []

    for i, page in enumerate(pages):
        url  = page["url"]
        text = page["text"]

        print(f"  [NER {i+1}/{len(pages)}] {url[:70]}")
        doc = nlp(text)

        seen_in_page = set()

        for ent in doc.ents:
            label    = ent.label_
            ent_text = ent.text.strip()

            if label not in TARGET_LABELS:
                continue
            if len(ent_text) < 3:
                continue
            if ent_text.lower() in STOPWORDS:
                continue
            if re.match(r"^\d+$", ent_text):
                continue

            # Deduplicate within the page
            key = (ent_text.lower(), label)
            if key in seen_in_page:
                continue
            seen_in_page.add(key)

            # Context = the sentence the entity appears in
            context = ent.sent.text.strip()[:200]

            all_entities.append({
                "entity":     ent_text,
                "label":      label,
                "context":    context,
                "source_url": url,
            })

    return all_entities


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2.2 — RELATION EXTRACTION (DEPENDENCY PARSING)
# ─────────────────────────────────────────────────────────────────────────────

def run_relation_extraction(pages: list, nlp) -> list:
    """
    For each sentence with ≥ 2 named entities, extract a triple:
      (subject_entity) → [connecting_verb] → (object_entity)

    Uses spaCy dependency tags:
      - ROOT verb  = main predicate
      - nsubj      = nominal subject
      - dobj       = direct object

    Example:
      "The University of Paris [ORG] is located in France [GPE]."
      → (University of Paris) --[is located in]--> (France)
    """
    TARGET_LABELS = {"PERSON", "ORG", "GPE", "LOC"}
    relations = []
    seen = set()

    for i, page in enumerate(pages):
        url  = page["url"]
        text = page["text"]

        print(f"  [REL {i+1}/{len(pages)}] {url[:70]}")
        doc = nlp(text)

        for sent in doc.sents:
            ents_in_sent = [e for e in sent.ents if e.label_ in TARGET_LABELS]

            if len(ents_in_sent) < 2:
                continue

            # Find the connecting ROOT verb in the sentence
            verb_token = None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl", "ccomp"):
                    verb_token = token
                    break

            # Subject = first entity, Object = last entity in sentence order
            subj_ent = ents_in_sent[0]
            obj_ent  = ents_in_sent[-1]

            if subj_ent.text.lower() == obj_ent.text.lower():
                continue

            # Build verb phrase from the dependency subtree
            if verb_token:
                verb_phrase = " ".join(
                    t.text for t in verb_token.subtree
                    if t.dep_ in ("aux", "auxpass", "neg", "ROOT", "prep", "prt")
                    and not t.is_punct
                ).strip()
                if not verb_phrase:
                    verb_phrase = verb_token.lemma_
            else:
                verb_phrase = "related_to"

            triple_key = (subj_ent.text.lower(), verb_phrase.lower(), obj_ent.text.lower())
            if triple_key in seen:
                continue
            seen.add(triple_key)

            relations.append({
                "subject":       subj_ent.text.strip(),
                "subject_label": subj_ent.label_,
                "relation":      verb_phrase,
                "object":        obj_ent.text.strip(),
                "object_label":  obj_ent.label_,
                "sentence":      sent.text.strip()[:300],
                "source_url":    url,
            })

    return relations


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Phase 1: Load JSONL ───────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1 — LOADING crawler_output.jsonl")
    print("=" * 60)

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"{INPUT_JSONL} not found.\n"
            "Place crawler_output.jsonl in the same folder as this script."
        )

    pages = load_jsonl(INPUT_JSONL)
    print(f"Pages loaded (>={MIN_WORD_COUNT} words, deduplicated): {len(pages)}")

    # ── Phase 2: NLP ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — INFORMATION EXTRACTION")
    print("=" * 60)

    print("\nLoading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_trf")
        print("  Loaded: en_core_web_trf")
    except OSError:
        print("  en_core_web_trf not found — falling back to en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # 2.1 NER
    print("\n── 2.1 Named Entity Recognition ─────────────────────────────────────")
    entities = run_ner(pages, nlp)
    print(f"\nEntities extracted: {len(entities)}")

    label_dist = pd.Series([e["label"] for e in entities]).value_counts()
    print("Label distribution:")
    print(label_dist.to_string())

    with open(OUTPUT_ENT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "label", "context", "source_url"])
        writer.writeheader()
        writer.writerows(entities)
    print(f"Saved → {OUTPUT_ENT_CSV}")

    # 2.2 Relation Extraction
    print("\n── 2.2 Relation Extraction (Dependency Parsing) ──────────────────────")
    relations = run_relation_extraction(pages, nlp)
    print(f"\nRelations extracted: {len(relations)}")

    print("\nSample triples:")
    for r in relations[:10]:
        print(f"  ({r['subject']}) --[{r['relation']}]--> ({r['object']})")

    with open(OUTPUT_REL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "subject_label", "relation",
                        "object", "object_label", "sentence", "source_url"],
        )
        writer.writeheader()
        writer.writerows(relations)
    print(f"Saved → {OUTPUT_REL_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pages processed     : {len(pages)}")
    print(f"  Entities extracted  : {len(entities)}")
    print(f"  Relations extracted : {len(relations)}")
    print(f"\n  Output files:")
    print(f"    {OUTPUT_ENT_CSV}  — NER entities")
    print(f"    {OUTPUT_REL_CSV}  — relations")


if __name__ == "__main__":
    main()
