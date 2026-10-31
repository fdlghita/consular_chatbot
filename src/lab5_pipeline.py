"""
Lab Session 5 — Knowledge Reasoning with SWRL and Knowledge Graph Embedding (KGE)
===================================================================================
Uses outputs from Lab 4 (knowledge_graph.ttl, ontology.ttl, expanded_kb.nt/.ttl)

Pipeline:
  Part 1 — SWRL Rule-Based Reasoning (OWLReady2)
  Part 2 — KGE Data Preparation  (train / valid / test splits)
  Part 3 — KGE Model Training     (TransE + DistMult via PyKEEN)
  Part 4 — Evaluation              (MRR, Hits@1/3/10)
  Part 5 — Embedding Analysis      (Nearest neighbours, t-SNE, relation behaviour)
  Part 6 — Rule vs Embedding Comparison

Inputs  (from data2/ folder produced by lab4_pipeline.py):
  data2/ontology.ttl
  data2/expanded_kb.nt   ← main KGE input
  data2/alignment.ttl    (optional, used by SWRL step)

Outputs (written to data5/):
  data5/train.txt
  data5/valid.txt
  data5/test.txt
  data5/swrl_results.txt
  data5/evaluation_results.json
  data5/tsne_plot.png
  data5/nn_results.json
  data5/relation_analysis.json

Requirements:
  pip install rdflib owlready2 pykeen torch scikit-learn matplotlib pandas
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data2")
OUT_DIR    = Path("data5")
OUT_DIR.mkdir(exist_ok=True)

# Lab 4 inputs
ONTO_TTL     = DATA_DIR / "ontology.ttl"
EXPANDED_NT  = DATA_DIR / "expanded_kb.nt"
EXPANDED_TTL = DATA_DIR / "expanded_kb.ttl"
KB_TTL       = DATA_DIR / "knowledge_graph.ttl"

# Lab 5 outputs
SWRL_OUT     = OUT_DIR / "swrl_results.txt"
TRAIN_TXT    = OUT_DIR / "train.txt"
VALID_TXT    = OUT_DIR / "valid.txt"
TEST_TXT     = OUT_DIR / "test.txt"
EVAL_JSON    = OUT_DIR / "evaluation_results.json"
NN_JSON      = OUT_DIR / "nn_results.json"
REL_JSON     = OUT_DIR / "relation_analysis.json"
TSNE_PNG     = OUT_DIR / "tsne_plot.png"

# KGE hyper-parameters (kept identical across models for fair comparison)
EMBEDDING_DIM  = 100
LEARNING_RATE  = 0.001
BATCH_SIZE     = 512
NUM_EPOCHS     = 200
NEG_SAMPLES    = 64
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — SWRL RULE-BASED REASONING
# ─────────────────────────────────────────────────────────────────────────────

def part1_swrl_reasoning():
    """
    Apply SWRL rules to the consular ontology using OWLReady2.

    Rules implemented on the Moroccan Consular KB:
      Rule A (custom): A consular office that isLocatedIn a country AND
                       isIssuedBy a document → issues that document.
                       Formalised as:
                         ConsularOffice(?o) ∧ isLocatedIn(?o,?c) ∧ isIssuedBy(?d,?o)
                         → issues(?o,?d)

      Rule B (custom): An administrative procedure that requires a document
                       AND that document isIssuedBy an office
                       → the procedure canBeSubmittedTo that office.
                         ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o)
                         → canBeSubmittedTo(?p,?o)
    """
    print("\n" + "=" * 60)
    print("PART 1 — SWRL Rule-Based Reasoning (OWLReady2)")
    print("=" * 60)

    try:
        from owlready2 import get_ontology, sync_reasoner_pellet, SWRL, Imp
        import owlready2
    except ImportError:
        print("  ⚠  owlready2 not installed. Run: pip install owlready2")
        print("  → Skipping SWRL part.")
        _write_swrl_stub()
        return

    # We load the private ontology (ontology.ttl).
    # OWLReady2 needs an OWL/XML or RDF/XML; we convert if necessary.
    if not ONTO_TTL.exists():
        print(f"  ⚠  {ONTO_TTL} not found. Skipping SWRL.")
        _write_swrl_stub()
        return

    print(f"  Loading ontology: {ONTO_TTL}")

    try:
        onto_iri = ONTO_TTL.resolve().as_uri()
        onto = get_ontology(onto_iri).load()
    except Exception as e:
        print(f"  ⚠  Could not load ontology via OWLReady2: {e}")
        print("  → Attempting rdflib-based SWRL simulation instead.")
        _simulate_swrl_with_rdflib()
        return

    with onto:
        # ── Declare classes if not already present ────────────────────────────
        if not onto.search(iri="*ConsularOffice"):
            class ConsularOffice(onto.GovernmentOrganization if hasattr(onto, 'GovernmentOrganization') else owlready2.Thing): pass
        if not onto.search(iri="*ConsularProcedure"):
            class ConsularProcedure(owlready2.Thing): pass
        if not onto.search(iri="*AdministrativeDocument"):
            class AdministrativeDocument(owlready2.Thing): pass

        # ── Declare properties if not already present ─────────────────────────
        for prop_name in ['isLocatedIn', 'isIssuedBy', 'issues', 'requires', 'canBeSubmittedTo']:
            if not onto.search(iri=f"*{prop_name}"):
                new_prop = owlready2.types.new_class(prop_name, (owlready2.ObjectProperty,))

        # ── Rule A: ConsularOffice isLocatedIn country ∧ isIssuedBy doc
        #            → issues doc ─────────────────────────────────────────────
        rule_a = Imp()
        rule_a.set_as_rule(
            "ConsularOffice(?o), isLocatedIn(?o,?c), isIssuedBy(?d,?o) -> issues(?o,?d)"
        )

        # ── Rule B: ConsularProcedure requires doc ∧ isIssuedBy office
        #            → canBeSubmittedTo office ───────────────────────────────
        rule_b = Imp()
        rule_b.set_as_rule(
            "ConsularProcedure(?p), requires(?p,?d), isIssuedBy(?d,?o) -> canBeSubmittedTo(?p,?o)"
        )

    # Run the reasoner
    print("  Running Pellet reasoner...")
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=False)
        print("  ✓ Reasoning complete.")
    except Exception as e:
        print(f"  ⚠  Reasoner error: {e}")
        print("  → Saving partial results.")

    # Collect inferred triples
    results = []
    for cls in [onto.ConsularOffice] if hasattr(onto, 'ConsularOffice') else []:
        for inst in cls.instances():
            issued = getattr(inst, 'issues', [])
            for doc in issued:
                results.append(f"RULE_A: {inst} issues {doc}")

    for cls in [onto.ConsularProcedure] if hasattr(onto, 'ConsularProcedure') else []:
        for inst in cls.instances():
            submitted = getattr(inst, 'canBeSubmittedTo', [])
            for office in submitted:
                results.append(f"RULE_B: {inst} canBeSubmittedTo {office}")

    _save_swrl_results(results)


def _simulate_swrl_with_rdflib():
    """
    Fallback: simulate SWRL rules using plain rdflib graph traversal.
    Implements the same two rules without a full OWL reasoner.
    """
    from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL

    print("  Loading KB with rdflib for SWRL simulation...")
    g = Graph()
    if EXPANDED_TTL.exists():
        g.parse(str(EXPANDED_TTL), format='turtle')
    elif EXPANDED_NT.exists():
        g.parse(str(EXPANDED_NT), format='nt')
    elif KB_TTL.exists():
        g.parse(str(KB_TTL), format='turtle')
    else:
        print("  ⚠  No KB file found.")
        _write_swrl_stub()
        return

    BASE  = Namespace("http://consulat.ma/kg/")
    WDT   = Namespace("http://www.wikidata.org/prop/direct/")
    SCHEMA = Namespace("http://schema.org/")

    # Collect triples indexed by predicate
    located_in    = defaultdict(set)   # org → place
    issued_by     = defaultdict(set)   # doc → org
    requires_map  = defaultdict(set)   # proc → doc
    issues_map    = defaultdict(set)   # org → doc  (existing)
    submit_map    = defaultdict(set)   # proc → org (existing)

    # We accept both private predicates and aligned Wikidata predicates
    LOCATED_PREDS = {BASE.isLocatedIn, WDT.P131}
    ISSUED_PREDS  = {BASE.isIssuedBy,  WDT.P176}
    REQUIRES_PREDS = {BASE.requires,   WDT.P4633}

    for s, p, o in g:
        if p in LOCATED_PREDS:
            located_in[s].add(o)
        elif p in ISSUED_PREDS:
            issued_by[s].add(o)
        elif p in REQUIRES_PREDS:
            requires_map[s].add(o)
        elif p == BASE.issues:
            issues_map[s].add(o)
        elif p == BASE.canBeSubmittedTo:
            submit_map[s].add(o)

    inferred = []

    # Rule A: office isLocatedIn country ∧ doc isIssuedBy office → office issues doc
    # (any entity that has isLocatedIn AND some doc points to it via isIssuedBy)
    offices_with_location = set(located_in.keys())
    for doc, orgs in issued_by.items():
        for org in orgs:
            if org in offices_with_location:
                if doc not in issues_map.get(org, set()):
                    inferred.append({
                        'rule': 'A',
                        'description': 'office issues document (inferred)',
                        'subject': str(org),
                        'predicate': str(BASE.issues),
                        'object': str(doc),
                    })

    # Rule B: proc requires doc ∧ doc isIssuedBy office → proc canBeSubmittedTo office
    for proc, docs in requires_map.items():
        for doc in docs:
            for org in issued_by.get(doc, set()):
                if org not in submit_map.get(proc, set()):
                    inferred.append({
                        'rule': 'B',
                        'description': 'procedure canBeSubmittedTo office (inferred)',
                        'subject': str(proc),
                        'predicate': str(BASE.canBeSubmittedTo),
                        'object': str(org),
                    })

    print(f"  Rule A inferred: {sum(1 for r in inferred if r['rule']=='A')} new 'issues' triples")
    print(f"  Rule B inferred: {sum(1 for r in inferred if r['rule']=='B')} new 'canBeSubmittedTo' triples")

    results = [f"RULE_{r['rule']}: {r['subject']} {r['predicate']} {r['object']}" for r in inferred]
    _save_swrl_results(results, inferred)


def _save_swrl_results(results: list, structured: list = None):
    lines = [
        "=" * 60,
        "SWRL Reasoning Results",
        "=" * 60,
        "",
        "Rule A: ConsularOffice(?o) ∧ isLocatedIn(?o,?c) ∧ isIssuedBy(?d,?o) → issues(?o,?d)",
        "Rule B: ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)",
        "",
        f"Total inferred triples: {len(results)}",
        "",
    ] + results[:200]  # cap display at 200

    with open(SWRL_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Saved SWRL results → {SWRL_OUT}")

    if structured:
        swrl_json = OUT_DIR / "swrl_results.json"
        with open(swrl_json, "w", encoding="utf-8") as f:
            json.dump(structured[:500], f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved structured SWRL results → {swrl_json}")


def _write_swrl_stub():
    with open(SWRL_OUT, "w", encoding="utf-8") as f:
        f.write("SWRL reasoning skipped (owlready2 not available or KB not found).\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — KGE DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def part2_prepare_kge_data() -> list:
    """
    Load expanded_kb.nt, clean, deduplicate, index entities/relations,
    split 80/10/10 into train/valid/test ensuring no entity appears only
    in validation or test.

    Returns the full list of (head, relation, tail) string triples.
    """
    print("\n" + "=" * 60)
    print("PART 2 — KGE Data Preparation")
    print("=" * 60)

    # ── Load triples ──────────────────────────────────────────────────────────
    if not EXPANDED_NT.exists() and not EXPANDED_TTL.exists():
        raise FileNotFoundError(
            "Neither data2/expanded_kb.nt nor data2/expanded_kb.ttl found.\n"
            "Run lab4_pipeline.py first."
        )

    from rdflib import Graph, Literal, BNode

    g = Graph()
    if EXPANDED_NT.exists():
        print(f"  Loading {EXPANDED_NT}...")
        g.parse(str(EXPANDED_NT), format='nt')
    else:
        print(f"  Loading {EXPANDED_TTL}...")
        g.parse(str(EXPANDED_TTL), format='turtle')
    print(f"  Raw triples loaded: {len(g)}")

    # ── Filter: keep only URI→URI triples (no literals, no blank nodes) ───────
    triples = []
    seen = set()
    for s, p, o in g:
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        if isinstance(o, Literal):
            continue
        h = str(s)
        r = str(p)
        t = str(o)
        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        triples.append(key)

    print(f"  Triples after cleaning (URI-only, deduped): {len(triples)}")

    # ── Remove predicates that add noise (rdf:type with literals, etc.) ───────
    EXCLUDED_PREDS = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",    # keep only if you want class membership
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://www.w3.org/2002/07/owl#sameAs",               # alignment triples
        "http://www.w3.org/2002/07/owl#equivalentProperty",
        "http://www.w3.org/2002/07/owl#equivalentClass",
    }
    triples = [(h, r, t) for (h, r, t) in triples if r not in EXCLUDED_PREDS]
    print(f"  Triples after predicate filtering: {len(triples)}")

    # Optionally keep rdf:type if you want class-membership edges
    # Uncomment the next block to re-add them:
    # for s, p, o in g:
    #     if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
    #         if not isinstance(o, Literal) and not isinstance(s, BNode):
    #             triples.append((str(s), str(p), str(o)))

    # ── Entity / relation index ───────────────────────────────────────────────
    entities  = sorted({h for h,r,t in triples} | {t for h,r,t in triples})
    relations = sorted({r for h,r,t in triples})

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    print(f"  Entities  : {len(entities)}")
    print(f"  Relations : {len(relations)}")

    # Save entity / relation mappings
    with open(OUT_DIR / "entity2id.json", "w", encoding="utf-8") as f:
        json.dump(ent2id, f, ensure_ascii=False)
    with open(OUT_DIR / "relation2id.json", "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False)

    # ── Train / Valid / Test split ────────────────────────────────────────────
    random.shuffle(triples)

    n      = len(triples)
    n_test = max(1, int(0.10 * n))
    n_val  = max(1, int(0.10 * n))
    n_train = n - n_test - n_val

    train = triples[:n_train]
    valid = triples[n_train: n_train + n_val]
    test  = triples[n_train + n_val:]

    # ── Ensure no entity appears ONLY in valid/test ───────────────────────────
    train_ents = {h for h,r,t in train} | {t for h,r,t in train}
    orphans_v = [(h,r,t) for (h,r,t) in valid if h not in train_ents or t not in train_ents]
    orphans_t = [(h,r,t) for (h,r,t) in test  if h not in train_ents or t not in train_ents]

    # Move orphan triples into train
    for tri in orphans_v:
        valid.remove(tri)
        train.append(tri)
    for tri in orphans_t:
        test.remove(tri)
        train.append(tri)

    print(f"  Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    print(f"  Orphan triples moved to train: {len(orphans_v)+len(orphans_t)}")

    # ── Write text files ──────────────────────────────────────────────────────
    def _write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in data:
                f.write(f"{h}\t{r}\t{t}\n")

    _write(TRAIN_TXT, train)
    _write(VALID_TXT, valid)
    _write(TEST_TXT,  test)

    print(f"  ✓ {TRAIN_TXT}")
    print(f"  ✓ {VALID_TXT}")
    print(f"  ✓ {TEST_TXT}")

    return triples


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — EMBEDDING MODEL TRAINING (PyKEEN)
# ─────────────────────────────────────────────────────────────────────────────

def part3_train_models():
    """
    Train TransE and DistMult using PyKEEN.
    Also trains on 20k and 50k subsets for KB-size sensitivity analysis.
    Returns dict: {model_name: pipeline_result}
    """
    print("\n" + "=" * 60)
    print("PART 3 — KGE Model Training (PyKEEN)")
    print("=" * 60)

    try:
        import torch
        import numpy as np
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
    except ImportError:
        print("  ⚠  PyKEEN / PyTorch not installed.")
        print("  Run: pip install pykeen torch")
        return {}

    # ── Load train/valid/test from text files ─────────────────────────────────
    def load_txt(path):
        """Load tab-separated triples and return as a numpy str array (N, 3)."""
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    rows.append(parts)
        import numpy as np
        return np.array(rows, dtype=str) if rows else np.empty((0, 3), dtype=str)

    train_raw = load_txt(TRAIN_TXT)
    valid_raw = load_txt(VALID_TXT)
    test_raw  = load_txt(TEST_TXT)

    print(f"  Train: {len(train_raw)} | Valid: {len(valid_raw)} | Test: {len(test_raw)}")

    # Build factories
    tf_train = TriplesFactory.from_labeled_triples(
        triples=train_raw, create_inverse_triples=False
    )

    def _safe_factory(raw, ent2id, rel2id):
        """Build a TriplesFactory keeping only triples whose entities and
        relations are already in the training vocabulary."""
        mask = np.array(
            [(r[0] in ent2id and r[2] in ent2id and r[1] in rel2id) for r in raw],
            dtype=bool,
        )
        filtered = raw[mask]
        if len(filtered) == 0:
            return None
        return TriplesFactory.from_labeled_triples(
            triples=filtered,
            entity_to_id=ent2id,
            relation_to_id=rel2id,
            create_inverse_triples=False,
        )

    tf_valid = _safe_factory(valid_raw, tf_train.entity_to_id, tf_train.relation_to_id)
    tf_test  = _safe_factory(test_raw,  tf_train.entity_to_id, tf_train.relation_to_id)

    if tf_valid is None or tf_test is None:
        print("  ⚠  Validation or test set is empty after vocabulary filtering.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    results = {}

    # ── Helper: train one model ───────────────────────────────────────────────
    def train_model(model_name: str, training_factory, label: str):
        print(f"\n  ── Training {model_name} ({label}) ─────────────────────────")
        t0 = time.time()
        result = pipeline(
            training=training_factory,
            validation=tf_valid,
            testing=tf_test,
            model=model_name,
            model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
            optimizer="Adam",
            optimizer_kwargs=dict(lr=LEARNING_RATE),
            training_kwargs=dict(
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            ),
            negative_sampler="basic",
            negative_sampler_kwargs=dict(num_negs_per_pos=NEG_SAMPLES),
            evaluator="RankBasedEvaluator",
            evaluator_kwargs=dict(filtered=True),
            random_seed=RANDOM_SEED,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"  ✓ {model_name} ({label}) trained in {elapsed:.1f}s")
        return result

    # ── Full dataset — TransE ─────────────────────────────────────────────────
    results["TransE_full"] = train_model("TransE", tf_train, "full")

    # ── Full dataset — DistMult ───────────────────────────────────────────────
    results["DistMult_full"] = train_model("DistMult", tf_train, "full")

    # ── Reusable filter helper for subset splits (uses numpy arrays) ──────────
    def _filter_split(raw, ent2id, rel2id):
        mask = np.array(
            [(r[0] in ent2id and r[2] in ent2id and r[1] in rel2id) for r in raw],
            dtype=bool,
        )
        filtered = raw[mask]
        if len(filtered) == 0:
            return None
        return TriplesFactory.from_labeled_triples(
            triples=filtered,
            entity_to_id=ent2id,
            relation_to_id=rel2id,
            create_inverse_triples=False,
        )

    # ── KB Size Sensitivity: 20k subset ──────────────────────────────────────
    n20 = min(20000, len(train_raw))
    if n20 < len(train_raw):
        idx_20k = np.random.choice(len(train_raw), n20, replace=False)
        tf_20k = TriplesFactory.from_labeled_triples(
            triples=train_raw[idx_20k],
            create_inverse_triples=False,
        )
        tf_valid_20k = _filter_split(valid_raw, tf_20k.entity_to_id, tf_20k.relation_to_id)
        tf_test_20k  = _filter_split(test_raw,  tf_20k.entity_to_id, tf_20k.relation_to_id)

        if tf_valid_20k and tf_test_20k:
            try:
                res_20k = pipeline(
                    training=tf_20k,
                    validation=tf_valid_20k,
                    testing=tf_test_20k,
                    model="TransE",
                    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
                    optimizer="Adam",
                    optimizer_kwargs=dict(lr=LEARNING_RATE),
                    training_kwargs=dict(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE),
                    negative_sampler="basic",
                    negative_sampler_kwargs=dict(num_negs_per_pos=NEG_SAMPLES),
                    evaluator="RankBasedEvaluator",
                    evaluator_kwargs=dict(filtered=True),
                    random_seed=RANDOM_SEED,
                    device=device,
                )
                results["TransE_20k"] = res_20k
                print("  ✓ TransE (20k) trained")
            except Exception as e:
                print(f"  ⚠  TransE (20k) failed: {e}")

    # ── KB Size Sensitivity: 50k subset ──────────────────────────────────────
    n50 = min(50000, len(train_raw))
    if n50 < len(train_raw) and n50 > n20:
        idx_50k = np.random.choice(len(train_raw), n50, replace=False)
        tf_50k = TriplesFactory.from_labeled_triples(
            triples=train_raw[idx_50k],
            create_inverse_triples=False,
        )
        tf_valid_50k = _filter_split(valid_raw, tf_50k.entity_to_id, tf_50k.relation_to_id)
        tf_test_50k  = _filter_split(test_raw,  tf_50k.entity_to_id, tf_50k.relation_to_id)

        if tf_valid_50k and tf_test_50k:
            try:
                res_50k = pipeline(
                    training=tf_50k,
                    validation=tf_valid_50k,
                    testing=tf_test_50k,
                    model="TransE",
                    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
                    optimizer="Adam",
                    optimizer_kwargs=dict(lr=LEARNING_RATE),
                    training_kwargs=dict(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE),
                    negative_sampler="basic",
                    negative_sampler_kwargs=dict(num_negs_per_pos=NEG_SAMPLES),
                    evaluator="RankBasedEvaluator",
                    evaluator_kwargs=dict(filtered=True),
                    random_seed=RANDOM_SEED,
                    device=device,
                )
                results["TransE_50k"] = res_50k
                print("  ✓ TransE (50k) trained")
            except Exception as e:
                print(f"  ⚠  TransE (50k) failed: {e}")

    # Save models
    for name, res in results.items():
        model_path = OUT_DIR / f"model_{name}"
        model_path.mkdir(exist_ok=True)
        try:
            import torch
            torch.save(res.model.state_dict(), model_path / "model.pt")
            print(f"  ✓ Saved model weights → {model_path}/model.pt")
        except Exception as e:
            print(f"  ⚠  Could not save {name}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — EVALUATION (MRR, Hits@1/3/10)
# ─────────────────────────────────────────────────────────────────────────────

def part4_evaluate(results: dict):
    """
    Extract and display evaluation metrics from PyKEEN pipeline results.
    Writes evaluation_results.json.
    """
    print("\n" + "=" * 60)
    print("PART 4 — Evaluation Results")
    print("=" * 60)

    if not results:
        print("  No trained models found — skipping evaluation.")
        return

    eval_data = {}

    def _get_metric(metrics, key, side="both"):
        """Safe metric extraction from PyKEEN's result object."""
        try:
            return metrics.get_metric(key)
        except Exception:
            try:
                return float(metrics.to_dict()[side][key])
            except Exception:
                return None

    print(f"\n  {'Model':<20} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}")
    print("  " + "-" * 52)

    for name, result in results.items():
        if result is None:
            continue
        try:
            m = result.metric_results
            mrr  = _get_metric(m, "mean_reciprocal_rank")
            h1   = _get_metric(m, "hits_at_1")
            h3   = _get_metric(m, "hits_at_3")
            h10  = _get_metric(m, "hits_at_10")

            # PyKEEN ≥1.10 uses different key format
            if mrr is None:
                try:
                    mrr  = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
                    h1   = result.get_metric("both.realistic.hits_at_1")
                    h3   = result.get_metric("both.realistic.hits_at_3")
                    h10  = result.get_metric("both.realistic.hits_at_10")
                except Exception:
                    pass

            eval_data[name] = {
                "MRR":     mrr,
                "Hits@1":  h1,
                "Hits@3":  h3,
                "Hits@10": h10,
            }

            def _fmt(v):
                return f"{v:.4f}" if v is not None else "  N/A "

            print(f"  {name:<20} {_fmt(mrr):>8} {_fmt(h1):>8} {_fmt(h3):>8} {_fmt(h10):>8}")

        except Exception as e:
            print(f"  ⚠  Could not extract metrics for {name}: {e}")
            eval_data[name] = {"error": str(e)}

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\n  ✓ Saved evaluation results → {EVAL_JSON}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — EMBEDDING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def part5_embedding_analysis(results: dict):
    """
    6.1 Nearest neighbours in embedding space
    6.2 t-SNE clustering plot
    6.3 Relation behaviour analysis (symmetric, inverse, composition)
    """
    print("\n" + "=" * 60)
    print("PART 5 — Embedding Analysis")
    print("=" * 60)

    if not results:
        print("  No trained models — skipping analysis.")
        return

    # Use the best model (DistMult_full preferred, else TransE_full)
    best_name = "DistMult_full" if "DistMult_full" in results else next(iter(results))
    best_result = results[best_name]
    print(f"  Using model: {best_name}")

    try:
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pykeen.triples import TriplesFactory
    except ImportError as e:
        print(f"  ⚠  Missing dependency: {e}")
        return

    model = best_result.model
    model.eval()

    # Entity embeddings
    with torch.no_grad():
        ent_embeddings = model.entity_representations[0](
            indices=None
        ).cpu().numpy()

    entity_labels = best_result.training.entity_id_to_label

    print(f"  Entity embedding shape: {ent_embeddings.shape}")

    # ── 6.1 Nearest Neighbours ────────────────────────────────────────────────
    print("\n  6.1 Nearest Neighbours")

    # Pick a few anchor entities (countries / organizations present in KB)
    ANCHOR_KEYWORDS = [
        "Q1028",   # Morocco
        "Q142",    # France
        "Q30",     # USA
        "Q145",    # UK
        "Q16",     # Canada
    ]

    nn_results = {}
    id_to_label = {v: k for k, v in best_result.training.entity_to_id.items()}

    sims = cosine_similarity(ent_embeddings)

    for kw in ANCHOR_KEYWORDS:
        # Find entity id that contains the keyword
        anchor_ids = [eid for lbl, eid in best_result.training.entity_to_id.items() if kw in lbl]
        if not anchor_ids:
            continue
        anchor_id = anchor_ids[0]
        anchor_label = id_to_label[anchor_id]

        sim_scores = sims[anchor_id]
        top_ids = sim_scores.argsort()[::-1][1:11]   # top-10 excluding self
        neighbours = [(id_to_label[i], float(sim_scores[i])) for i in top_ids]

        nn_results[anchor_label] = neighbours
        print(f"    Nearest to {anchor_label}:")
        for lbl, sc in neighbours[:5]:
            short = lbl.split("/")[-1]
            print(f"      {short:<30} (cos={sc:.4f})")

    with open(NN_JSON, "w", encoding="utf-8") as f:
        json.dump(nn_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved nearest-neighbour results → {NN_JSON}")

    # ── 6.2 t-SNE Plot ────────────────────────────────────────────────────────
    print("\n  6.2 t-SNE Clustering")

    # Subsample for speed (up to 3000 entities)
    max_ents = min(3000, ent_embeddings.shape[0])
    sample_ids = random.sample(range(ent_embeddings.shape[0]), max_ents)
    sample_emb = ent_embeddings[sample_ids]
    sample_labels = [id_to_label[i] for i in sample_ids]

    # Assign rough ontology class by URI pattern
    WIKIDATA_BASE = "http://www.wikidata.org"
    PRIVATE_BASE  = "http://consulat.ma"

    def _classify(uri: str) -> str:
        if WIKIDATA_BASE in uri:
            return "Wikidata"
        if PRIVATE_BASE in uri:
            return "Consular"
        if "schema.org" in uri:
            return "Schema.org"
        return "Other"

    colors_map = {"Wikidata": "steelblue", "Consular": "tomato",
                  "Schema.org": "gold", "Other": "grey"}
    classes = [_classify(l) for l in sample_labels]

    print(f"  Running t-SNE on {max_ents} entities...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=300)
    coords = tsne.fit_transform(sample_emb)

    fig, ax = plt.subplots(figsize=(12, 9))
    for cls_name, color in colors_map.items():
        idx = [i for i, c in enumerate(classes) if c == cls_name]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=color, label=cls_name, s=10, alpha=0.6)

    ax.set_title(f"t-SNE of Entity Embeddings ({best_name}, n={max_ents})")
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(TSNE_PNG, dpi=150)
    plt.close()
    print(f"  ✓ Saved t-SNE plot → {TSNE_PNG}")

    # ── 6.3 Relation Behaviour ────────────────────────────────────────────────
    print("\n  6.3 Relation Behaviour Analysis")

    # Relation embeddings (TransE/DistMult both expose relation_representations)
    try:
        with torch.no_grad():
            rel_emb = model.relation_representations[0](
                indices=None
            ).cpu().numpy()
    except Exception:
        print("  ⚠  Could not extract relation embeddings.")
        return

    id_to_rel = {v: k for k, v in best_result.training.relation_to_id.items()}

    rel_analysis = {}

    rel_sims = cosine_similarity(rel_emb)

    for rid, ruri in id_to_rel.items():
        r_name = ruri.split("/")[-1]

        # Symmetric: check if r ≈ r^(-1) by looking at top similar relations
        # (a proper check needs inverse triples; here we use cosine as proxy)
        top_similar_rels = rel_sims[rid].argsort()[::-1][1:6]
        top_labels = [id_to_rel[i].split("/")[-1] for i in top_similar_rels]

        rel_analysis[r_name] = {
            "uri": ruri,
            "top_similar_relations": top_labels,
            "embedding_norm": float(np.linalg.norm(rel_emb[rid])),
        }

    # Identify candidate inverse pairs (r1 · r2 ≈ identity)
    # TransE: if r1 + r2 ≈ 0 then they are inverses
    inverse_pairs = []
    norms = np.linalg.norm(rel_emb, axis=1, keepdims=True) + 1e-9
    rel_emb_norm = rel_emb / norms

    for i in range(len(id_to_rel)):
        for j in range(i + 1, len(id_to_rel)):
            dot = float(rel_emb_norm[i] @ rel_emb_norm[j])
            # Inverse: vectors should be close to anti-parallel (dot ≈ -1)
            if dot < -0.85:
                inverse_pairs.append({
                    "r1": id_to_rel[i].split("/")[-1],
                    "r2": id_to_rel[j].split("/")[-1],
                    "cosine": dot,
                })

    rel_output = {
        "per_relation": rel_analysis,
        "candidate_inverse_pairs": inverse_pairs[:20],
    }

    with open(REL_JSON, "w", encoding="utf-8") as f:
        json.dump(rel_output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved relation analysis → {REL_JSON}")
    if inverse_pairs:
        print(f"  Candidate inverse pairs found: {len(inverse_pairs)}")
        for p in inverse_pairs[:5]:
            print(f"    {p['r1']}  ↔  {p['r2']}  (cos={p['cosine']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — RULE vs EMBEDDING COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def part6_rule_vs_embedding(results: dict):
    """
    Compare the SWRL rule:
      ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)

    with the KGE analogy:
      vector(requires) + vector(isIssuedBy) ≈ vector(canBeSubmittedTo)

    Measures the cosine similarity between the composed vector and the target.
    """
    print("\n" + "=" * 60)
    print("PART 6 — Rule vs Embedding Comparison")
    print("=" * 60)

    if not results:
        print("  No trained models — skipping comparison.")
        return

    try:
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  ⚠  Missing numpy/torch/sklearn — skipping.")
        return

    best_name = "TransE_full" if "TransE_full" in results else next(iter(results))
    best_result = results[best_name]
    print(f"  Using model: {best_name}")

    model = best_result.model
    model.eval()

    rel2id = best_result.training.relation_to_id

    # Identify the relation IDs we need
    REQUIRES_KEYWORDS   = ["requires", "P4633"]
    ISSUED_KEYWORDS     = ["isIssuedBy", "P176"]
    SUBMITTED_KEYWORDS  = ["canBeSubmittedTo", "submittedTo"]

    def _find_rel_id(keywords):
        for kw in keywords:
            for ruri, rid in rel2id.items():
                if kw.lower() in ruri.lower():
                    return rid, ruri
        return None, None

    req_id,  req_uri  = _find_rel_id(REQUIRES_KEYWORDS)
    iss_id,  iss_uri  = _find_rel_id(ISSUED_KEYWORDS)
    sub_id,  sub_uri  = _find_rel_id(SUBMITTED_KEYWORDS)

    if req_id is None or iss_id is None:
        print("  ⚠  Could not find 'requires' or 'isIssuedBy' in relation index.")
        print("  → Listing available relations for manual check:")
        for r in sorted(rel2id.keys())[:20]:
            print(f"      {r.split('/')[-1]}")
        return

    with torch.no_grad():
        rel_emb = model.relation_representations[0](indices=None).cpu().numpy()

    v_req = rel_emb[req_id]
    v_iss = rel_emb[iss_id]
    v_sub = rel_emb[sub_id] if sub_id is not None else None

    composed = v_req + v_iss   # TransE composition: r1 + r2

    if v_sub is not None:
        sim = float(cosine_similarity([composed], [v_sub])[0][0])
        print(f"\n  SWRL Rule B: requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)")
        print(f"  Embedding analogy: v(requires) + v(isIssuedBy) ≈ v(canBeSubmittedTo)")
        print(f"  Cosine similarity: {sim:.4f}")
        if sim > 0.7:
            print("  → Strong match ✅ — embedding captures the compositional rule.")
        elif sim > 0.4:
            print("  → Moderate match ⚠  — partial compositional structure learned.")
        else:
            print("  → Weak match ❌ — TransE did not capture this composition.")
    else:
        # Find nearest relation to the composed vector
        sims = cosine_similarity([composed], rel_emb)[0]
        top_ids = sims.argsort()[::-1][:5]
        id_to_rel = {v: k for k, v in rel2id.items()}
        print(f"\n  Composed vector v(requires) + v(isIssuedBy)")
        print(f"  nearest relations in embedding space:")
        for i in top_ids:
            print(f"    {id_to_rel[i].split('/')[-1]:<40} (cos={sims[i]:.4f})")

    comparison_output = {
        "rule": "ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)",
        "model_used": best_name,
        "requires_uri": req_uri,
        "isIssuedBy_uri": iss_uri,
        "canBeSubmittedTo_uri": sub_uri,
        "cosine_sim_composed_vs_target": float(
            cosine_similarity([composed], [v_sub])[0][0]
        ) if v_sub is not None else None,
    }
    with open(OUT_DIR / "rule_vs_embedding.json", "w", encoding="utf-8") as f:
        json.dump(comparison_output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved comparison → {OUT_DIR}/rule_vs_embedding.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Lab 5 — Knowledge Reasoning + Knowledge Graph Embedding")
    print("=" * 60)

    # ── Part 1: SWRL ──────────────────────────────────────────────────────────
    part1_swrl_reasoning()

    # ── Part 2: KGE data preparation ─────────────────────────────────────────
    part2_prepare_kge_data()

    # ── Part 3: Train models ──────────────────────────────────────────────────
    results = part3_train_models()

    # ── Part 4: Evaluate ──────────────────────────────────────────────────────
    part4_evaluate(results)

    # ── Part 5: Embedding analysis ────────────────────────────────────────────
    part5_embedding_analysis(results)

    # ── Part 6: Rule vs Embedding ─────────────────────────────────────────────
    part6_rule_vs_embedding(results)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅  Pipeline complete — output files:")
    print("=" * 60)
    outputs = [
        (SWRL_OUT,             "Part 1 — SWRL reasoning results"),
        (OUT_DIR/"swrl_results.json", "Part 1 — SWRL structured output"),
        (TRAIN_TXT,            "Part 2 — KGE training set"),
        (VALID_TXT,            "Part 2 — KGE validation set"),
        (TEST_TXT,             "Part 2 — KGE test set"),
        (OUT_DIR/"entity2id.json",   "Part 2 — Entity index"),
        (OUT_DIR/"relation2id.json", "Part 2 — Relation index"),
        (EVAL_JSON,            "Part 4 — Evaluation metrics"),
        (NN_JSON,              "Part 5 — Nearest neighbours"),
        (TSNE_PNG,             "Part 5 — t-SNE clustering plot"),
        (REL_JSON,             "Part 5 — Relation analysis"),
        (OUT_DIR/"rule_vs_embedding.json", "Part 6 — Rule vs embedding"),
    ]
    for path, desc in outputs:
        status = "✅" if Path(path).exists() else "⬜"
        print(f"  {status}  {str(path):<45} {desc}")


if __name__ == "__main__":
    main()