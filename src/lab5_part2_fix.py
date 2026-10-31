"""
lab5_fix.py — Complète les parties manquantes de lab5 SANS ré-entraîner les modèles.

Ce script suppose que lab5_pipeline.py a déjà tourné et produit :
  data5/train.txt, valid.txt, test.txt
  data5/model_TransE_full/model.pt
  data5/model_DistMult_full/model.pt
  data5/relation2id.json
  data5/entity2id.json

Il exécute uniquement :
  Part 1 — SWRL (rdflib simulation, sans owlready2)
  Part 5 — Analyse des embeddings (nearest neighbours, t-SNE, relations)
  Part 6 — Comparaison règle vs embedding

Install manquant : pip install matplotlib
"""

import json
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_DIR  = Path("data2")
OUT_DIR   = Path("data5")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EXPANDED_NT  = DATA_DIR / "expanded_kb.nt"
EXPANDED_TTL = DATA_DIR / "expanded_kb.ttl"
KB_TTL       = DATA_DIR / "knowledge_graph.ttl"
SWRL_OUT     = OUT_DIR  / "swrl_results.txt"
TRAIN_TXT    = OUT_DIR  / "train.txt"
NN_JSON      = OUT_DIR  / "nn_results.json"
REL_JSON     = OUT_DIR  / "relation_analysis.json"
TSNE_PNG     = OUT_DIR  / "tsne_plot.png"

EMBEDDING_DIM = 100


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_txt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                rows.append(parts)
    return np.array(rows, dtype=str) if rows else np.empty((0, 3), dtype=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — SWRL (simulation rdflib, toujours disponible)
# ─────────────────────────────────────────────────────────────────────────────

def part1_swrl():
    print("\n" + "=" * 60)
    print("PART 1 — SWRL Rule-Based Reasoning (rdflib simulation)")
    print("=" * 60)

    from rdflib import Graph, Namespace, Literal, BNode

    print("  Chargement du KB avec rdflib...")
    g = Graph()
    if EXPANDED_NT.exists():
        g.parse(str(EXPANDED_NT), format="nt")
    elif EXPANDED_TTL.exists():
        g.parse(str(EXPANDED_TTL), format="turtle")
    elif KB_TTL.exists():
        g.parse(str(KB_TTL), format="turtle")
    else:
        print("  ⚠  Aucun fichier KB trouvé.")
        return

    print(f"  {len(g)} triples chargés.")

    BASE   = Namespace("http://consulat.ma/kg/")
    WDT    = Namespace("http://www.wikidata.org/prop/direct/")
    SCHEMA = Namespace("http://schema.org/")

    # Index par prédicat (on accepte les prédicats privés ET Wikidata alignés)
    located_in   = defaultdict(set)   # org → lieu
    issued_by    = defaultdict(set)   # doc → org
    requires_map = defaultdict(set)   # proc → doc
    issues_map   = defaultdict(set)   # org → doc (existants)
    submit_map   = defaultdict(set)   # proc → org (existants)

    # On accepte les prédicats privés ET tous les équivalents Wikidata/schema
    LOCATED_PREDS  = {
        BASE.isLocatedIn, BASE.locatedIn,
        WDT.P131,   # located in administrative territorial entity
        WDT.P17,    # country
    }
    ISSUED_PREDS   = {
        BASE.isIssuedBy,
        WDT.P176,   # manufacturer / issuer
        WDT.P749,   # parent organisation (proxy)
    }
    REQUIRES_PREDS = {
        BASE.requires,
        WDT.P4633,  # requirements
        WDT.P361,   # part of (proxy: procedure part-of system)
    }
    # On collecte aussi hasPart / instanceOf / country pour Rule A proxy
    PART_OF_PREDS  = {WDT.P361, WDT.P527, WDT.P150}

    for s, p, o in g:
        if isinstance(o, (Literal, BNode)) or isinstance(s, BNode):
            continue
        if p in LOCATED_PREDS:
            located_in[s].add(o)
            issued_by[o].add(s)   # si org isLocatedIn country → country "issued" par org (proxy Rule A)
        if p in ISSUED_PREDS:
            issued_by[s].add(o)
        if p in REQUIRES_PREDS:
            requires_map[s].add(o)
        if p == BASE.issues:
            issues_map[s].add(o)
        if p == BASE.canBeSubmittedTo:
            submit_map[s].add(o)
        if p in PART_OF_PREDS:
            # proc partOf system → proc "requires" system (proxy)
            requires_map[s].add(o)

    inferred = []

    # ── Règle A ───────────────────────────────────────────────────────────────
    # ConsularOffice(?o) ∧ isLocatedIn(?o,?c) ∧ isIssuedBy(?d,?o) → issues(?o,?d)
    offices_with_location = set(located_in.keys())
    for doc, orgs in issued_by.items():
        for org in orgs:
            if org in offices_with_location:
                if doc not in issues_map.get(org, set()):
                    inferred.append({
                        "rule": "A",
                        "description": "office issues document (inferred)",
                        "subject":   str(org),
                        "predicate": str(BASE.issues),
                        "object":    str(doc),
                    })

    # ── Règle B ───────────────────────────────────────────────────────────────
    # ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)
    for proc, docs in requires_map.items():
        for doc in docs:
            for org in issued_by.get(doc, set()):
                if org not in submit_map.get(proc, set()):
                    inferred.append({
                        "rule": "B",
                        "description": "procedure canBeSubmittedTo office (inferred)",
                        "subject":   str(proc),
                        "predicate": str(BASE.canBeSubmittedTo),
                        "object":    str(org),
                    })

    n_a = sum(1 for r in inferred if r["rule"] == "A")
    n_b = sum(1 for r in inferred if r["rule"] == "B")
    print(f"  Règle A inférée : {n_a} nouveaux triples 'issues'")
    print(f"  Règle B inférée : {n_b} nouveaux triples 'canBeSubmittedTo'")

    # Fichier texte lisible
    lines = [
        "=" * 60,
        "Résultats SWRL — simulation rdflib",
        "=" * 60,
        "",
        "Règle A : ConsularOffice(?o) ∧ isLocatedIn(?o,?c) ∧ isIssuedBy(?d,?o) → issues(?o,?d)",
        "Règle B : ConsularProcedure(?p) ∧ requires(?p,?d) ∧ isIssuedBy(?d,?o) → canBeSubmittedTo(?p,?o)",
        "",
        f"Total triples inférés : {len(inferred)}",
        f"  Règle A : {n_a}",
        f"  Règle B : {n_b}",
        "",
        "Exemples (50 premiers) :",
        "",
    ]
    for r in inferred[:50]:
        lines.append(f"  RULE_{r['rule']}: {r['subject'].split('/')[-1]}"
                     f"  →[{r['predicate'].split('/')[-1]}]→  "
                     f"{r['object'].split('/')[-1]}")

    with open(SWRL_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Sauvegardé → {SWRL_OUT}")

    # Fichier JSON structuré
    swrl_json = OUT_DIR / "swrl_results.json"
    with open(swrl_json, "w", encoding="utf-8") as f:
        json.dump(inferred[:500], f, indent=2, ensure_ascii=False)
    print(f"  ✓ Sauvegardé → {swrl_json}")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES SAUVEGARDÉS
# ─────────────────────────────────────────────────────────────────────────────

def _dims_from_checkpoint(pt_path):
    """Lit les dimensions (n_entities, n_relations, emb_dim) directement
    depuis le state_dict sauvegardé, sans instancier de modèle."""
    import torch
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    n_ent = state["entity_representations.0._embeddings.weight"].shape[0]
    n_rel = state["relation_representations.0._embeddings.weight"].shape[0]
    emb_d = state["entity_representations.0._embeddings.weight"].shape[1]
    return n_ent, n_rel, emb_d


def _make_dummy_factory(n_entities: int, n_relations: int):
    """Crée une TriplesFactory synthétique avec exactement n_entities / n_relations,
    en générant des triples factices couvrant tous les IDs.
    Aucune donnée réelle n'est nécessaire — on l'utilise uniquement pour
    initialiser le modèle avec les bonnes dimensions."""
    import numpy as np
    from pykeen.triples import TriplesFactory

    # Un triple par entité/relation pour garantir la couverture
    n = max(n_entities, n_relations)
    heads     = [str(i % n_entities)  for i in range(n)]
    relations = [str(i % n_relations) for i in range(n)]
    tails     = [str((i + 1) % n_entities) for i in range(n)]

    triples = np.column_stack([heads, relations, tails])
    return TriplesFactory.from_labeled_triples(
        triples=triples,
        create_inverse_triples=False,
    )


def load_saved_models():
    """
    Recharge TransE_full et DistMult_full depuis les .pt sauvegardés
    sans ré-entraîner.

    Stratégie robuste : on lit les dimensions depuis le checkpoint lui-même
    (n_entities, n_relations, emb_dim), on crée une factory synthétique avec
    ces dimensions exactes, on instancie le modèle, puis on charge le state_dict.
    Cela évite tout mismatch entre relation2id.json et le checkpoint.

    Retourne un dict {nom: (model, id_to_ent, id_to_rel)}
    où id_to_ent / id_to_rel sont les vrais mappings reconstruits depuis train.txt.
    """
    import torch
    from pykeen.models import TransE, DistMult

    print("\n  Chargement des modèles sauvegardés...")

    # Charger les vrais mappings (pour l'analyse — pas pour les dimensions du modèle)
    ent2id_full = load_json(OUT_DIR / "entity2id.json")
    rel2id_full = load_json(OUT_DIR / "relation2id.json")
    id_to_ent   = {v: k for k, v in ent2id_full.items()}
    id_to_rel   = {v: k for k, v in rel2id_full.items()}

    loaded = {}

    model_configs = {
        "TransE_full":   (TransE,   OUT_DIR / "model_TransE_full"   / "model.pt"),
        "DistMult_full": (DistMult, OUT_DIR / "model_DistMult_full" / "model.pt"),
        "TransE_20k":    (TransE,   OUT_DIR / "model_TransE_20k"    / "model.pt"),
    }

    for name, (ModelClass, pt_path) in model_configs.items():
        if not pt_path.exists():
            print(f"  ⚠  {pt_path} introuvable — ignoré.")
            continue
        try:
            # 1. Lire les dimensions réelles du checkpoint
            n_ent, n_rel, emb_d = _dims_from_checkpoint(pt_path)
            print(f"  {name}: checkpoint → {n_ent} entités, {n_rel} relations, dim={emb_d}")

            # 2. Créer une factory synthétique aux bonnes dimensions
            dummy_tf = _make_dummy_factory(n_ent, n_rel)

            # 3. Instancier le modèle sur cette factory
            model = ModelClass(
                triples_factory=dummy_tf,
                embedding_dim=emb_d,
            )

            # 4. Charger les poids — garanti compatible maintenant
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            # 5. Construire les mappings id→label cohérents avec ce checkpoint
            #    Pour TransE_full / DistMult_full : n_rel == 221, on utilise
            #    les 221 premières entrées de rel2id_full (triées par valeur)
            sorted_ent = sorted(ent2id_full.items(), key=lambda x: x[1])
            sorted_rel = sorted(rel2id_full.items(), key=lambda x: x[1])
            local_id_to_ent = {i: uri for i, (uri, _) in enumerate(sorted_ent[:n_ent])}
            local_id_to_rel = {i: uri for i, (uri, _) in enumerate(sorted_rel[:n_rel])}
            local_rel_to_id = {v: k for k, v in local_id_to_rel.items()}

            loaded[name] = {
                "model":        model,
                "id_to_ent":    local_id_to_ent,
                "id_to_rel":    local_id_to_rel,
                "rel_to_id":    local_rel_to_id,
                "n_ent":        n_ent,
                "n_rel":        n_rel,
            }
            print(f"  ✓ {name} chargé avec succès")

        except Exception as e:
            print(f"  ⚠  Impossible de charger {name}: {e}")

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — ANALYSE DES EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def part5_analysis(loaded_models: dict):
    print("\n" + "=" * 60)
    print("PART 5 — Embedding Analysis")
    print("=" * 60)

    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not loaded_models:
        print("  ⚠  Aucun modèle chargé.")
        return

    # Utiliser DistMult_full en priorité (meilleurs métriques)
    best_name = "DistMult_full" if "DistMult_full" in loaded_models else next(iter(loaded_models))
    entry     = loaded_models[best_name]
    model     = entry["model"]
    id_to_ent = entry["id_to_ent"]
    id_to_rel = entry["id_to_rel"]
    print(f"  Modèle utilisé : {best_name} ({entry['n_ent']} entités, {entry['n_rel']} relations)")

    # ── Extraire embeddings entités ───────────────────────────────────────────
    with torch.no_grad():
        ent_emb = model.entity_representations[0](indices=None).cpu().numpy()
    print(f"  Shape embeddings entités : {ent_emb.shape}")

    # ── 6.1 Nearest Neighbours ────────────────────────────────────────────────
    print("\n  6.1 Nearest Neighbours")

    # Ancres : pays importants du KB (Wikidata QIDs)
    ANCHORS = {
        "Q1028": "Maroc",
        "Q142":  "France",
        "Q30":   "USA",
        "Q145":  "Royaume-Uni",
        "Q16":   "Canada",
        "Q36":   "Pologne",
        "Q55":   "Pays-Bas",
    }

    sims_matrix = cosine_similarity(ent_emb)
    nn_results = {}

    # Inverser id_to_ent pour la recherche par URI
    ent_to_id = {uri: eid for eid, uri in id_to_ent.items()}

    for qid, label in ANCHORS.items():
        # Chercher l'URI qui contient ce QID
        matches = [(eid, uri) for uri, eid in ent_to_id.items()
                   if f"/{qid}" in uri or uri.endswith(qid)]
        if not matches:
            print(f"    ⚠  {label} ({qid}) non trouvé dans les embeddings")
            continue
        anchor_id, anchor_uri = matches[0]

        sim_row  = sims_matrix[anchor_id]
        top_ids  = sim_row.argsort()[::-1][1:11]
        neighbours = [(id_to_ent[i], float(sim_row[i])) for i in top_ids]

        nn_results[anchor_uri] = {
            "label": label,
            "neighbours": neighbours,
        }
        print(f"\n    Voisins de {label} ({qid}) :")
        for uri, sc in neighbours[:5]:
            print(f"      {uri.split('/')[-1]:<35} cos={sc:.4f}")

    with open(NN_JSON, "w", encoding="utf-8") as f:
        json.dump(nn_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Nearest neighbours → {NN_JSON}")

    # ── 6.2 t-SNE ─────────────────────────────────────────────────────────────
    print("\n  6.2 t-SNE Clustering")

    n_sample = min(3000, ent_emb.shape[0])
    idx = np.random.choice(ent_emb.shape[0], n_sample, replace=False)
    emb_sample = ent_emb[idx]
    uri_sample = [id_to_ent[i] for i in idx]

    def _classify(uri):
        if "wikidata.org/entity/Q" in uri:
            return "Wikidata (pays/lieux)"
        if "wikidata.org" in uri:
            return "Wikidata (autre)"
        if "consulat.ma" in uri:
            return "Consulaire (privé)"
        return "Autre"

    palette = {
        "Wikidata (pays/lieux)": "steelblue",
        "Wikidata (autre)":      "cornflowerblue",
        "Consulaire (privé)":    "tomato",
        "Autre":                 "grey",
    }
    classes = [_classify(u) for u in uri_sample]

    print(f"  t-SNE sur {n_sample} entités (perplexity=30, iter=500)...")
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    tsne_kwargs = dict(n_components=2, random_state=RANDOM_SEED,
                       perplexity=30, learning_rate="auto", init="pca")
    tsne_kwargs["max_iter" if sk_ver >= (1, 2) else "n_iter"] = 500
    tsne = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(emb_sample)

    fig, ax = plt.subplots(figsize=(13, 10))
    for cls_name, color in palette.items():
        idx_cls = [i for i, c in enumerate(classes) if c == cls_name]
        if not idx_cls:
            continue
        ax.scatter(coords[idx_cls, 0], coords[idx_cls, 1],
                   c=color, label=f"{cls_name} (n={len(idx_cls)})",
                   s=8, alpha=0.65, linewidths=0)

    ax.set_title(f"t-SNE — Embeddings entités ({best_name}, n={n_sample})", fontsize=14)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(TSNE_PNG, dpi=150)
    plt.close()
    print(f"  ✓ t-SNE plot → {TSNE_PNG}")

    # ── 6.3 Comportement des relations ────────────────────────────────────────
    print("\n  6.3 Relation Behaviour")

    with torch.no_grad():
        rel_emb = model.relation_representations[0](indices=None).cpu().numpy()

    rel_sims = cosine_similarity(rel_emb)
    norms    = np.linalg.norm(rel_emb, axis=1, keepdims=True) + 1e-9
    rel_norm = rel_emb / norms

    rel_analysis = {}
    for rid, ruri in id_to_rel.items():
        short = ruri.split("/")[-1]
        top_sim_ids = rel_sims[rid].argsort()[::-1][1:6]
        rel_analysis[short] = {
            "uri":                   ruri,
            "top_similar_relations": [id_to_rel[i].split("/")[-1] for i in top_sim_ids],
            "embedding_norm":        float(np.linalg.norm(rel_emb[rid])),
        }

    # Paires inverses candidates (cos ≈ -1 pour TransE)
    inverse_pairs = []
    for i in range(len(id_to_rel)):
        for j in range(i + 1, len(id_to_rel)):
            dot = float(rel_norm[i] @ rel_norm[j])
            if dot < -0.80:
                inverse_pairs.append({
                    "r1":     id_to_rel[i].split("/")[-1],
                    "r2":     id_to_rel[j].split("/")[-1],
                    "cosine": round(dot, 4),
                })

    inverse_pairs.sort(key=lambda x: x["cosine"])

    rel_output = {
        "per_relation":           rel_analysis,
        "candidate_inverse_pairs": inverse_pairs[:30],
    }
    with open(REL_JSON, "w", encoding="utf-8") as f:
        json.dump(rel_output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Relation analysis → {REL_JSON}")

    if inverse_pairs:
        print(f"  Paires inverses candidates : {len(inverse_pairs)}")
        for p in inverse_pairs[:5]:
            print(f"    {p['r1']:<35} ↔  {p['r2']:<35} (cos={p['cosine']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — RÈGLE vs EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def part6_rule_vs_embedding(loaded_models: dict):
    print("\n" + "=" * 60)
    print("PART 6 — Rule vs Embedding Comparison")
    print("=" * 60)

    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    if not loaded_models:
        print("  ⚠  Aucun modèle chargé.")
        return

    # TransE pour la composition vectorielle (h + r ≈ t)
    model_name = "TransE_full" if "TransE_full" in loaded_models else next(iter(loaded_models))
    entry     = loaded_models[model_name]
    model     = entry["model"]
    rel2id    = entry["rel_to_id"]
    id_to_rel = entry["id_to_rel"]
    print(f"  Modèle utilisé : {model_name} ({entry['n_rel']} relations)")

    print("\n  Relations disponibles dans le KB :")
    for r in sorted(rel2id.keys()):
        print(f"    {r.split('/')[-1]}")

    # ── Chercher les relations par nom court ──────────────────────────────────
    def _find(keywords):
        """Cherche un rel_id dont le nom court (après /) contient un des keywords."""
        for kw in keywords:
            for ruri, rid in rel2id.items():
                short = ruri.split("/")[-1].lower()
                if kw.lower() in short:
                    return rid, ruri
        return None, None

    # Les noms réels dans ton KB (d'après la liste affichée lors du run précédent)
    # Règle B : proc requires doc ∧ doc isIssuedBy org → proc canBeSubmittedTo org
    # On va composer : locatedIn + instanceOf comme proxy si les predicats exacts manquent
    req_id, req_uri = _find(["requires", "require"])
    iss_id, iss_uri = _find(["isIssuedBy", "issuedby", "P176"])
    sub_id, sub_uri = _find(["canBeSubmittedTo", "submittedTo", "submitTo"])

    # Fallback : utiliser des prédicats présents (locatedIn + country)
    if req_id is None:
        req_id, req_uri = _find(["locatedIn", "locatedin"])
    if iss_id is None:
        iss_id, iss_uri = _find(["country", "hasPart", "instanceOf"])
    if sub_id is None:
        sub_id, sub_uri = _find(["capital", "headquartersLocation", "containsAdministrativeDivision"])

    if req_id is None or iss_id is None:
        print("  ⚠  Impossible de trouver deux relations à composer.")
        print("  Relations connues :", [r.split("/")[-1] for r in sorted(rel2id.keys())])
        return

    with torch.no_grad():
        rel_emb = model.relation_representations[0](indices=None).cpu().numpy()

    v1 = rel_emb[req_id]
    v2 = rel_emb[iss_id]
    composed = v1 + v2

    r1_name = req_uri.split("/")[-1]
    r2_name = iss_uri.split("/")[-1]
    print(f"\n  Composition : v({r1_name}) + v({r2_name})")

    # Trouver la relation la plus proche du vecteur composé
    all_sims = cosine_similarity([composed], rel_emb)[0]
    top_ids  = all_sims.argsort()[::-1][:8]

    print(f"  Relations les plus proches du vecteur composé :")
    for i in top_ids:
        marker = " ← cible" if sub_id is not None and i == sub_id else ""
        print(f"    {id_to_rel[i].split('/')[-1]:<40} cos={all_sims[i]:.4f}{marker}")

    target_sim = None
    if sub_id is not None:
        target_sim = float(all_sims[sub_id])
        r3_name = sub_uri.split("/")[-1]
        print(f"\n  Similarité cosinus composé → {r3_name} : {target_sim:.4f}")
        if target_sim > 0.70:
            print("  → Forte correspondance ✅ — le modèle capture la composition.")
        elif target_sim > 0.40:
            print("  → Correspondance modérée ⚠  — structure partielle apprise.")
        else:
            print("  → Faible correspondance ❌ — composition non capturée.")

    # Règle SWRL associée
    rule_text = (
        f"SWRL Rule B (approx) :\n"
        f"  {r1_name}(?a,?b) ∧ {r2_name}(?b,?c) → {sub_uri.split('/')[-1] if sub_uri else '?'}(?a,?c)\n"
        f"Analogie vectorielle TransE :\n"
        f"  v({r1_name}) + v({r2_name}) ≈ v({sub_uri.split('/')[-1] if sub_uri else '?'})"
    )
    print(f"\n  {rule_text}")

    output = {
        "model_used":           model_name,
        "r1_uri":               req_uri,
        "r2_uri":               iss_uri,
        "target_uri":           sub_uri,
        "cosine_composed_target": target_sim,
        "top_nearest_relations": [
            {"relation": id_to_rel[i].split("/")[-1], "cosine": float(all_sims[i])}
            for i in top_ids
        ],
        "swrl_rule_analog": rule_text,
    }
    out_path = OUT_DIR / "rule_vs_embedding.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Sauvegardé → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("lab5_fix.py — Complétion des parties manquantes")
    print("=" * 60)

    # Vérifications préalables
    missing = [p for p in [TRAIN_TXT, OUT_DIR / "entity2id.json",
                            OUT_DIR / "relation2id.json"] if not p.exists()]
    if missing:
        print(f"⚠  Fichiers manquants : {missing}")
        print("   Relancez d'abord lab5_pipeline.py (Parts 1-4).")
        return

    # ── Part 1 : SWRL ─────────────────────────────────────────────────────────
    part1_swrl()

    # ── Charger modèles sauvegardés ───────────────────────────────────────────
    try:
        import torch
        from pykeen.triples import TriplesFactory
    except ImportError:
        print("⚠  PyTorch ou PyKEEN non disponibles.")
        return

    loaded_models = load_saved_models()

    if not loaded_models:
        print("⚠  Aucun modèle chargé. Vérifiez data5/model_*/model.pt")
        return

    # ── Part 5 : Analyse ──────────────────────────────────────────────────────
    try:
        import matplotlib
        part5_analysis(loaded_models)
    except ImportError:
        print("⚠  matplotlib manquant : pip install matplotlib")
        print("   Part 5 ignorée.")

    # ── Part 6 : Règle vs Embedding ───────────────────────────────────────────
    part6_rule_vs_embedding(loaded_models)

    # ── Récap final ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅  Fichiers produits :")
    print("=" * 60)
    for p in [
        OUT_DIR / "swrl_results.txt",
        OUT_DIR / "swrl_results.json",
        NN_JSON,
        TSNE_PNG,
        REL_JSON,
        OUT_DIR / "rule_vs_embedding.json",
    ]:
        status = "✅" if p.exists() else "⬜"
        print(f"  {status}  {p}")


if __name__ == "__main__":
    main()