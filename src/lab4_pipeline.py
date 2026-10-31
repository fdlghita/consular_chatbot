"""
Lab Session — Knowledge Base Construction, Alignment, and Expansion
====================================================================
Reads the outputs of Lab 1 directly:
  - crawler_output.jsonl          (raw pages)
  - extracted_knowledge.csv       (NER entities)
  - extracted_relations.csv       (relations)

Produces:
  Step 1 → knowledge_graph.ttl              (private RDF KB)
  Step 2 → alignment_table.csv             (entity → Wikidata QID + confidence)
           alignment.ttl                   (owl:sameAs triples)
           new_entities.json               (entities not found in Wikidata)
  Step 3 → predicate_alignment_candidates.csv  (for manual review)
           predicate_alignment.ttl         (owl:equivalentProperty triples)
  Step 4 → expanded_kb.nt                  (50k–200k triples, for KGE)
           expanded_kb.ttl                 (human-readable version)
           ontology.ttl                    (ontology file)
           statistics_report.json          (counts: triples, entities, relations)

Requirements:
  pip install rdflib requests
"""

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from rdflib import (
    Graph, Namespace, URIRef, Literal,
    RDF, RDFS, OWL, XSD, BNode
)
from rdflib.namespace import SKOS

# ─────────────────────────────────────────────────────────────────────────────
# NAMESPACES
# ─────────────────────────────────────────────────────────────────────────────

BASE   = Namespace("http://consulat.ma/kg/")
SCHEMA = Namespace("http://schema.org/")
WD     = Namespace("http://www.wikidata.org/entity/")
WDT    = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_SEARCH  = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL  = "https://query.wikidata.org/sparql"

# ─────────────────────────────────────────────────────────────────────────────
# INPUT / OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_ENT_CSV   = Path("data/extracted_knowledge.csv")
INPUT_REL_CSV   = Path("data/extracted_relations.csv")

OUT_TTL         = Path("data2/knowledge_graph.ttl")
OUT_ALIGN_CSV   = Path("data2/alignment_table.csv")
OUT_ALIGN_TTL   = Path("data2/alignment.ttl")
OUT_NEW_ENT     = Path("data2/new_entities.json")
OUT_PRED_CAND   = Path("data2/predicate_alignment_candidates.csv")
OUT_PRED_TTL    = Path("data2/predicate_alignment.ttl")
OUT_EXP_NT      = Path("data2/expanded_kb.nt")
OUT_EXP_TTL     = Path("data2/expanded_kb.ttl")
OUT_ONTO_TTL    = Path("data2/ontology.ttl")
OUT_STATS       = Path("data2/statistics_report.json")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert a French label to a safe URI local name."""
    replacements = {
        'à':'a','á':'a','â':'a','ä':'a',
        'è':'e','é':'e','ê':'e','ë':'e',
        'ì':'i','î':'i','ï':'i',
        'ò':'o','ô':'o','ö':'o',
        'ù':'u','û':'u','ü':'u',
        'ç':'c','ñ':'n',
    }
    for s, d in replacements.items():
        text = text.replace(s, d).replace(s.upper(), d.upper())
    text = re.sub(r"['\u2019\u2018]", '', text)
    text = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text[:80]


LABEL_TO_CLASS = {
    'ORG':         SCHEMA.GovernmentOrganization,
    'GPE':         SCHEMA.Place,
    'DOCUMENT':    BASE.AdministrativeDocument,
    'PERSON_ROLE': BASE.AdministrativeRole,
    'PROCEDURE':   BASE.ConsularProcedure,
    'LAW':         BASE.LegalText,
    'FEE':         BASE.AdministrativeFee,
    'DURATION':    BASE.Duration,
    # spaCy standard labels
    'PERSON':      SCHEMA.Person,
    'LOC':         SCHEMA.Place,
    'DATE':        BASE.Duration,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD PRIVATE RDF KB
# ─────────────────────────────────────────────────────────────────────────────

def step1_build_kb() -> Graph:
    """
    Load extracted_knowledge.csv + extracted_relations.csv
    and serialise as a valid RDF/Turtle knowledge graph.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Building Private RDF Knowledge Base")
    print("=" * 60)

    # Load entities
    entities = []
    with open(INPUT_ENT_CSV, encoding="utf-8") as f:
        entities = list(csv.DictReader(f))
    print(f"  Entities loaded : {len(entities)}")

    # Load relations
    relations = []
    with open(INPUT_REL_CSV, encoding="utf-8") as f:
        relations = list(csv.DictReader(f))
    print(f"  Relations loaded: {len(relations)}")

    g = Graph()
    g.bind('',      BASE)
    g.bind('schema', SCHEMA)
    g.bind('owl',   OWL)
    g.bind('skos',  SKOS)
    g.bind('xsd',   XSD)

    # Ontology declaration
    onto = BASE['ontology']
    g.add((onto, RDF.type,       OWL.Ontology))
    g.add((onto, RDFS.label,     Literal("Moroccan Consular Services Knowledge Graph")))
    g.add((onto, RDFS.comment,   Literal("Entities and relations extracted from consulat.ma")))
    g.add((onto, OWL.versionInfo, Literal("1.0")))

    # Custom class declarations
    custom_classes = {
        BASE.AdministrativeDocument: "An official document issued by consular authorities",
        BASE.AdministrativeRole:     "A named administrative role or person category",
        BASE.ConsularProcedure:      "An administrative procedure at consular level",
        BASE.LegalText:              "A law, decree, circular, or legal reference",
        BASE.AdministrativeFee:      "A fee associated with a consular service",
        BASE.Duration:               "A time period or validity duration",
    }
    for cls, comment in custom_classes.items():
        g.add((cls, RDF.type,        OWL.Class))
        g.add((cls, RDFS.subClassOf, OWL.Thing))
        g.add((cls, RDFS.comment,    Literal(comment)))

    # Property declarations
    props = {
        'isIssuedBy':          "Links a document to the authority that issues it",
        'issues':              "Links an authority to a document it issues",
        'isEstablishedBy':     "Links an entity to its establishing authority",
        'hasJurisdictionOver': "Links a court to its area of competence",
        'isResponsibleFor':    "Links an organization to the service it manages",
        'transmitsTo':         "Links a body that transmits documents to a receiver",
        'reportsTo':           "Links an entity to its supervisory authority",
        'dependsOn':           "Links an entity to another it depends on",
        'isAttachedTo':        "Links an entity to its parent body",
        'isLocatedIn':         "Links an organization to its geographic location",
        'isAvailableAt':       "Links a service to where it can be obtained",
        'represents':          "Links an entity that represents another",
        'processes':           "Links an authority to a procedure it handles",
        'canBeSubmittedTo':    "Links a request to where it can be submitted",
        'submittedTo':         "Links a procedure to the authority it goes to",
        'requestedFrom':       "Links a procedure to where it must be requested",
        'hasEstablished':      "Links an authority to something it set up",
        'enables':             "Links a tool to what it enables",
        'requires':            "Links a procedure to a required document",
        'relatedTo':           "Generic relation between two co-occurring entities",
    }
    for name, comment in props.items():
        g.add((BASE[name], RDF.type,    OWL.ObjectProperty))
        g.add((BASE[name], RDFS.comment, Literal(comment)))

    # ── Noise filters ────────────────────────────────────────────────────────
    # Labels to exclude entirely from the private KB
    _NOISE_LABELS = {
        'ORG': set(),   # all ORG kept
        'GPE': set(),   # all GPE kept
    }

    # Patterns for entity text that should not enter the KB
    _NOISE_TEXT_PATTERN = re.compile(
        r'^(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|'
        r'fermer|ouvrir|cliquer|appeler|demander|consulter|valider|'
        r'accepter|envoyer|imprimer|cette|cet|ces|noel|noël|'
        r'prix de l|mentions légales|choisissez|portail|'
        r'droits de chancellerie|procédure formulaires|'
        r'formulaires droits|observations droits|'
        r'présentation procédure|procédure droits|'
        r'préstations|formulaires observations|'
        r'copie de la|copie du|copie intégrale).*',
        re.IGNORECASE
    )

    # Better type mapping: use spaCy label + entity text heuristics
    def _get_class(entity_text: str, label: str):
        """Return the most appropriate OWL class for an entity."""
        el = entity_text.strip().lower()
        # Social media / tech companies → use Organization but not GovernmentOrganization
        social_media = {
            'skype','telegram','viber','whatsapp','reddit','twitter','facebook',
            'instagram','linkedin','youtube','tiktok','snapchat','wechat',
            'kakao','kik','plurk','weibo','odnoklassniki','viadeo','renren',
            'tumblr','pinterest','myspace','wordpress','delicious','slashdot',
            'blinklist','symbaloo','hatena','protopage','svejo','fark',
            'diigo','draugiem','mendeley','paperli','yoolink','netvouz',
            'twiddla','papaly','pusha','sina','wykop','wanelo','care2',
            'balatarin','livejounal','livejournal','aol','aol mail',
            'yahoo','yahoo mail','yahoo messenger','google bookmarks',
            'google maps','paypal','american express','amazon','reddit',
            'tumblr','slashdot'
        }
        if el in social_media:
            return SCHEMA.Organization

        # Procedures / documents
        if label in ('DOCUMENT', 'PROCEDURE'):
            return LABEL_TO_CLASS.get(label, BASE.ConsularProcedure)

        # Default
        return LABEL_TO_CLASS.get(label, OWL.Thing)

    # Entity nodes — with noise filtering
    entity_registry = {}  # lower_text -> URIRef
    for e in entities:
        ent_text = e['entity'].strip()
        key      = ent_text.lower()
        label    = e.get('label', '')

        if key in entity_registry:
            continue
        # Skip noise
        if _NOISE_TEXT_PATTERN.match(ent_text):
            continue

        uri = BASE[slugify(ent_text)]
        cls = _get_class(ent_text, label)
        entity_registry[key] = uri
        g.add((uri, RDF.type,   cls))
        g.add((uri, RDFS.label, Literal(ent_text, lang='fr')))

    # Triple assertions — only keep clean semantic predicates
    CLEAN_PREDICATES = {
        'isIssuedBy', 'issues', 'isEstablishedBy', 'hasJurisdictionOver',
        'isResponsibleFor', 'transmitsTo', 'reportsTo', 'dependsOn',
        'isAttachedTo', 'isLocatedIn', 'isAvailableAt', 'represents',
        'processes', 'canBeSubmittedTo', 'submittedTo', 'requestedFrom',
        'hasEstablished', 'enables', 'requires', 'relatedTo',
        'related_to',  # spaCy may produce this variant
    }

    pred_field = 'predicate' if 'predicate' in (relations[0] if relations else {}) else 'relation'
    added = set()
    for r in relations:
        sk = r.get('subject', '').strip().lower()
        ok = r.get('object',  '').strip().lower()
        raw_pred = r.get(pred_field, '').strip()

        if sk not in entity_registry or ok not in entity_registry:
            continue
        if not raw_pred:
            continue

        # Normalise predicate: only accept clean known predicates
        # or short verb phrases (≤ 4 words) — reject long noisy dep chains
        pred_slug = slugify(raw_pred)
        word_count = len(raw_pred.split())
        if raw_pred not in CLEAN_PREDICATES and word_count > 4:
            pred_uri = BASE['relatedTo']   # demote noisy relations to relatedTo
        else:
            pred_uri = BASE[pred_slug] if raw_pred else BASE['relatedTo']

        subj_uri = entity_registry[sk]
        obj_uri  = entity_registry[ok]

        triple = (subj_uri, pred_uri, obj_uri)
        if triple not in added:
            added.add(triple)
            g.add(triple)

    g.serialize(destination=str(OUT_TTL), format='turtle')
    print(f"\n  ✓ {OUT_TTL}  ({len(g)} triples, {len(entity_registry)} entities)")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENTITY LINKING WITH WIKIDATA
# ─────────────────────────────────────────────────────────────────────────────

# Entities that are clearly not in Wikidata — skip them entirely to save quota
_SKIP_PATTERNS = re.compile(
    r'(formulaire|pièce|copie|droits de|présentation|procédure|'
    r'mentions légales|choisissez|portail|hatena|houzz|svejo|fark|'
    r'draugiem|diigo|protopage|mendeley|myspace|wordpress|wechat|'
    r'préstations|istilhaq|droit de chancellerie|livret d|'
    r'prix de l|biais des|centre d.appels)',
    re.IGNORECASE
)

# Days of week, verbs, pronouns — meaningless to link to Wikidata
_FUNCTIONAL_WORDS = {
    'lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche',
    'fermer','ouvrir','cliquer','appeler','demander','consulter','valider',
    'cette','cet','ces','celui','celle','ceux',
    'accepter','envoyer','imprimer','noel','noël',
}

# Descriptions indicating a French/Belgian commune or village — out-of-domain
_COMMUNE_PATTERN = re.compile(
    r'\b(commune|municipality of france|village in france|commune française|'
    r'arrondissement|hameau|lieu-dit|chef-lieu)\b',
    re.IGNORECASE
)

def _wikidata_search(label: str, lang: str = 'fr', retries: int = 3) -> list:
    """Query Wikidata with exponential backoff on 429."""
    params = {
        'action':   'wbsearchentities',
        'search':   label,
        'language': lang,
        'format':   'json',
        'limit':    '3',
    }
    url = WIKIDATA_SEARCH + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={'User-Agent': 'KBLabPipeline/1.0 (research)'}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            results = []
            for i, item in enumerate(data.get('search', [])):
                score = round(0.95 - i * 0.15, 2)
                results.append({
                    'qid':         item.get('id', ''),
                    'wiki_label':  item.get('label', ''),
                    'description': item.get('description', ''),
                    'rank_score':  max(score, 0.40),
                })
            return results
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                print(f"    [429] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [WARN] HTTP {e.code} for '{label}'")
                return []
        except Exception as e:
            print(f"    [WARN] Wikidata search failed for '{label}': {e}")
            return []
    return []


# Keywords in Wikidata descriptions that indicate a wrong entity type
# If these appear in the description, confidence is heavily penalised
_BAD_DESCRIPTION_KEYWORDS = re.compile(
    r'\b(album|chanson|song|film|movie|horse|cheval|trotteur|roman|novel|'
    r'bande dessinée|comic|série|series|single|ep|discography|discographie|'
    r'video game|jeu vidéo|painting|tableau|artwork|sculpture|'
    r'race hippique|horse race|thoroughbred|jockey|'
    r'given name|family name|surname|forename|prénom|nom de famille|'
    r'masculine name|feminine name|human name|'
    r'architect|architecte|painter|peintre|sculptor|'
    r'footballer|rugby|tennis player|cyclist|athlete|sportsman|'
    r'fictional character|personnage fictif|commune française|'
    r'municipality of france|village in france|arrondissement)\b',
    re.IGNORECASE
)

# Keywords that indicate a GOOD match for our domain (consular/admin entities)
_GOOD_DESCRIPTION_KEYWORDS = re.compile(
    r'\b(pays|country|état|state|ville|city|commune|région|region|province|'
    r'municipalité|municipality|nation|republic|royaume|kingdom|'
    r'ministère|ministry|organisme|organization|institution|'
    r'document|acte|certificat|procédure|service|administration|'
    r'passeport|passport|visa|nationalité|nationality|mariage|marriage|'
    r'naissance|birth|décès|death|divorce|consulat|embassy|ambassade)\b',
    re.IGNORECASE
)

# Expected Wikidata entity types per private label type
# Maps label → list of acceptable Wikidata description keywords
_EXPECTED_TYPE_BY_LABEL = {
    'GPE':     _GOOD_DESCRIPTION_KEYWORDS,  # countries, cities
    'ORG':     _GOOD_DESCRIPTION_KEYWORDS,  # ministries, organisations
    'PERSON':  None,                         # any person is fine
}


def _confidence(private_label: str, wiki_label: str,
                wiki_description: str, rank_score: float,
                entity_label: str = '') -> float:
    """
    confidence = 0.5 × rank_score
               + 0.3 × string_similarity
               + 0.2 × type_bonus
               - penalty if description suggests wrong entity type

    The type_bonus rewards descriptions that match our domain vocabulary.
    The penalty strongly discounts albums, films, horses, etc.
    """
    pl = re.sub(r'\s+', ' ', private_label.lower().strip())
    wl = re.sub(r'\s+', ' ', wiki_label.lower().strip())

    # String similarity
    if pl == wl:
        str_sim = 1.0
    elif pl in wl or wl in pl:
        str_sim = min(len(pl), len(wl)) / max(len(pl), len(wl))
    else:
        tp = set(pl.split()); tw = set(wl.split())
        str_sim = len(tp & tw) / max(len(tp | tw), 1)

    # Type bonus: +0.2 if description matches our domain
    desc = wiki_description.lower()
    type_bonus = 0.2 if _GOOD_DESCRIPTION_KEYWORDS.search(desc) else 0.0

    # Hard penalty: -0.5 if description clearly indicates wrong type (album, horse, commune...)
    penalty = 0.5 if (_BAD_DESCRIPTION_KEYWORDS.search(desc) or _COMMUNE_PATTERN.search(desc)) else 0.0

    raw = 0.5 * rank_score + 0.3 * str_sim + type_bonus - penalty
    return round(min(max(raw, 0.0), 0.99), 3)


def step2_entity_linking():
    """
    For each ORG/GPE entity in extracted_knowledge.csv:
      - Query Wikidata Entity Search API
      - If confidence >= 0.60 → owl:sameAs link
      - Otherwise → define semantically (new entity)

    Deliverable: alignment_table.csv
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Entity Linking with Wikidata")
    print("=" * 60)

    THRESHOLD = 0.60

    entities = []
    with open(INPUT_ENT_CSV, encoding="utf-8") as f:
        entities = list(csv.DictReader(f))

    # Deduplicate: only unique (entity, label) pairs
    seen = set()
    unique = []
    for e in entities:
        key = (e['entity'].strip().lower(), e.get('label', ''))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    print(f"  Unique entities to link: {len(unique)}")

    alignment_rows  = []
    new_entity_rows = []

    for i, e in enumerate(unique):
        ent_text = e['entity'].strip()
        label    = e.get('label', '')
        uri      = str(BASE[slugify(ent_text)])

        # Skip non-linkable types entirely
        if label not in ('ORG', 'GPE', 'PERSON'):
            new_entity_rows.append({
                'private_uri':   uri,
                'private_label': ent_text,
                'entity_label':  label,
                'reason':        f'Type {label} defined locally',
            })
            continue

        # Skip noisy boilerplate entities (social media, form labels, etc.)
        if _SKIP_PATTERNS.search(ent_text):
            new_entity_rows.append({
                'private_uri':   uri,
                'private_label': ent_text,
                'entity_label':  label,
                'reason':        'Boilerplate entity — skipped',
            })
            continue

        # Skip functional words (days, verbs, pronouns)
        if ent_text.strip().lower() in _FUNCTIONAL_WORDS:
            new_entity_rows.append({
                'private_uri':   uri,
                'private_label': ent_text,
                'entity_label':  label,
                'reason':        'Functional word — not linkable',
            })
            continue

        # Also skip very long strings (unlikely to match anything in Wikidata)
        if len(ent_text) > 60:
            new_entity_rows.append({
                'private_uri':   uri,
                'private_label': ent_text,
                'entity_label':  label,
                'reason':        'Label too long for entity search',
            })
            continue

        results = _wikidata_search(ent_text, lang='fr')
        # Only try English fallback if French returned nothing (not on 429)
        if not results and not _SKIP_PATTERNS.search(ent_text):
            results = _wikidata_search(ent_text, lang='en')

        matched = False
        for res in results:
            conf = _confidence(ent_text, res['wiki_label'],
                               res['description'], res['rank_score'],
                               entity_label=label)
            if conf >= THRESHOLD:
                alignment_rows.append({
                    'private_entity':   ent_text,
                    'private_uri':      uri,
                    'wikidata_qid':     res['qid'],
                    'wikidata_uri':     f"http://www.wikidata.org/entity/{res['qid']}",
                    'wiki_label':       res['wiki_label'],
                    'wiki_description': res['description'],
                    'confidence':       conf,
                })
                matched = True
                break

        if not matched:
            new_entity_rows.append({
                'private_uri':   uri,
                'private_label': ent_text,
                'entity_label':  label,
                'reason':        'No Wikidata match above threshold',
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(unique)}] matched={len(alignment_rows)}")
        time.sleep(1.5)   # 1.5s between calls — stays safely under Wikidata's limit

    # Save alignment_table.csv
    with open(OUT_ALIGN_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'private_entity','private_uri','wikidata_qid',
            'wikidata_uri','wiki_label','wiki_description','confidence'
        ])
        writer.writeheader()
        writer.writerows(alignment_rows)

    # Save new_entities.json
    with open(OUT_NEW_ENT, 'w', encoding='utf-8') as f:
        json.dump(new_entity_rows, f, ensure_ascii=False, indent=2)

    # Build alignment.ttl (owl:sameAs + new entity OWL definitions)
    g = Graph()
    g.bind('',    BASE)
    g.bind('wd',  WD)
    g.bind('owl', OWL)
    g.bind('skos', SKOS)
    g.bind('schema', SCHEMA)

    for row in alignment_rows:
        private = URIRef(row['private_uri'])
        wiki    = URIRef(row['wikidata_uri'])
        g.add((private, OWL.sameAs,      wiki))
        g.add((private, SKOS.exactMatch, wiki))
        # Confidence annotation
        node = BASE['conf_' + slugify(row['private_entity'])]
        g.add((node, RDF.type,             BASE.AlignmentConfidence))
        g.add((node, BASE.privateEntity,   private))
        g.add((node, BASE.externalEntity,  wiki))
        g.add((node, BASE.confidenceScore,
               Literal(row['confidence'], datatype=XSD.decimal)))

    type_map = {
        'DOCUMENT':    BASE.AdministrativeDocument,
        'PERSON_ROLE': BASE.AdministrativeRole,
        'PROCEDURE':   BASE.ConsularProcedure,
        'LAW':         BASE.LegalText,
        'FEE':         BASE.AdministrativeFee,
        'DURATION':    BASE.Duration,
    }
    for row in new_entity_rows:
        uri = URIRef(row['private_uri'])
        cls = type_map.get(row['entity_label'], OWL.Thing)
        g.add((uri, RDF.type,    cls))
        g.add((uri, RDFS.label,  Literal(row['private_label'], lang='fr')))
        g.add((uri, RDFS.comment, Literal(f"Defined locally: {row['reason']}")))

    g.serialize(destination=str(OUT_ALIGN_TTL), format='turtle')

    print(f"\n  ✓ {OUT_ALIGN_CSV}     ({len(alignment_rows)} matched)")
    print(f"  ✓ {OUT_ALIGN_TTL}      ({len(g)} triples)")
    print(f"  ✓ {OUT_NEW_ENT} ({len(new_entity_rows)} locally defined)")

    # Print mapping table sample
    print("\n  Sample alignment table:")
    print(f"  {'Private Entity':<30} {'QID':<12} {'Conf':<7} {'Wikidata description'}")
    print("  " + "-" * 85)
    for row in alignment_rows[:20]:
        desc = row.get('wiki_description','')[:38]
        print(f"  {row['private_entity']:<30} {row['wikidata_qid']:<12} {row['confidence']:<7} {desc}")

    return alignment_rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREDICATE ALIGNMENT VIA SPARQL
# ─────────────────────────────────────────────────────────────────────────────

def _sparql_query(query: str, timeout: int = 30) -> list:
    params = {'query': query, 'format': 'json'}
    url = WIKIDATA_SPARQL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'KBLabPipeline/1.0 (research)',
            'Accept':     'application/sparql-results+json',
        }
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            # Strip invalid control characters before JSON parsing
            text = raw_bytes.decode('utf-8', errors='replace')
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
            data = json.loads(text)
            return data.get('results', {}).get('bindings', [])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)   # 10s, 20s, 30s, 40s
                print(f"    [429] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [WARN] HTTP {e.code} on SPARQL query")
                return []
        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse error (attempt {attempt+1}): {e}")
            if attempt < 3:
                time.sleep(5)
            else:
                return []
        except Exception as e:
            print(f"    [WARN] SPARQL error: {e}")
            return []
    return []


# Pre-validated predicate alignments
# (result of running the SPARQL keyword search + manual review)
VALIDATED_ALIGNMENTS = {
    'isIssuedBy':          ('P176',  'equivalentProperty'),
    'isLocatedIn':         ('P131',  'equivalentProperty'),
    'reportsTo':           ('P749',  'equivalentProperty'),
    'isAttachedTo':        ('P361',  'subPropertyOf'),
    'represents':          ('P1001', 'subPropertyOf'),
    'hasJurisdictionOver': ('P1001', 'equivalentProperty'),
    'requires':            ('P4633', 'subPropertyOf'),
    'transmitsTo':         ('P527',  'subPropertyOf'),
    'isEstablishedBy':     ('P571',  'subPropertyOf'),
    'relatedTo':           ('P1659', 'equivalentProperty'),
}

PREDICATE_KEYWORDS = {
    'isIssuedBy':          'issued',
    'issues':              'issue',
    'isEstablishedBy':     'founded',
    'hasJurisdictionOver': 'jurisdiction',
    'isResponsibleFor':    'responsible',
    'transmitsTo':         'transmit',
    'reportsTo':           'parent organization',
    'isAttachedTo':        'part of',
    'isLocatedIn':         'located',
    'represents':          'represents',
    'requires':            'requires',
    'relatedTo':           'related',
}


def step3_predicate_alignment(alignment_rows: list):
    """
    Phase A: Search Wikidata properties by label keyword (SPARQL)
             → predicate_alignment_candidates.csv  (for manual review)
    Phase B: For aligned entity pairs, query actual connecting properties
    Phase C: Write validated owl:equivalentProperty / rdfs:subPropertyOf triples
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Predicate Alignment via SPARQL")
    print("=" * 60)

    # ── Phase A: keyword search ────────────────────────────────────────────────
    print("\n  Phase A: SPARQL keyword search for each private predicate...")
    candidate_rows = []

    for pred_name, keyword in PREDICATE_KEYWORDS.items():
        print(f"    [{pred_name}] searching '{keyword}'...")
        query = f"""
SELECT ?property ?propertyLabel ?propertyDescription WHERE {{
  ?property a wikibase:Property .
  ?property rdfs:label ?propertyLabel .
  OPTIONAL {{ ?property schema:description ?propertyDescription .
              FILTER(LANG(?propertyDescription) = "en") }}
  FILTER(CONTAINS(LCASE(?propertyLabel), "{keyword.lower()}"))
  FILTER(LANG(?propertyLabel) = "en")
}}
LIMIT 5
"""
        rows = _sparql_query(query)
        for row in rows:
            pid   = row['property']['value'].split('/')[-1]
            label = row['propertyLabel']['value']
            desc  = row.get('propertyDescription', {}).get('value', '')
            candidate_rows.append({
                'private_predicate':    pred_name,
                'search_keyword':       keyword,
                'wikidata_pid':         pid,
                'wikidata_label':       label,
                'wikidata_description': desc,
                'validated_alignment':  VALIDATED_ALIGNMENTS.get(pred_name, ('',''))[0],
                'alignment_type':       VALIDATED_ALIGNMENTS.get(pred_name, ('',''))[1],
            })
        time.sleep(0.5)

    with open(OUT_PRED_CAND, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'private_predicate','search_keyword','wikidata_pid',
            'wikidata_label','wikidata_description',
            'validated_alignment','alignment_type'
        ])
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(f"  ✓ {OUT_PRED_CAND} ({len(candidate_rows)} candidates)")

    # ── Phase B: entity-pair predicate search ──────────────────────────────────
    print("\n  Phase B: entity-pair predicate search (wiki(s) ?p wiki(o))...")
    qid_map = {row['private_entity'].strip().lower(): row['wikidata_qid']
               for row in alignment_rows if row.get('wikidata_qid')}

    relations = []
    with open(INPUT_REL_CSV, encoding='utf-8') as f:
        relations = list(csv.DictReader(f))

    checked = set()
    pair_results = []
    for rel in relations[:50]:   # limit to first 50 to avoid rate limiting
        sk = rel.get('subject','').strip().lower()
        ok = rel.get('object','').strip().lower()
        if sk in qid_map and ok in qid_map:
            pair = (qid_map[sk], qid_map[ok])
            if pair in checked:
                continue
            checked.add(pair)
            s_qid, o_qid = pair
            query = f"""
SELECT DISTINCT ?property ?propertyLabel WHERE {{
  wd:{s_qid} ?property wd:{o_qid} .
  ?prop wikibase:directClaim ?property .
  ?prop rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
}}
LIMIT 10
"""
            rows = _sparql_query(query)
            for row in rows:
                pid   = row['property']['value'].split('/')[-1]
                label = row['propertyLabel']['value']
                pair_results.append({
                    'private_subject': rel.get('subject',''),
                    'private_pred':    rel.get('predicate', rel.get('relation','')),
                    'private_object':  rel.get('object',''),
                    'wiki_s':          s_qid,
                    'wiki_o':          o_qid,
                    'wikidata_pid':    pid,
                    'wikidata_label':  label,
                })
            time.sleep(0.5)

    if pair_results:
        pair_csv = Path("predicate_entity_pair_candidates.csv")
        with open(pair_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_results[0].keys()))
            writer.writeheader()
            writer.writerows(pair_results)
        print(f"  ✓ predicate_entity_pair_candidates.csv ({len(pair_results)} rows)")

    # ── Phase C: write validated alignment TTL ─────────────────────────────────
    print("\n  Phase C: writing predicate_alignment.ttl...")
    g = Graph()
    g.bind('',    BASE)
    g.bind('wdt', WDT)
    g.bind('owl', OWL)
    g.bind('rdfs', RDFS)

    for pred_name, (pid, align_type) in VALIDATED_ALIGNMENTS.items():
        private_prop  = BASE[pred_name]
        wikidata_prop = WDT[pid]
        g.add((private_prop, RDF.type, OWL.ObjectProperty))
        if align_type == 'equivalentProperty':
            g.add((private_prop, OWL.equivalentProperty, wikidata_prop))
        else:
            g.add((private_prop, RDFS.subPropertyOf, wikidata_prop))

    g.serialize(destination=str(OUT_PRED_TTL), format='turtle')
    print(f"  ✓ {OUT_PRED_TTL} ({len(g)} triples)")

    print("\n  Validated alignments:")
    print(f"  {'Private Predicate':<30} {'Type':<22} {'Wikidata PID'}")
    print("  " + "-" * 65)
    for name, (pid, atype) in VALIDATED_ALIGNMENTS.items():
        print(f"  {name:<30} {atype:<22} wdt:{pid}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — KB EXPANSION VIA SPARQL
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_PIDS = {
    'P31':   'instanceOf',
    'P279':  'subclassOf',
    'P361':  'partOf',
    'P17':   'country',
    'P131':  'locatedIn',
    'P159':  'headquartersLocation',
    'P749':  'parentOrganization',
    'P527':  'hasPart',
    'P1001': 'appliesTo',
    'P150':  'containsAdministrativeDivision',
    'P36':   'capital',
    'P37':   'officialLanguage',
    'P571':  'inceptionDate',
    'P576':  'dissolvedDate',
}
DROP_PIDS = {'P18', 'P625', 'P856', 'P1566', 'P297', 'P298'}


def _add_triples(g: Graph, raw: list) -> int:
    added = 0
    for t in raw:
        s_raw = t.get('subject',   t.get('s', {}).get('value', ''))
        p_raw = t.get('predicate', t.get('p', {}).get('value', ''))
        o_raw = t.get('object',    t.get('o', {}).get('value', ''))

        if isinstance(s_raw, dict): s_raw = s_raw.get('value','')
        if isinstance(p_raw, dict): p_raw = p_raw.get('value','')
        if isinstance(o_raw, dict): o_raw = o_raw.get('value','')

        if '/entity/Q' not in s_raw or '/entity/Q' not in o_raw:
            continue

        s_qid = s_raw.split('/')[-1]
        o_qid = o_raw.split('/')[-1]
        p_pid = p_raw.split('/')[-1]

        if p_pid in DROP_PIDS:
            continue

        s_uri = BASE[s_qid]
        o_uri = BASE[o_qid]
        p_name = ALLOWED_PIDS.get(p_pid, p_pid)
        p_uri  = BASE[p_name] if p_pid in ALLOWED_PIDS else WDT[p_pid]

        triple = (s_uri, p_uri, o_uri)
        if triple not in g:
            g.add(triple)
            # Add labels if provided
            for lkey in ('sLabel', 'subj_label'):
                lv = t.get(lkey, '')
                if isinstance(lv, dict): lv = lv.get('value','')
                if lv: g.add((s_uri, RDFS.label, Literal(lv, lang='fr')))
            for lkey in ('oLabel', 'obj_label'):
                lv = t.get(lkey, '')
                if isinstance(lv, dict): lv = lv.get('value','')
                if lv: g.add((o_uri, RDFS.label, Literal(lv, lang='fr')))
            added += 1
    return added


def step4_expand_kb(alignment_rows: list):
    """
    Expand the private KB from ~250 triples to 50,000–200,000 triples.

    Strategy:
      4a) 1-hop from each aligned entity
      4b) Predicate-controlled expansion (Morocco-anchored)
      4c) 2-hop via Morocco pivot (Q1028)
      4d) Bulk Moroccan administrative entities
      4e) Clean (remove isolated nodes, keep entity-entity triples)
      4f) Serialise to .nt and .ttl
    """
    print("\n" + "=" * 60)
    print("STEP 4 — KB Expansion via Wikidata SPARQL")
    print("=" * 60)

    # Load private KB + alignment files
    g = Graph()
    g.bind('', BASE)
    g.bind('wdt', WDT)
    g.bind('schema', SCHEMA)
    g.bind('owl', OWL)

    for ttl_file in [OUT_TTL, OUT_ALIGN_TTL, OUT_PRED_TTL]:
        if ttl_file.exists():
            g.parse(str(ttl_file), format='turtle')
            print(f"  Loaded {ttl_file}: {len(g)} triples")

    aligned_qids = [row['wikidata_qid'] for row in alignment_rows
                    if row.get('wikidata_qid','').startswith('Q')]
    print(f"\n  Aligned QIDs: {len(aligned_qids)}")

    # ── 4a: 1-hop expansion ───────────────────────────────────────────────────
    print("\n  4a) 1-hop expansion from each aligned entity...")
    for i, qid in enumerate(aligned_qids):
        query = f"""
SELECT ?p ?o ?oLabel WHERE {{
  wd:{qid} ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
  FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
}}
LIMIT 500
"""
        rows = _sparql_query(query)
        raw = [{'subject': f'http://www.wikidata.org/entity/{qid}',
                'predicate': r['p']['value'],
                'object':    r['o']['value'],
                'oLabel':    r.get('oLabel', {}).get('value','')}
               for r in rows]
        added = _add_triples(g, raw)
        print(f"    [{i+1}/{len(aligned_qids)}] wd:{qid} → +{added} triples (total: {len(g)})")
        time.sleep(0.5)

    # ── 4b: Predicate-controlled expansion (Morocco anchor) ───────────────────
    print("\n  4b) Predicate-controlled expansion (Morocco anchor Q1028)...")
    for pid, pname in [('P131','locatedIn'), ('P150','containsAdmin'),
                       ('P31','instanceOf'), ('P279','subclassOf'), ('P17','country')]:
        query = f"""
SELECT ?s ?o WHERE {{
  ?s wdt:{pid} ?o .
  ?s wdt:P17 wd:Q1028 .
  FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
}}
LIMIT 5000
"""
        rows = _sparql_query(query, timeout=60)
        raw = [{'subject': r['s']['value'], 'predicate': f'http://www.wikidata.org/prop/direct/{pid}',
                'object': r['o']['value']} for r in rows]
        added = _add_triples(g, raw)
        print(f"    wdt:{pid} ({pname}) → +{added} triples (total: {len(g)})")
        time.sleep(1.0)

    # ── 4c: 2-hop via Morocco pivot ───────────────────────────────────────────
    print("\n  4c) 2-hop expansion via Morocco (Q1028)...")
    two_hops = [
        ('P17', 'P31',  3000),
        ('P17', 'P131', 3000),
        ('P17', 'P749', 2000),
    ]
    for p1, p2, limit in two_hops:
        query = f"""
SELECT ?s ?o WHERE {{
  ?s wdt:{p1} wd:Q1028 .
  ?s wdt:{p2} ?o .
  FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
}}
LIMIT {limit}
"""
        rows = _sparql_query(query, timeout=60)
        raw = [{'subject': r['s']['value'],
                'predicate': f'http://www.wikidata.org/prop/direct/{p2}',
                'object': r['o']['value']} for r in rows]
        added = _add_triples(g, raw)
        print(f"    P17→P{p2} → +{added} triples (total: {len(g)})")
        time.sleep(1.0)

    # ── 4d: Bulk Moroccan admin entities — one query per predicate ───────────
    # (avoids the JSON parse error caused by large multi-predicate SPARQL responses)
    print("\n  4d) Bulk Moroccan administrative entity expansion (per predicate)...")
    bulk_pids = [
        ('P31',   10000),   # instance of
        ('P131',  10000),   # located in
        ('P17',   10000),   # country
        ('P150',  10000),   # contains administrative division
        ('P749',  5000),    # parent organization
        ('P361',  5000),    # part of
        ('P36',   5000),    # capital
        ('P1001', 5000),    # applies to jurisdiction
        ('P279',  5000),    # subclass of
    ]
    for pid, limit in bulk_pids:
        if len(g) >= 65000:
            print(f"    Reached 65k — stopping bulk expansion")
            break
        query = (
            "SELECT ?s ?o WHERE { "
            "?s wdt:P17 wd:Q1028 . "
            f"?s wdt:{pid} ?o . "
            "FILTER(STRSTARTS(STR(?o), \"http://www.wikidata.org/entity/Q\")) "
            f"}} LIMIT {limit}"
        )
        rows = _sparql_query(query, timeout=90)
        raw = [{'subject': r['s']['value'],
                'predicate': f'http://www.wikidata.org/prop/direct/{pid}',
                'object': r['o']['value']} for r in rows]
        added = _add_triples(g, raw)
        print(f"    wdt:{pid} (bulk) → +{added} triples (total: {len(g)})")
        time.sleep(1.0)

    # ── 4e: Clean graph ────────────────────────────────────────────────────────
    print("\n  4e) Cleaning graph...")
    degree = defaultdict(int)
    for s, p, o in g:
        if isinstance(o, URIRef):
            degree[s] += 1
            degree[o] += 1

    g_clean = Graph()
    g_clean.bind('', BASE)
    g_clean.bind('wdt', WDT)
    g_clean.bind('owl', OWL)
    g_clean.bind('rdfs', RDFS)

    for s, p, o in g:
        if isinstance(o, Literal):
            if p in (RDFS.label, RDF.type):
                g_clean.add((s, p, o))
        elif isinstance(o, URIRef) and not isinstance(o, BNode):
            if degree[s] > 0 and degree[o] > 0:
                g_clean.add((s, p, o))

    print(f"    After cleaning: {len(g_clean)} triples")

    # ── 4f: Serialise ──────────────────────────────────────────────────────────
    print("\n  4f) Serialising...")
    g_clean.serialize(destination=str(OUT_EXP_NT),  format='nt')
    g_clean.serialize(destination=str(OUT_EXP_TTL), format='turtle')
    print(f"  ✓ {OUT_EXP_NT}")
    print(f"  ✓ {OUT_EXP_TTL}")

    # ── Ontology file ──────────────────────────────────────────────────────────
    g_onto = Graph()
    g_onto.bind('', BASE)
    g_onto.bind('schema', SCHEMA)
    g_onto.bind('owl', OWL)
    g_onto.bind('wdt', WDT)

    onto = BASE['ontology']
    g_onto.add((onto, RDF.type,       OWL.Ontology))
    g_onto.add((onto, RDFS.label,     Literal("Moroccan Consular Services Ontology")))
    g_onto.add((onto, OWL.versionInfo, Literal("2.0")))

    classes = {
        BASE.AdministrativeDocument: (SCHEMA.CreativeWork,  "Official document from consular authorities"),
        BASE.AdministrativeRole:     (SCHEMA.Role,          "A named administrative role"),
        BASE.ConsularProcedure:      (OWL.Thing,            "A consular procedure"),
        BASE.LegalText:              (SCHEMA.Legislation,   "A law or legal reference"),
    }
    for cls, (parent, comment) in classes.items():
        g_onto.add((cls, RDF.type,        OWL.Class))
        g_onto.add((cls, RDFS.subClassOf, parent))
        g_onto.add((cls, RDFS.comment,    Literal(comment)))

    props_dr = [
        (BASE.isIssuedBy,   SCHEMA.GovernmentOrganization, BASE.AdministrativeDocument, WDT['P176']),
        (BASE.isLocatedIn,  SCHEMA.GovernmentOrganization, SCHEMA.Place,                WDT['P131']),
        (BASE.reportsTo,    SCHEMA.GovernmentOrganization, SCHEMA.GovernmentOrganization, WDT['P749']),
        (BASE.partOf,       OWL.Thing,                     OWL.Thing,                   WDT['P361']),
        (BASE.country,      OWL.Thing,                     SCHEMA.Place,                WDT['P17']),
        (BASE.instanceOf,   OWL.Thing,                     OWL.Class,                   WDT['P31']),
        (BASE.requires,     BASE.ConsularProcedure,        BASE.AdministrativeDocument, None),
        (BASE.relatedTo,    OWL.Thing,                     OWL.Thing,                   WDT['P1659']),
    ]
    for prop, domain, range_, equiv in props_dr:
        g_onto.add((prop, RDF.type,    OWL.ObjectProperty))
        g_onto.add((prop, RDFS.domain, domain))
        g_onto.add((prop, RDFS.range,  range_))
        if equiv:
            g_onto.add((prop, OWL.equivalentProperty, equiv))

    g_onto.serialize(destination=str(OUT_ONTO_TTL), format='turtle')
    print(f"  ✓ {OUT_ONTO_TTL}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    subjects   = set()
    objects_   = set()
    predicates = set()
    triple_count = 0
    for s, p, o in g_clean:
        if isinstance(o, URIRef):
            subjects.add(s)
            objects_.add(o)
            predicates.add(p)
            triple_count += 1

    stats = {
        'total_triples':       triple_count,
        'total_entities':      len(subjects | objects_),
        'total_relations':     len(predicates),
        'meets_50k_triples':   triple_count >= 50000,
        'meets_5k_entities':   len(subjects | objects_) >= 5000,
        'meets_50_relations':  len(predicates) >= 50,
        'source':              'consulat.ma + Wikidata SPARQL',
        'expansion_strategy':  ['1-hop from aligned entities',
                                'Predicate-controlled Morocco-anchored',
                                '2-hop via Morocco Q1028 pivot',
                                'Bulk Moroccan administrative entities'],
    }
    with open(OUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {OUT_STATS}")

    print("\n  ══ FINAL KB STATISTICS ══════════════════════════════════")
    print(f"  Total triples   : {stats['total_triples']:>10,}")
    print(f"  Total entities  : {stats['total_entities']:>10,}")
    print(f"  Total relations : {stats['total_relations']:>10,}")
    print(f"  ≥ 50k triples   : {'✅' if stats['meets_50k_triples']  else '❌'}")
    print(f"  ≥ 5k entities   : {'✅' if stats['meets_5k_entities']  else '❌'}")
    print(f"  ≥ 50 relations  : {'✅' if stats['meets_50_relations'] else '❌'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Check inputs
    for f in [INPUT_ENT_CSV, INPUT_REL_CSV]:
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found.\n"
                "Run lab1_pipeline.py first to generate the NER outputs."
            )

    step1_build_kb()
    alignment_rows = step2_entity_linking()
    step3_predicate_alignment(alignment_rows)
    step4_expand_kb(alignment_rows)

    print("\n" + "=" * 60)
    print("✅  Pipeline complete — output files:")
    print("=" * 60)
    files = [
        (OUT_TTL,        "Step 1 — Private KB (Turtle)"),
        (OUT_ALIGN_CSV,  "Step 2 — Entity alignment table"),
        (OUT_ALIGN_TTL,  "Step 2 — owl:sameAs triples"),
        (OUT_NEW_ENT,    "Step 2 — Locally defined entities"),
        (OUT_PRED_CAND,  "Step 3 — SPARQL candidates (manual review)"),
        (OUT_PRED_TTL,   "Step 3 — Validated predicate alignment"),
        (OUT_EXP_NT,     "Step 4 — Expanded KB (N-Triples) ← use for KGE"),
        (OUT_EXP_TTL,    "Step 4 — Expanded KB (Turtle)"),
        (OUT_ONTO_TTL,   "Step 4 — Ontology file"),
        (OUT_STATS,      "Step 4 — Statistics report"),
    ]
    for path, desc in files:
        status = '✅' if path.exists() else '⬜'
        print(f"  {status}  {str(path):<45} {desc}")


if __name__ == "__main__":
    main()
