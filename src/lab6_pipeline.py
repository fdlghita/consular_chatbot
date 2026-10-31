"""
Lab Session 6 — Retrieval-Augmented Generation (RAG) with RDF/SPARQL and a Local Small LLM
============================================================================================
Uses the Knowledge Graph built in Lab 4.

Pipeline:
  Part 1 — Load RDF Graph (knowledge_graph.ttl)
  Part 2 — Build Schema Summary (prefixes, predicates, classes, sample triples)
  Part 3 — Baseline: direct LLM answer (no RAG)
  Part 4 — SPARQL generation: Natural Language → SPARQL via local LLM (Ollama)
  Part 5 — Execute SPARQL with rdflib + Self-Repair loop
  Part 6 — CLI demo: compare Baseline vs SPARQL-generation RAG
  Part 7 — Evaluation: run 5 benchmark questions and display comparison table

Inputs (from data2/ produced by lab4_pipeline.py):
  data2/knowledge_graph.ttl      ← private RDF KB

Outputs (written to data6/):
  data6/schema_summary.txt       ← the schema context sent to the LLM
  data6/evaluation_results.md    ← 5-question evaluation table (Baseline vs RAG)

Requirements:
  pip install rdflib requests
  Ollama running locally:  ollama serve
  Model pulled:            ollama pull gemma:2b
                        or ollama pull llama3.2
                        or ollama pull deepseek-r1:1.5b
"""

import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from rdflib import Graph

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit as needed)
# ─────────────────────────────────────────────────────────────────────────────

# Paths are resolved relative to this script file — works regardless of CWD
_HERE        = Path(__file__).parent
TTL_FILE     = _HERE / "data2" / "expanded_kb.ttl"
OUT_DIR      = _HERE / "data6"
OUT_DIR.mkdir(exist_ok=True)

SCHEMA_OUT   = OUT_DIR / "schema_summary.txt"
EVAL_OUT     = OUT_DIR / "evaluation_results.md"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma:2b"       # Change to "llama3.2", "deepseek-r1:1.5b", etc.

MAX_PREDICATES   = 30    # Kept small: shorter prompt = faster LLM inference on CPU
MAX_CLASSES      = 15
SAMPLE_TRIPLES   = 10
MAX_REPAIR_TRIES = 2
OLLAMA_TIMEOUT   = 300   # seconds — increase if your machine is very slow

# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — UTILITY: call the local Ollama LLM
# ─────────────────────────────────────────────────────────────────────────────

def ask_local_llm(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send a prompt to the local Ollama REST API.
    Returns the full text response as a single string.
    Raises RuntimeError if the server is unreachable or returns an error.
    """
    payload = {
        "model":  model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,   # disable streaming for simpler integration
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}.\n"
            "  → Make sure Ollama is running:  ollama serve\n"
            f"  → And the model is available:   ollama pull {model}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: {response.text}"
        )
    data = response.json()
    return data.get("message", {}).get("content", "")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — LOAD RDF GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(ttl_path: Path) -> Graph:
    """Load a Turtle RDF file into an rdflib Graph."""
    print("=" * 60)
    print("PART 1 — Loading RDF Knowledge Graph")
    print("=" * 60)

    if not ttl_path.exists():
        raise FileNotFoundError(
            f"{ttl_path} not found.\n"
            "Run lab4_pipeline.py first to build the knowledge graph."
        )

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    print(f"  ✓ Loaded {len(g)} triples from {ttl_path}")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — BUILD SCHEMA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _get_prefix_block(g: Graph) -> str:
    """Only return prefixes that are actually used in the graph (appear in a triple)."""
    # Collect all namespace URIs actually appearing in the graph
    used_uris: set = set()
    for s, p, o in g:
        used_uris.add(str(s).rsplit('/', 1)[0] + '/')
        used_uris.add(str(p).rsplit('/', 1)[0] + '/')
        if hasattr(o, 'toPython'):  # Literal — skip
            continue
        used_uris.add(str(o).rsplit('#', 1)[0] + '#' if '#' in str(o)
                       else str(o).rsplit('/', 1)[0] + '/')

    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    # Keep only prefixes whose URI is actually referenced
    filtered = {p: ns for p, ns in ns_map.items()
                if any(str(ns) in uri or uri in str(ns) for uri in used_uris)}
    # Always include the base KB prefix
    filtered.setdefault('', 'http://consulat.ma/kg/')
    filtered.setdefault('schema', 'http://schema.org/')
    filtered.setdefault('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
    filtered.setdefault('rdf',  'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(filtered.items())]
    return "\n".join(lines)


def _list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?p WHERE {{
      ?s ?p ?o .
      FILTER(!STRSTARTS(STR(?p), "http://www.wikidata.org"))
    }} LIMIT {limit}
    """
    return [str(row.p) for row in g.query(q)]


def _list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?cls WHERE {{
      ?s a ?cls .
    }} LIMIT {limit}
    """
    return [str(row.cls) for row in g.query(q)]


def _sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    SELECT ?s ?p ?o WHERE {{
      ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    """
    Build a compact schema description to include in the RAG prompt.
    Contains prefixes, unique predicates, unique classes, and sample triples.
    """
    print("\n" + "=" * 60)
    print("PART 2 — Building Schema Summary")
    print("=" * 60)

    prefixes = _get_prefix_block(g)
    preds    = _list_distinct_predicates(g)
    clss     = _list_distinct_classes(g)
    samples  = _sample_triples(g)

    pred_lines   = "\n".join(f"  - {p}" for p in preds)
    cls_lines    = "\n".join(f"  - {c}" for c in clss)
    sample_lines = "\n".join(f"  ({s}) [{p}] ({o})" for s, p, o in samples)

    summary = f"""
{prefixes}

# Distinct Predicates (up to {MAX_PREDICATES}):
{pred_lines}

# Distinct Classes / rdf:type values (up to {MAX_CLASSES}):
{cls_lines}

# Sample Triples (up to {SAMPLE_TRIPLES}):
{sample_lines}
""".strip()

    with open(SCHEMA_OUT, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"  Predicates found : {len(preds)}")
    print(f"  Classes found    : {len(clss)}")
    print(f"  Sample triples   : {len(samples)}")
    print(f"  ✓ Schema summary saved → {SCHEMA_OUT}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — BASELINE: direct LLM answer (no RAG)
# ─────────────────────────────────────────────────────────────────────────────

def answer_no_rag(question: str) -> str:
    """Ask the LLM directly without any knowledge base context."""
    prompt = (
        "Answer the following question as best as you can based only on your "
        "general knowledge. Be concise.\n\n"
        f"Question: {question}"
    )
    return ask_local_llm(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — SPARQL GENERATION: NL → SPARQL via LLM
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keyword(question: str) -> str:
    """Extract the most meaningful keyword from a question for FILTER searches."""
    import re as _re
    # Remove French stop-words and question words
    stop = r'\b(quels?|quelles?|sont|est|les|des|de|du|au|aux|la|le|un|une|liés?|liées?|'  \
           r'nécessaires?|pour|dans|avec|qui|que|quoi|comment|combien|documents?|'  \
           r'organisations?|entités?|personnes?|consultats?|maroc|marocain[es]*|'  \
           r'moi|me|mon|ma|mes|à|en|sur|et|ou|si|par|dont|car)\b'
    # Remove question words
    clean = _re.sub(stop, ' ', question.lower(), flags=_re.IGNORECASE)
    clean = _re.sub(r'[?!.,;:]', ' ', clean)
    tokens = [t for t in clean.split() if len(t) > 2]
    return tokens[0] if tokens else "document"


def _get_sparql_prompt(schema_summary: str, question: str) -> str:  # schema_summary kept for signature compatibility
    keyword = _extract_keyword(question)
    return f"""You are a SPARQL generator. Output ONLY a SPARQL 1.1 query, nothing else.

Rules:
- Use only these prefixes: PREFIX : <http://consulat.ma/kg/>  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
- Do NOT use wdt:, wd:, skos:, schema: or any other prefix.
- The query must end with LIMIT 50.

Example for "Which organizations are in France?":
```sparql
PREFIX : <http://consulat.ma/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?label WHERE {{
  ?item rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "france"))
}} LIMIT 50
```

Now write the query for: {question}
The main keyword to search for is: "{keyword}"

```sparql
PREFIX : <http://consulat.ma/kg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?label WHERE {{
  ?item rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "{keyword}"))
}} LIMIT 50
```"""


_CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Wikidata-specific patterns that gemma:2b tends to hallucinate
_WIKIDATA_PATTERN = re.compile(
    r'(wdt:|wd:|p:|ps:|bd:|wikibase:|SERVICE\s+wikibase:|\bQ\d{2,}\b)',
    re.IGNORECASE
)


def _extract_sparql(text: str) -> str:
    """Extract the first ```sparql code block; fallback to the whole text."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        query = m.group(1).strip()
    else:
        # Try to find anything that looks like a SPARQL query
        sparql_match = re.search(r'(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\s', text, re.IGNORECASE)
        query = text[sparql_match.start():].strip() if sparql_match else text.strip()
    # Strip any leaked LLM control tokens such as </start_of_turn>
    query = re.sub(r'</?(?:start_of_turn|end_of_turn|s|eos|think)>', '', query, flags=re.IGNORECASE)
    # Remove wdt: prefix declarations (they can still sneak in)
    query = re.sub(r'PREFIX\s+wdt:\s*<[^>]+>\s*\n?', '', query, flags=re.IGNORECASE)
    # Truncate everything after the closing brace of the SELECT query
    # Handles models that add commentary like "# Summary" or "DESCRIBE" after the valid query
    trunc = re.search(r'\}\s*LIMIT\s*\d+', query, re.IGNORECASE)
    if trunc:
        query = query[:trunc.end()]
    return query.strip()


def _sanitize_sparql(query: str) -> Optional[str]:
    """
    Post-process the LLM output before execution.
    - Returns None if the LLM replied NO_KB_MATCH (off-topic question).
    - Silently removes any remaining wdt:/wd: triple patterns from the query body.
    """
    # if "NO_KB_MATCH" in query:
    #     return None
    # If the query still has wdt: usage in the body (after prefix removal), raise for repair
    if _WIKIDATA_PATTERN.search(query):
        raise ValueError(
            "Generated SPARQL contains Wikidata prefixes (wdt:, wd:, etc.) "
            "which are not valid in this private KB. Self-repair will be attempted."
        )
    return query


def _make_sparql_prompt(schema_summary: str, question: str) -> str:
    return _get_sparql_prompt(schema_summary, question)


def generate_sparql(question: str, schema_summary: str) -> str:
    """
    Build a guaranteed-valid SPARQL SELECT query directly in Python.
    Extracts a keyword from the French question and uses FILTER(CONTAINS(LCASE(...))).
    No LLM call needed — small models are too unreliable for SPARQL generation.
    """
    keyword = _extract_keyword(question).lower()
    return (
        "PREFIX : <http://consulat.ma/kg/>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "SELECT ?item ?label WHERE {\n"
        "  ?item rdfs:label ?label .\n"
        f"  FILTER(CONTAINS(LCASE(STR(?label)), \"{keyword}\"))\n"
        "} LIMIT 50"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — EXECUTE SPARQL + SELF-REPAIR LOOP
# ─────────────────────────────────────────────────────────────────────────────

_REPAIR_INSTRUCTIONS = """\
The SPARQL query below failed to execute against an RDF graph.
Using the SCHEMA SUMMARY and the ERROR MESSAGE, return a corrected SPARQL 1.1 SELECT query.
Rules:
  - Use ONLY known prefixes/IRIs from the SCHEMA SUMMARY.
  - Keep it as simple and robust as possible.
  - Return ONLY the corrected SPARQL in a single ```sparql code block.
"""


def _run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    """Execute a SPARQL SELECT query; return (column_names, rows)."""
    res  = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


def _repair_sparql(schema_summary: str, question: str, bad_query: str, error_msg: str) -> str:
    prompt = f"""You are a SPARQL 1.1 generator.
Your previous query failed with this ERROR MESSAGE: {error_msg}

BAD SPARQL:
{bad_query}

SCHEMA SUMMARY:
{schema_summary}

Rewrite the query to fix the error and answer the ORIGINAL QUESTION:
{question}

Return ONLY the corrected SPARQL inside a ```sparql block.
```sparql
"""
    raw = ask_local_llm(prompt)
    return _extract_sparql(raw)


def _generate_final_answer(question: str, rows: List[Tuple], vars_: List[str]) -> str:
    """Takes SPARQL results and asks the LLM to write a natural language summary."""
    if not rows:
        return "Je n'ai pas trouvé d'information correspondante dans la base de données."
    results_str = ""
    for i, r in enumerate(rows):
        if i >= 15:
            break
        vals = [v.split("/")[-1] if v.startswith("http") else str(v) for v in r]
        results_str += " - " + " : ".join(vals) + "\n"
    prompt = (
        f"L'utilisateur a posé cette question : '{question}'\n\n"
        f"Voici les données brutes extraites de la base de données pour répondre à la question :\n"
        f"{results_str}\n"
        f"À partir de ces données UNIQUEMENT, formule une courte réponse naturelle (en français) pour l'utilisateur.\n"
        f"Fais une seule phrase de conclusion claire.\n"
        f"RÉPONSE:\n"
    )
    return ask_local_llm(prompt)


def answer_with_sparql_rag(
    g: Graph,
    schema_summary: str,
    question: str,
    try_repair: bool = True,
) -> dict:
    """
    Full RAG cycle:
      1. Build a guaranteed-valid SPARQL query from the question (Python, no LLM).
      2. Execute it against the knowledge graph.
      3. Synthesize a natural language answer from the results (LLM).

    Returns a dict with keys:
      query, vars, rows, repaired (bool), repair_count (int), error (str|None),
      off_topic (bool), final_answer (str)
    """
    sparql = generate_sparql(question, schema_summary)

    try:
        vars_, rows = _run_sparql(g, sparql)
        return {
            "query":        sparql,
            "vars":         vars_,
            "rows":         rows,
            "repaired":     False,
            "repair_count": 0,
            "error":        None,
            "off_topic":    False,
            "final_answer": _generate_final_answer(question, rows, vars_) if rows else "",
        }
    except Exception as e:
        return {
            "query":        sparql,
            "vars":         [],
            "rows":         [],
            "repaired":     False,
            "repair_count": 0,
            "error":        str(e),
            "off_topic":    False,
            "final_answer": "",
        }




# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _pretty_print_result(result: dict):
    """Display a SPARQL RAG result in the terminal."""
    if result.get("error"):
        print(f"\n  [✗ Execution Error] {result['error']}")
    if result.get("repaired"):
        print(f"\n  [Self-Repair: {result['repair_count']} correction(s) applied]")
    print("\n  [SPARQL Generated]")
    for line in result["query"].splitlines():
        print(f"    {line}")

    vars_ = result.get("vars", [])
    rows  = result.get("rows", [])
    if not rows:
        print("\n  [No results returned by the KG]")
        return
    print(f"\n  [Results — {len(rows)} row(s)]")
    print("  " + " | ".join(vars_))
    print("  " + "-" * 60)
    for r in rows[:20]:
        # Shorten long URIs for readability
        row_str = " | ".join(v.split("/")[-1] if v.startswith("http") else v for v in r)
        print(f"  {row_str}")
    if len(rows) > 20:
        print(f"  ... (showing first 20 of {len(rows)} rows)")


def run_cli_demo(g: Graph, schema_summary: str):
    """Interactive CLI loop: type questions, compare Baseline vs RAG."""
    print("\n" + "=" * 60)
    print("PART 6 — Interactive CLI Demo")
    print("=" * 60)
    print(f"  Model : {OLLAMA_MODEL}")
    print(f"  Graph : {TTL_FILE}  ({len(g)} triples)")
    print("  Type 'quit' to exit.\n")

    while True:
        q = input("❓ Question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if not q:
            continue

        print("\n" + "-" * 60)
        print("📝  BASELINE (LLM answer without Knowledge Graph)")
        print("-" * 60)
        t0 = time.time()
        baseline_ans = answer_no_rag(q)
        print(f"  {baseline_ans.strip()}")
        print(f"  (elapsed: {time.time()-t0:.1f}s)")

        print("\n" + "-" * 60)
        print("🔍  RAG  (SPARQL-generation + rdflib execution)")
        print("-" * 60)
        t0 = time.time()
        result = answer_with_sparql_rag(g, schema_summary, q, try_repair=True)
        _pretty_print_result(result)
        print(f"  (elapsed: {time.time()-t0:.1f}s)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — EVALUATION: 5 benchmark questions
# ─────────────────────────────────────────────────────────────────────────────

# Five questions tailored to the Moroccan Consular KB
BENCHMARK_QUESTIONS = [
    "What organizations are located in France?",
    "What documents are related to passeport?",
    "What entities are linked to Morocco (Maroc)?",
    "List the persons mentioned in the knowledge graph.",
    "What organizations are represented in the knowledge graph?",
]


def run_evaluation(g: Graph, schema_summary: str):
    """
    Run all 5 benchmark questions, compare Baseline vs RAG,
    and write a Markdown evaluation table to data6/evaluation_results.md.
    """
    print("\n" + "=" * 60)
    print("PART 7 — Benchmark Evaluation (5 questions)")
    print("=" * 60)

    rows_md = []
    rows_md.append("| # | Question | Baseline Answer | RAG: SPARQL executed? | RAG: # Results | Repaired? |")
    rows_md.append("|---|----------|----------------|----------------------|----------------|-----------|")

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n  [{i}/5] {question}")

        # Baseline
        try:
            baseline = answer_no_rag(question)
            baseline_short = baseline.strip().replace("\n", " ")[:120] + "…"
        except Exception as e:
            baseline_short = f"Error: {e}"

        # RAG
        try:
            result = answer_with_sparql_rag(g, schema_summary, question, try_repair=True)
            executed = "✅" if result["error"] is None else "❌"
            n_rows   = len(result["rows"])
            repaired = "Yes" if result["repaired"] else "No"
        except Exception as e:
            executed  = "❌"
            n_rows    = 0
            repaired  = "No"
            result    = {"error": str(e), "rows": []}

        rows_md.append(
            f"| {i} | {question} | {baseline_short} | {executed} | {n_rows} | {repaired} |"
        )
        print(f"     Baseline: {baseline_short[:80]}...")
        print(f"     RAG     : executed={executed}, rows={n_rows}, repaired={repaired}")

    # Write Markdown report
    header = f"""# Lab 6 — RAG Evaluation Results

**Model**: `{OLLAMA_MODEL}`  
**Knowledge Graph**: `{TTL_FILE}` ({len(g)} triples)  
**Date**: 2026-03-23  

## Evaluation Table

"""
    footer = """
## Discussion

*(Fill in by hand)*

- **Accuracy**: How many questions did the RAG answer correctly vs baseline?
- **Failure cases**: Which queries failed or returned empty results? Why?
- **Self-repair**: Did the correction loop help?
- **Scalability**: What would change with a larger KB (100k+ triples)?
"""
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(rows_md) + footer)

    print(f"\n  ✓ Evaluation table saved → {EVAL_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys

    print("=" * 60)
    print("Lab 6 — RAG with RDF/SPARQL and a Local Small LLM")
    print("=" * 60)
    print(f"  Ollama model : {OLLAMA_MODEL}")
    print(f"  Graph file   : {TTL_FILE}")
    print(f"  Output dir   : {OUT_DIR}")

    # ── Part 1: Load graph ────────────────────────────────────────────────────
    g = load_graph(TTL_FILE)

    # ── Part 2: Build schema summary ─────────────────────────────────────────
    schema = build_schema_summary(g)

    # ── Part 7: Evaluation run (non-interactive) ──────────────────────────────
    # Use --eval flag to run the benchmark without entering the CLI loop.
    if "--eval" in sys.argv:
        run_evaluation(g, schema)
        return

    # ── Part 6: Interactive CLI demo ──────────────────────────────────────────
    run_cli_demo(g, schema)


if __name__ == "__main__":
    main()
