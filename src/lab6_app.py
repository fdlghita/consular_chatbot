"""
Lab 6 — RAG Chatbot Interface (Streamlit)
==========================================
Run with:
    streamlit run src/lab6_app.py
"""

import time
from pathlib import Path

import streamlit as st
from rdflib import Graph

# ── Import the core RAG functions from lab6_pipeline ─────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from lab6_pipeline import (
    load_graph,
    build_schema_summary,
    answer_no_rag,
    answer_with_sparql_rag,
    OLLAMA_MODEL,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Consulaire 🇲🇦",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark, sleek background */
.stApp {
    background: #0f172a;
    color: #f8fafc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid #334155;
    color: #f8fafc;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
    color: #f8fafc !important;
}

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.85);
    margin: 8px 0 0;
    font-size: 1rem;
}

/* Chat messages */
.msg-user {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0 8px 15%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    font-weight: 500;
}
.msg-bot-baseline {
    background: #334155;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 16px;
    margin: 4px 15% 4px 0;
    border-left: 4px solid #94a3b8;
    color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
.msg-bot-rag {
    background: rgba(22, 163, 74, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 8px;
    padding: 16px;
    margin: 4px 15% 8px 0;
    border-left: 4px solid #22c55e;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    color: #bbf7d0;
}

/* Labels */
.label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.label-baseline { color: #94a3b8; }
.label-rag { color: #4ade80; }

/* Result table */
.result-pill {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #86efac;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    margin: 3px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

/* Stats bar */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.stat-badge {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #cbd5e1;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

/* Input area */
.stTextInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ef4444 !important;
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2) !important;
}

/* Button */
.stButton > button {
    background: #991b1b !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #dc2626 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
}

/* Expander */
details {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    color: #cbd5e1 !important;
}

/* Divider */
hr { border-color: #334155; }

/* Sidebar elements */
.info-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.info-card strong { color: #ef4444; }

/* Global text styling overrides for dark mode */
p, label, span {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "graph_loaded" not in st.session_state:
    st.session_state.graph_loaded = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    # Model selector
    model_choice = st.selectbox(
        "Modèle LLM (Ollama)",
        ["gemma:2b", "deepseek-r1:1.5b", "llama3.2", "qwen:0.5b"],
        index=0,
        help="Modèle local utilisé via Ollama"
    )

    # Max repair tries
    max_repair = st.slider("Tentatives self-repair", 1, 5, 2)

    st.markdown("---")

    # Graph loading
    _here = Path(__file__).parent
    default_ttl = str((_here / "data2" / "expanded_kb.ttl").resolve())
    ttl_path_input = st.text_input(
        "📂 Chemin vers le graphe (.ttl)",
        value=default_ttl
    )

    if st.button("Charger le Graphe RDF", use_container_width=True):
        ttl_path = Path(ttl_path_input).resolve()
        with st.spinner("Chargement du graphe RDF..."):
            try:
                st.session_state.graph = load_graph(ttl_path)
                st.session_state.schema = build_schema_summary(st.session_state.graph)
                st.session_state.graph_loaded = True
                st.success(f"✅ {len(st.session_state.graph)} triplets chargés avec succès.")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.graph_loaded:
        g = st.session_state.graph
        st.markdown(f"""
<div class="info-card">
  <strong>📊 Graphe chargé</strong><br>
  Triplets : {len(g)}<br>
  Modèle : {model_choice}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
<div style="font-size:0.75rem; color:#cbd5e1; text-align:center; margin-top:16px;">
  Lab 6 — RAG Consulaire Marocain<br>ESILV A4 · Web Data Mining
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
  <h1>Chatbot Consulaire Marocain</h1>
  <p>Posez vos questions sur les consulats, documents et procédures — propulsé par RAG + SPARQL</p>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user"> {msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        data = msg["content"]

        # Baseline
        st.markdown('<div class="label label-baseline">🧠 Baseline — LLM sans connaissance</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-bot-baseline">{data["baseline"]}</div>', unsafe_allow_html=True)

        # RAG
        st.markdown('<div class="label label-rag">📍 RAG — SPARQL + Graphe RDF</div>', unsafe_allow_html=True)

        rag = data["rag"]
        rows = rag.get("rows", [])
        err  = rag.get("error")
        rep  = rag.get("repaired", False)
        off_topic = rag.get("off_topic", False)

        # Status badges
        badges = []
        badges.append(f"⏱️ Baseline : {data['t_baseline']:.1f}s")
        badges.append(f"⏱️ RAG : {data['t_rag']:.1f}s")
        if off_topic:
            badges.append("🚫 Hors-sujet")
        elif rep:
            badges.append(f"🔧 Auto-réparé ({rag.get('repair_count',0)}x)")
        if err:
            badges.append("❌ Erreur SPARQL")
        elif not off_topic:
            badges.append(f"✅ {len(rows)} résultat(s)")

        badge_html = "".join(f'<span class="stat-badge">{b}</span>' for b in badges)
        st.markdown(f'<div class="stat-row">{badge_html}</div>', unsafe_allow_html=True)

        if off_topic:
            st.markdown('<div class="msg-bot-rag">🚫 Remarque : Cette question ne concerne pas la base de connaissances consulaire marocaine. Posez une question sur les consulats, documents, ou procédures administratives.</div>', unsafe_allow_html=True)
        elif err:
            st.markdown(f'<div class="msg-bot-rag">⚠️ Erreur d\'exécution : Le graphe n\'a pas pu répondre à cette question.<br><small style="opacity:0.6">{err[:200]}</small></div>', unsafe_allow_html=True)
        elif not rows:
            st.markdown('<div class="msg-bot-rag">📭 Aucun résultat trouvé dans le graphe RDF pour cette question.</div>', unsafe_allow_html=True)
        else:
            vars_ = rag.get("vars", [])
            # Format results as pills
            items_html = ""
            for row in rows[:15]:
                # Show only the last part of URIs for readability
                vals = [v.split("/")[-1] if v.startswith("http") else str(v) for v in row]
                items_html += '<span class="result-pill">' + " · ".join(vals) + "</span>"
            if len(rows) > 15:
                items_html += f'<span class="result-pill">+{len(rows)-15} autres</span>'

            final_ans = rag.get("final_answer", "")
            if final_ans:
                st.markdown(f'<div class="msg-bot-rag">💬 <strong>{final_ans}</strong><br><br><small><i>Sources brutes (Graph RDF) :</i></small><br>{items_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-bot-rag">{items_html}</div>', unsafe_allow_html=True)

        # Expandable SPARQL details
        with st.expander("🔎 Voir le SPARQL généré", expanded=False):
            sparql_query = rag.get("query", "")
            st.code(sparql_query, language="sparql")
            if rows:
                import pandas as pd
                vars_ = rag.get("vars", [])
                df = pd.DataFrame(rows[:50], columns=vars_ if vars_ else None)
                st.dataframe(df, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
if not st.session_state.graph_loaded:
    st.info("**Commencez par charger le graphe RDF** dans la barre latérale.")
else:
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Question",
                placeholder="Ex: Quelles organisations sont localisées en France ?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Envoyer", use_container_width=True)

    # Example questions
    st.markdown("<p style='font-size:0.8rem; color:#cbd5e1; margin-top:4px;'>💡 Exemples :</p>", unsafe_allow_html=True)
    examples = [
        "Quelles organisations sont en France ?",
        "Quels documents sont liés au passeport ?",
        "Quelles entités mentionnent le Maroc ?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                user_input = ex
                submitted = True

    # ── Process question ──────────────────────────────────────────────────────
    if submitted and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": q})

        g      = st.session_state.graph
        schema = st.session_state.schema

        with st.spinner("🧠 Baseline en cours..."):
            t0 = time.time()
            try:
                baseline_ans = answer_no_rag(q)
            except Exception as e:
                baseline_ans = f"Erreur : {e}"
            t_baseline = time.time() - t0

        with st.spinner("🔍 Génération SPARQL + exécution RAG..."):
            t0 = time.time()
            try:
                import lab6_pipeline as _p6
                _p6.OLLAMA_MODEL = model_choice
                _p6.MAX_REPAIR_TRIES = max_repair
                rag_result = answer_with_sparql_rag(g, schema, q, try_repair=True)
            except Exception as e:
                rag_result = {"query": "", "vars": [], "rows": [], "repaired": False, "repair_count": 0, "error": str(e), "off_topic": False}
            t_rag = time.time() - t0

        st.session_state.messages.append({
            "role": "assistant",
            "content": {
                "baseline":   baseline_ans,
                "rag":        rag_result,
                "t_baseline": t_baseline,
                "t_rag":      t_rag,
            }
        })
        st.rerun()
