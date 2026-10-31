# Lab 6 — RAG with RDF/SPARQL and a Local Small LLM

This script implements a **Retrieval-Augmented Generation (RAG)** pipeline that answers natural language questions about the Moroccan Consular Knowledge Graph using a **local Small Language Model** (via Ollama).

## How it works

```
User Question
    │
    ▼
[Step 1] LLM answers directly (Baseline, no KB)
[Step 2] Schema context extracted from knowledge_graph.ttl
[Step 3] LLM generates SPARQL from question + schema
[Step 4] rdflib executes SPARQL against the KG
[Step 5] Self-repair: if SPARQL fails, LLM fixes it
[Step 6] Results displayed and compared to Baseline
```

## Requirements

### 1. Install Python dependencies
```bash
pip install rdflib requests
```

### 2. Install Ollama
Download from [https://ollama.com](https://ollama.com) and install.

### 3. Start Ollama & pull a model
```bash
ollama serve
ollama pull gemma:2b
```
> You can use other small models: `llama3.2`, `deepseek-r1:1.5b`, `qwen:0.5b`, etc.  
> Edit the `OLLAMA_MODEL` constant in `src/lab6_pipeline.py` to match.

### 4. Verify Ollama is running
```bash
curl http://localhost:11434/api/generate -d '{"model":"gemma:2b","prompt":"hello","stream":false}'
```
You should see a JSON response with a `"response"` field.

## Usage

### Interactive CLI Demo
Run from the `src/` directory:
```bash
python src/lab6_pipeline.py
```
Type any question and the app will show you:
- The **Baseline** answer (LLM with no knowledge)
- The **RAG** answer (SPARQL generated, executed, self-repaired if needed)

Type `quit` to exit.

### Benchmark Evaluation (5 preset questions)
```bash
python src/lab6_pipeline.py --eval
```
Generates `data6/evaluation_results.md` with a comparison table.

## Output Files

| File | Description |
|------|-------------|
| `data6/schema_summary.txt` | Schema context sent to the LLM (prefixes, predicates, classes) |
| `data6/evaluation_results.md` | Evaluation table: Baseline vs RAG for 5 questions |

## Changing the Model

Open `src/lab6_pipeline.py` and change:
```python
OLLAMA_MODEL = "gemma:2b"   # ← replace with your model name
```
