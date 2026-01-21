REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Running RAG QA..."
python "$REPO_ROOT/src/f1nder/rag/qa.py" \
    --test_queries_path "$REPO_ROOT/data/test_queries.json" \
    --test_queries_answers_path "$REPO_ROOT/data/test_query_answers.json" \
    --output_path "$REPO_ROOT/artifacts/runs/rag_answers.jsonl"