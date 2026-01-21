REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Evaluating RAG answers..."
python "$REPO_ROOT/src/f1nder/eval/evaluate_rag.py" \
    --test_queries_answers_path "$REPO_ROOT/data/test_query_answers.json" \
    --model_answers_path "$REPO_ROOT/artifacts/runs/rag_answers.jsonl" \
    --output_path "$REPO_ROOT/artifacts/runs/rag_eval.jsonl"