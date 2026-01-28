REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Evaluating RAG QA..."
python "$REPO_ROOT/src/f1nder/eval/evaluate_qa.py" \
    --top1_path "$REPO_ROOT/artifacts/runs/rag_eval.jsonl" \
    --top5_path "$REPO_ROOT/artifacts/runs/rag_eval_top5.jsonl" \
    --output_path "$REPO_ROOT/artifacts/runs/rag_eval_top5.jsonl"