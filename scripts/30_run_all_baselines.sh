REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "Running all baselines..."
python3 "${REPO_ROOT}/src/f1nder/retrieval/baselines.py" \
    --corpus_path="${REPO_ROOT}/data/document_collection.json" \
    --queries_path="${REPO_ROOT}/data/test_queries.json" \
    --qrels_path="${REPO_ROOT}/data/test_qurels.json" \
    --index_dir="${REPO_ROOT}/artifacts/index" \
    --runs_dir="${REPO_ROOT}/artifacts/runs" \
    --metrics_dir="${REPO_ROOT}/artifacts/metrics"
echo "✅ All baselines completed!"