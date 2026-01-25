REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Running all baselines..."
python3 "${REPO_ROOT}/src/f1nder/retrieval/baselines.py" \
    --corpus_path="${REPO_ROOT}/data/document_collection_preprocessed_super_light.json" \
    --queries_path="${REPO_ROOT}/data/test_queries_preprocessed_super_light.json" \
    --qrels_path="${REPO_ROOT}/data/test_qrels.json" \
    --index_dir="${REPO_ROOT}/artifacts/index/index_preprocessed_super_light" \
    --runs_dir="${REPO_ROOT}/artifacts/runs/after_preprocess_super_light" \
    --metrics_dir="${REPO_ROOT}/artifacts/metrics/after_preprocess_super_light" \
    --docno_field="para_id" \
    --text_field="preprocessed"
echo "✅ All baselines completed!"