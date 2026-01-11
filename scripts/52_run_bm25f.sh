REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


echo "Running BM25F retrieval with entities..."
python "${REPO_ROOT}/src/f1nder/retrieval/bm25f.py" \
    --corpus_path="${REPO_ROOT}/data/entities/document_collection_merged.json" \
    --queries_path="${REPO_ROOT}/data/test_queries.json" \
    --qrels_path="${REPO_ROOT}/data/test_qurels.json" \
    --index_dir="${REPO_ROOT}/data/index_bm25f" \
    --runs_dir="${REPO_ROOT}/artifacts/runs" \
    --metrics_dir="${REPO_ROOT}/artifacts/metrics" \
&& echo "✅ BM25F run and evaluation completed!" \
|| echo "❌ BM25F failed!"