REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "Running retrieval..."
QRELS_PATH="$REPO_ROOT/data/test_qurels.json"
TEST_QUERIES_PATH="$REPO_ROOT/data/test_queries.json"
INDEX_PATH="$REPO_ROOT/data/index/index"
OUTPUT_PATH="$REPO_ROOT/results/runs/"
python -m f1nder.retrieval.bm25 \
    --qrels_path $QRELS_PATH \
    --test_queries_path $TEST_QUERIES_PATH \
    --index_path $INDEX_PATH \
    --output_path $OUTPUT_PATH
echo "✅ !"