REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


echo "Building Index for BM25F with entities..."
python "${REPO_ROOT}/src/f1nder/index/index_with_entities.py" \
    --document_collection_file "$REPO_ROOT/data/entities/document_collection_entities.json" \
    --index_path "$REPO_ROOT/data/index_bm25f" \
&& echo "✅ Index built!" \
|| echo "❌ Index build failed!"