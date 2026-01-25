REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Building Index..."
python "$REPO_ROOT/src/f1nder/index/index.py" \
    --document_collection_file $REPO_ROOT/data/document_collection_preprocessed_light.json \
    --index_path "$REPO_ROOT/data/index/index_preprocessed_light" \
    --text_field "preprocessed" \
    --docno_field "para_id" \
    --force_rebuild false
echo "✅ Index built!"