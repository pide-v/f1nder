REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "Building Index..."
INDEX_PATH="$REPO_ROOT/data/index/index"
python -m f1nder.index.build_index \
    --document_collection_file $REPO_ROOT/data/document_collection.json \
    --index_path $INDEX_PATH
echo "✅ Index builted!"