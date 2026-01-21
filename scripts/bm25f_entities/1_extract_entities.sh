REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Extracting entities from documents and queries..."
python "$REPO_ROOT/src/f1nder/entities/extract_entities.py" \
    --documents_input "$REPO_ROOT/data/document_collection.json" \
    --documents_output "$REPO_ROOT/data/entities/document_collection_entities.json" \
    --queries_input "$REPO_ROOT/data/test_queries.json" \
    --queries_output "$REPO_ROOT/data/entities/test_queries_entities.json" \
&& echo "✅ Entities extracted!" \
|| echo "❌ Extraction failed!"