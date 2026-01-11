REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Extracting entities..."
python "$REPO_ROOT/src/f1nder/entities/extract_entities.py" \
    --input_file "$REPO_ROOT/data/document_collection.json" \
    --output_file "$REPO_ROOT/data/entities/document_collection_entities.json" \
&& echo "✅ Entities extracted!" \
|| echo "❌ Extraction failed!"

echo "Merging documents and entities..."
python "$REPO_ROOT/src/f1nder/entities/merge_documents_entities.py" \
    --input_doc_file "$REPO_ROOT/data/document_collection.json" \
    --input_entities_file "$REPO_ROOT/data/entities/document_collection_entities.json" \
    --output_file "$REPO_ROOT/data/entities/document_collection_merged.json" \
&& echo "✅ Merge completed!" \
|| echo "❌ Merge failed!"
