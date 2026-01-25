
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Pre-processing the Document Collection..."

INPUT_CORPUS_FILE="$REPO_ROOT/data/document_collection.json"
OUTPUT_CORPUS_FILE="$REPO_ROOT/data/document_collection_preprocessed_super_light.json"

INPUT_QUERY_FILE="$REPO_ROOT/data/test_queries.json"
OUTPUT_QUERY_FILE="$REPO_ROOT/data/test_queries_preprocessed_super_light.json"

python -m f1nder.preprocess.preprocess_super_light \
    --input_corpus_file $INPUT_CORPUS_FILE \
    --output_corpus_file $OUTPUT_CORPUS_FILE \
    --input_queries_file $INPUT_QUERY_FILE \
    --output_queries_file $OUTPUT_QUERY_FILE
    
echo "✅ Document Collection pre-processed!"
