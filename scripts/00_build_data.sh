REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "Downloading Chronicling America QA Dataset..."
LOCAL_DATASET_DIR="$REPO_ROOT/data/ChroniclingAmericaQA"
if [ -d "$LOCAL_DATASET_DIR" ]; then
    echo "✅ Data directory $LOCAL_DATASET_DIR already exists. Skipping download."
else
    echo "Creating data directory $LOCAL_DATASET_DIR..."
    mkdir -p $LOCAL_DATASET_DIR
    curl -L "https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA/resolve/main/test.json?download=true" -o $LOCAL_DATASET_DIR/test.json
    curl -L "https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA/resolve/main/train.json?download=true" -o $LOCAL_DATASET_DIR/train.json
    curl -L "https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA/resolve/main/dev.json?download=true" -o $LOCAL_DATASET_DIR/validation.json
    echo "✅ Data downloaded to $LOCAL_DATASET_DIR"
fi



echo 
echo "Building the Document Collection..."
DOCUMENT_COLLECTION_FILE="$REPO_ROOT/data/document_collection.json"
python -m f1nder.data.build_collection \
    --input_dir $LOCAL_DATASET_DIR \
    --output_file $DOCUMENT_COLLECTION_FILE
echo "✅ Collection builted!"



echo
echo "Building the Test Queries Data Structure..."
python -m f1nder.data.build_queries \
    --input_file $LOCAL_DATASET_DIR/test.json \
    --output_file $REPO_ROOT/data/test_queries.json
echo "✅ Test Queries Data Structure builted!"



echo
echo "Building Qrels for test set..."
python -m f1nder.data.build_qrels \
    --input_file $LOCAL_DATASET_DIR/test.json \
    --qrels_file $REPO_ROOT/data/test_qurels.json \
    --answers_file $REPO_ROOT/data/test_query_answers.json
echo "✅ Qrels builted!"