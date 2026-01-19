echo "Building Dense Index..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CORPUS_PATH="$REPO_ROOT/data/document_collection.json"
INDEX_OUT="$REPO_ROOT/artifacts/index_dense/dense_index.faiss"
META_OUT="$REPO_ROOT/artifacts/index_dense/meta.jsonl"

MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
DEVICE="${DEVICE:-mps}" # mps is the device backend Metal (Apple GPU) supportato da PyTorch
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LEN="${MAX_LEN:-512}"

python "$REPO_ROOT/src/f1nder/index/build_dense_index.py" \
  --corpus "$CORPUS_PATH" \
  --index-out "$INDEX_OUT" \
  --meta-out "$META_OUT" \
  --model "$MODEL_NAME" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --max-length "$MAX_LEN" \
  --prepend-date true \
  --show-progress true \
  --verbose true

echo "✅ Dense Index built at $INDEX_OUT"