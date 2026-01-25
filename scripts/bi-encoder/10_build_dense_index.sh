
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Building Dense Index with Chunked Pooling..."

CORPUS_PATH="$REPO_ROOT/data/document_collection.json"
INDEX_OUT="$REPO_ROOT/artifacts/index/index_dense_chunked_pooling_v4_BAAI_bge-large-en-v1.5/dense_index.faiss"
META_OUT="$REPO_ROOT/artifacts/index/index_dense_chunked_pooling_v4_BAAI_bge-large-en-v1.5/meta.jsonl"

MODEL_NAME="${MODEL_NAME:-BAAI/bge-large-en-v1.5}"
DEVICE="${DEVICE:-mps}" # mps is the device backend Metal (Apple GPU) supportato da PyTorch
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-128}" # 512, 384, 256, 128
CHUNK_TOKENS="${CHUNK_TOKENS:-128}" # numero di token per chunk (tra 256 e 512)
OVERLAP_TOKENS="${OVERLAP_TOKENS:-64}" # numero di token di overlap tra chunk (tra 64 e 128)
MIN_LAST_CHUNK_TOKENS="${MIN_LAST_CHUNK_TOKENS:-128}" # (tra 80 e 120) numero minimo di token per chunk valido

python "$REPO_ROOT/src/f1nder/index/build_dense_index.py" \
  --corpus "$CORPUS_PATH" \
  --index-out "$INDEX_OUT" \
  --meta-out "$META_OUT" \
  --model "$MODEL_NAME" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --max-length "$MAX_LEN" \
  --chunk-tokens "$CHUNK_TOKENS" \
  --overlap-tokens "$OVERLAP_TOKENS" \
  --min-last-chunk-tokens "$MIN_LAST_CHUNK_TOKENS" \
  --prepend-date true \
  --show-progress true \
  --verbose true

echo "✅ Dense Index built at $INDEX_OUT"