
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



# echo "🚀 Building Dense Index with Chunked Pooling..."

# CORPUS_PATH="$REPO_ROOT/data/document_collection.json"
# INDEX_OUT="$REPO_ROOT/artifacts/index/index_dense_BAAI_bge-large-en-v1.5/dense_index.faiss"
# META_OUT="$REPO_ROOT/artifacts/index/index_dense_BAAI_bge-large-en-v1.5/meta.jsonl"

# MODEL_NAME="${MODEL_NAME:-BAAI/bge-large-en-v1.5}"
# DEVICE="${DEVICE:-mps}" # mps is the device backend Metal (Apple GPU) supportato da PyTorch
# BATCH_SIZE="${BATCH_SIZE:-32}"
# MAX_LEN="${MAX_LEN:-512}" # 512, 384, 256, 128

# python "$REPO_ROOT/src/f1nder/index/build_dense_index.py" \
#   --corpus "$CORPUS_PATH" \
#   --index-out "$INDEX_OUT" \
#   --meta-out "$META_OUT" \
#   --model "$MODEL_NAME" \
#   --batch-size "$BATCH_SIZE" \
#   --device "$DEVICE" \
#   --max-length "$MAX_LEN" \
#   --prepend-date true \
#   --show-progress true \
#   --verbose true

# echo "✅ Dense Index built at $INDEX_OUT"



echo "🚀 Running Dense Retriever..."

RUN_OUT="$REPO_ROOT/artifacts/runs/before_preprocess/run_dense_BAAI_bge-large-en-v1.5.txt"

# MODEL_NAME="${MODEL_NAME:-BAAI/bge-small-en-v1.5}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-large-en-v1.5}"
DEVICE="${DEVICE:-mps}" # mps is the device backend Metal (Apple GPU) supportato da PyTorch
K="${K:-1000}"
MAX_LEN="${MAX_LEN:-512}"


python "$REPO_ROOT/src/f1nder/retrieval/run_dense_retrieval.py" \
  --index "$REPO_ROOT/artifacts/index/index_dense_BAAI_bge-large-en-v1.5/dense_index.faiss" \
  --meta "$REPO_ROOT/artifacts/index/index_dense_BAAI_bge-large-en-v1.5/meta.jsonl" \
  --queries "$REPO_ROOT/data/test_queries.json" \
  --run-out "$RUN_OUT" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --k "$K" \
  --max-length "$MAX_LEN"

echo "✅ Dense Retrieval run completed. Results saved to $RUN_OUT"


echo "🚀 Evaluating run: $RUN..."

RUN="run_dense_BAAI_bge-large-en-v1.5.txt"
OUTPUT=$RUN

QRELS_FILE=$REPO_ROOT/data/test_qrels.json
RUN_FILE="$REPO_ROOT/artifacts/runs/before_preprocess/$RUN"
OUT_FILE="$REPO_ROOT/artifacts/metrics/before_preprocess/$OUTPUT"


if [[ ! -f "$QRELS_FILE" ]]; then
  echo "❌ Qrels file not found: $QRELS_FILE" >&2
  exit 2
fi

if [[ ! -f "$RUN_FILE" ]]; then
  echo "❌ Run file not found: $RUN_FILE" >&2
  exit 2
fi


python -m f1nder.eval.evaluate \
    --qrels_file "$QRELS_FILE" \
    --run_file "$RUN_FILE" \
    --output_file "$OUT_FILE"

echo "✅ Evaluation completed. Results saved to $OUT_FILE"