
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Running Dense Retriever..."

RUN_OUT="$REPO_ROOT/artifacts/runs/before_preprocess/run_dense_chunked_pooled_v4_BAAI_bge-large-en-v1.5.txt"

# MODEL_NAME="${MODEL_NAME:-BAAI/bge-small-en-v1.5}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-large-en-v1.5}"
DEVICE="${DEVICE:-mps}" # mps is the device backend Metal (Apple GPU) supportato da PyTorch
K="${K:-1000}"
CHUNK_K="${CHUNK_K:-5000}"
MAX_LEN="${MAX_LEN:-512}"


python "$REPO_ROOT/src/f1nder/retrieval/run_dense_retrieval.py" \
  --index "$REPO_ROOT/artifacts/index/index_dense_chunked_pooling_v4_BAAI_bge-large-en-v1.5/dense_index.faiss" \
  --meta "$REPO_ROOT/artifacts/index/index_dense_chunked_pooling_v4_BAAI_bge-large-en-v1.5/meta.jsonl" \
  --queries "$REPO_ROOT/data/test_queries.json" \
  --run-out "$RUN_OUT" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --k "$K" \
  --chunk-k "$CHUNK_K" \
  --max-length "$MAX_LEN"

echo "✅ Dense Retrieval run completed. Results saved to $RUN_OUT"