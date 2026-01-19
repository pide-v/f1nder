REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Running Dense Retriever..."

INDEX_PATH="${1:?Provide INDEX_PATH (.faiss)}"
META_PATH="${2:?Provide META_PATH (.jsonl)}"
QUERIES_PATH="${3:?Provide QUERIES_PATH (.json/.jsonl)}"
RUN_OUT="${4:?Provide RUN_OUT (trec run file)}"

MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
DEVICE="${DEVICE:-cpu}"
K="${K:-100}"
MAX_LEN="${MAX_LEN:-512}"

python scripts/run_dense_retrieval.py \
  --index "$INDEX_PATH" \
  --meta "$META_PATH" \
  --queries "$QUERIES_PATH" \
  --run-out "$RUN_OUT" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --k "$K" \
  --max-length "$MAX_LEN"

echo "✅ Dense Retrieval run completed. Results saved to $RUN_OUT"