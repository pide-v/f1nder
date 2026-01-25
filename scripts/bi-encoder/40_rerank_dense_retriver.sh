REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


echo "Running Rerank Dense Retriever..."

RUN_IN="${1:?Provide RUN_IN (dense trec run)}"
RUN_OUT="${2:?Provide RUN_OUT (reranked trec run)}"
QUERIES_PATH="${3:?Provide QUERIES_PATH (.json/.jsonl)}"
META_PATH="${4:?Provide META_PATH (.jsonl)}"

RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
DEVICE="${DEVICE:-cpu}"
TOPN="${TOPN:-20}"          # rerank only top-N per query (cost control)
BATCH_SIZE="${BATCH_SIZE:-32}"

python scripts/run_rerank.py \
  --run-in "$RUN_IN" \
  --run-out "$RUN_OUT" \
  --queries "$QUERIES_PATH" \
  --meta "$META_PATH" \
  --model "$RERANKER_MODEL" \
  --device "$DEVICE" \
  --topn "$TOPN" \
  --batch-size "$BATCH_SIZE"

echo "✅ Reranked run saved to $RUN_OUT"