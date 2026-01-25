REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"




echo "🚀 Evaluating run: $RUN..."

RUN="run_dense_chunked_pooled_v4_BAAI_bge-large-en-v1.5.txt"
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