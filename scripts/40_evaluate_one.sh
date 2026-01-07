REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



RUN=""  # <-- TYPE HERE
OUTPUT=""  # <-- TYPE HERE


echo "..."
QRELS_FILE=$REPO_ROOT/data/test_qurels.json
RUN_FILE="$REPO_ROOT/results/runs/$RUN"
OUT_FILE="$REPO_ROOT/results/metrics/$OUTPUT"


if [[ ! -f "$QRELS_FILE" ]]; then
  echo "❌ Qrels file not found: $QRELS_FILE" >&2
  exit 2
fi

if [[ ! -f "$RUN_FILE" ]]; then
  echo "❌ Run file not found: $RUN_DIR" >&2
  exit 2
fi


python -m f1nder.eval.evaluate \
    --qrels_file $ \
    --run_file $RUN_FILE \
    --output_file $OUTPUT_FILE

echo "✅ !"