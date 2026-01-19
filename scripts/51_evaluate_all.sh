REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



# ---- Defaults (override by passing args) ----
# 1) qrels path
QRELS_FILE="${1:-$REPO_ROOT/data/test_qrels.json}"
# 2) runs directory
RUNS_DIR="${2:-$REPO_ROOT/results/runs}"
# 3) output metrics directory
OUT_DIR="${3:-$REPO_ROOT/results/metrics}"

mkdir -p "$OUT_DIR"

echo "Evaluating all runs..."
echo "  QRELS: $QRELS_FILE"
echo "  RUNS:  $RUNS_DIR"
echo "  OUT:   $OUT_DIR"
echo

# Basic checks
if [[ ! -f "$QRELS_FILE" ]]; then
  echo "❌ Qrels file not found: $QRELS_FILE" >&2
  exit 2
fi

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "❌ Runs directory not found: $RUNS_DIR" >&2
  exit 2
fi

shopt -s nullglob

run_count=0
for run_path in "$RUNS_DIR"/*; do
  [[ -f "$run_path" ]] || continue

  run_file="$(basename "$run_path")"

  # Build output filename from run filename, handling common extensions
  base="$run_file"
  base="${base%.gz}"
  base="${base%.trec}"
  base="${base%.run}"
  output_file="${base}.json"

  echo "➡️  Evaluating: $run_file  ->  $output_file"

  python -m f1nder.eval.evaluate \
    --qrels_file "$QRELS_FILE" \
    --run_file "$run_path" \
    --output_file "$OUT_DIR/$output_file"

  run_count=$((run_count + 1))
done

if [[ "$run_count" -eq 0 ]]; then
  echo "⚠️  No run files found in: $RUNS_DIR"
  exit 0
fi

echo
echo "✅ Done! Evaluated $run_count run(s). Metrics saved to: $OUT_DIR"