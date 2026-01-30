
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "🚀 Evaluating Dense ..."

QUERIES_PATH="$REPO_ROOT/data/test_queries.json"
QRELS_PATH="$REPO_ROOT/data/test_qrels.json"
RUNS_DIR="$REPO_ROOT/artifacts/runs/before_preprocess/dense/"
METRICS_DIR="$REPO_ROOT/artifacts/metrics/before_preprocess/dense_new/"


python $REPO_ROOT/src/f1nder/eval/evaluate_all_dense.py \
  --queries_path "$QUERIES_PATH" \
  --qrels_path "$QRELS_PATH" \
  --runs_dir "$RUNS_DIR" \
  --metrics_dir "$METRICS_DIR"

echo "✅ Dense evaluation completed."