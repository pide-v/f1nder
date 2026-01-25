
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "🚀 Starting all baseline pipeline..."
bash $REPO_ROOT/20_build_index.sh
bash $REPO_ROOT/30_run_all_baselines.sh
bash $REPO_ROOT/51_evaluate_all.sh
echo "✅ All baseline pipeline completed."