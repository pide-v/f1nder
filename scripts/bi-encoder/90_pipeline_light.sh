
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Starting Bi-Encoder Pipeline..."
bash $REPO_ROOT/bi-encoder/10_build_dense_index.sh
bash $REPO_ROOT/bi-encoder/20_run_dense_retrieve.sh
bash $REPO_ROOT/bi-encoder/50_evaluate_dense_retriver.sh
echo "✅ Bi-Encoder Pipeline completed."