REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Run tf-idf retrieval
echo "Running TF-IDF retrieval..."
python "$REPO_ROOT/src/f1nder/retrieval/tfidf.py"
echo "✅ TF-IDF run completed!"