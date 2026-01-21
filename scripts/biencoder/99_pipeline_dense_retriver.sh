REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


echo "Starting Full Pipeline: Build Index -> Dense Retrieval -> Rerank..."

# Usage:
# 99_pipeline_all.sh CORPUS QUERIES OUTDIR
CORPUS_PATH="${1:?Provide CORPUS_PATH}"
QUERIES_PATH="${2:?Provide QUERIES_PATH}"
OUTDIR="${3:?Provide OUTDIR}"

mkdir -p "$OUTDIR"

INDEX_PATH="$OUTDIR/index.faiss"
META_PATH="$OUTDIR/meta.jsonl"
RUN_DENSE="$OUTDIR/run_dense.trec"
RUN_RERANK="$OUTDIR/run_dense_rerank.trec"

bash scripts/biencoder/10_build_dense_index.sh "$CORPUS_PATH" "$INDEX_PATH" "$META_PATH"
bash scripts/biencoder/20_run_dense_retrieve.sh "$INDEX_PATH" "$META_PATH" "$QUERIES_PATH" "$RUN_DENSE"
bash scripts/biencoder/40_rerank.sh "$RUN_DENSE" "$RUN_RERANK" "$QUERIES_PATH" "$META_PATH"
bash scripts/biencoder/evaluate_dense_retriever.sh "$RUN_RERANK" "$QUERIES_PATH" "$META_PATH"
bash scripts/biencoder/60_RAG.sh "$INDEX_PATH" "$META_PATH" "$QUERIES_PATH" "$OUTDIR/answers.jsonl"

echo "DONE."
echo "Index:   $INDEX_PATH"
echo "Meta:    $META_PATH"
echo "Run1:    $RUN_DENSE"
echo "Run2:    $RUN_RERANK"

echo "✅ Full pipeline completed!"