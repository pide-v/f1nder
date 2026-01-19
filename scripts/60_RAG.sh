REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


echo "RAG Script Running..."

INDEX_PATH="${1:?Provide INDEX_PATH}"
META_PATH="${2:?Provide META_PATH}"
QUERIES_PATH="${3:?Provide QUERIES_PATH}"
ANSWERS_OUT="${4:?Provide ANSWERS_OUT (.jsonl)}"

# knobs
EMBEDDER="${EMBEDDER:-BAAI/bge-base-en-v1.5}"
K="${K:-50}"
RERANK="${RERANK:-0}"                  # 1=on, 0=off
RERANKER="${RERANKER:-BAAI/bge-reranker-base}"
RERANK_TOPN="${RERANK_TOPN:-20}"
CTX_DOCS="${CTX_DOCS:-5}"

GENERATOR="${GENERATOR:-hf}"         # none | hf
HF_MODEL="${HF_MODEL:-google/flan-t5-base}"
HF_DEVICE="${HF_DEVICE:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

ARGS=(
  --index "$INDEX_PATH"
  --meta "$META_PATH"
  --queries "$QUERIES_PATH"
  --out "$ANSWERS_OUT"
  --embedder "$EMBEDDER"
  --k "$K"
  --context-docs "$CTX_DOCS"
  --generator "$GENERATOR"
  --hf-model "$HF_MODEL"
  --hf-device "$HF_DEVICE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ "$RERANK" == "1" ]]; then
  ARGS+=(--rerank --reranker "$RERANKER" --rerank-topn "$RERANK_TOPN")
fi

python scripts/run_rag.py "${ARGS[@]}"

echo "✅ RAG run completed!"