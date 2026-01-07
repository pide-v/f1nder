REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"



echo "Installing requirements..."

pip install -U "pyterrier[java]"
pip install -r $REPO_ROOT/requirements.txt

echo "✅ Requirements installed!"