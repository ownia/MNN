SHERPA_MNN_ROOT="$HOME/MNN/apps/frameworks/sherpa-mnn"
SHERPA_MNN_PYTHON="${SHERPA_MNN_PYTHON:-$HOME/MNN/apps/frameworks/.venv/bin/python}"
MODEL_DIR="$HOME/vits-melo-tts-zh_en"
MNN_CACHE_FILE="${MNN_CACHE_FILE:-$MODEL_DIR/melo_tts.opencl.mnncache}"

PYTHONPATH="$SHERPA_MNN_ROOT/sherpa-mnn/python:$SHERPA_MNN_ROOT/build-python/lib" \
LD_LIBRARY_PATH="$HOME/MNN/build/sherpa-mnn-deps/lib" \
$SHERPA_MNN_PYTHON \
  sherpa-mnn/python-api-examples/offline-tts.py \
  --provider opencl --num-threads 1 \
  --mnn-cache-file="$MNN_CACHE_FILE" \
  --vits-model=$MODEL_DIR/melo_tts.mnn \
  --vits-lexicon=$MODEL_DIR/lexicon.txt \
  --vits-tokens=$MODEL_DIR/tokens.txt \
  --tts-rule-fsts=$MODEL_DIR/phone.fst,$MODEL_DIR/date.fst,$MODEL_DIR/number.fst \
  --vits-dict-dir=$MODEL_DIR/dict \
  --sid=0 --output-filename=./test-melo.wav \
  "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着 时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。"
