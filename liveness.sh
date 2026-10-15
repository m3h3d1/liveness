#!/bin/sh
# Manage liveness v2: setup | convert | check | test | trace DIR | eval DIR | serve | start | stop | status | clean.
set -eu
cd "$(dirname "$0")"
[ -f .env ] && . ./.env
PY=.venv/bin/python
ZOO=https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models
SILENT=https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/b6d5f04ad78778917853b25c778acef6d5626d15
LANDMARKER=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MINIFASNET_V2=https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx

fetch() { curl -fsSL -o "$1" "$2"; }

case "${1:-}" in
  setup)
    uv venv --python 3.13 .venv
    if [ -f requirements.lock ]; then
      uv pip install --python "$PY" -r requirements.lock
    else
      uv pip install --python "$PY" mediapipe onnxruntime fastapi uvicorn python-multipart pytest httpx2
      uv pip freeze --python "$PY" > requirements.lock
    fi
    mkdir -p models samples
    fetch models/face_landmarker.task "$LANDMARKER"
    fetch models/face_detection_yunet_2026may.onnx "$ZOO/face_detection_yunet/face_detection_yunet_2026may.onnx"
    fetch models/minifasnet_v2.onnx "$MINIFASNET_V2"
    for f in image_F1 image_F2 image_T1; do
      fetch "samples/$f.jpg" "$SILENT/images/sample/$f.jpg"
    done
    shasum -a 256 -c SHA256SUMS
    [ -f models/minifasnet_v1se.onnx ] || "$0" convert
    ;;
  convert)
    # One-time V1SE .pth -> ONNX in a throwaway torch venv.
    C=.convert
    rm -rf "$C" && mkdir "$C"
    fetch "$C/MiniFASNet.py" "$SILENT/src/model_lib/MiniFASNet.py"
    fetch "$C/v1se.pth" "$SILENT/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
    (cd "$C" && shasum -a 256 -c ../tools/convert.sha256)
    uv venv -q --python 3.13 "$C/venv"
    uv pip install -q --python "$C/venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cpu \
      torch onnx onnxruntime
    "$C/venv/bin/python" tools/convert_v1se.py "$C" models/minifasnet_v1se.onnx
    rm -rf "$C"
    ;;
  check)
    "$PY" -m app.check
    ;;
  test)
    "$PY" -m pytest -q
    ;;
  trace)
    "$PY" -m tools.trace "$2"
    ;;
  eval)
    "$PY" -m tools.eval "$2"
    ;;
  serve)
    # Log to the terminal and to ~/Library/Logs/liveness-v2.log (measurements only, no images).
    .venv/bin/uvicorn app.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8090}" \
      --limit-concurrency 20 --no-server-header 2>&1 | tee -a "$HOME/Library/Logs/liveness-v2.log"
    ;;
  start)
    # Background serve; the log still goes to ~/Library/Logs/liveness-v2.log.
    pgrep -f "uvicorn app.server:app" >/dev/null && { echo "already running"; exit 0; }
    nohup "$0" serve >/dev/null 2>&1 &
    sleep 5
    "$0" status
    ;;
  stop)
    pkill -f "uvicorn app.server:app" && echo "stopped" || echo "not running"
    ;;
  status)
    if pgrep -f "uvicorn app.server:app" >/dev/null; then
      echo "running (pid $(pgrep -f 'uvicorn app.server:app' | head -1))"
      curl -s -m 3 "http://${HOST:-127.0.0.1}:${PORT:-8090}/health"; echo
    else
      echo "not running"
    fi
    ;;
  clean)
    rm -rf .venv .convert models samples
    ;;
  *)
    echo "usage: $0 setup | convert | check | test | trace DIR | eval DIR | serve | start | stop | status | clean" >&2
    exit 1
    ;;
esac
