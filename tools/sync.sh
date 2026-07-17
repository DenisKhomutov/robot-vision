#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU 0"; then
    EXTRA=cu126
    echo "[sync] GPU найдена: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    EXTRA=cpu
    echo "[sync] GPU нет — ставлю CPU-сборку torch"
fi

echo "[sync] uv sync --extra $EXTRA $*"
uv sync --extra "$EXTRA" "$@"

.venv/bin/python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"[sync] torch {torch.__version__} | cuda {ok}" + (f" | {torch.cuda.get_device_name(0)}" if ok else ""))
if ("+cu" in torch.__version__) != ok:
    print("[sync] ВНИМАНИЕ: сборка torch не совпадает с наличием GPU")
EOF
