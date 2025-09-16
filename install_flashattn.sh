#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}   # set PYTHON=/path/to/python if you need a specific env

# 1) quick sanity checks and read versions from the active python environment
PY_OUT="$($PYTHON - <<'PY' || true
import sys
try:
    import torch, platform
except Exception as e:
    print("ERROR:TORCH_MISSING")
    sys.exit(2)
py = sys.version_info
py_tag = f"cp{py.major}{py.minor}"            # cp310, cp311, ...
tor = torch.__version__.split('+')[0]         # e.g. "2.5.1"
tor_mm = '.'.join(tor.split('.')[:2])         # e.g. "2.5"
cuda = getattr(torch.version, 'cuda', None)   # e.g. "12.1" or "11.8" or None
if cuda is None:
    print("ERROR:CUDA_UNKNOWN")
    sys.exit(3)
osname = platform.system().lower()            # linux, windows, darwin
print(py_tag, "torch"+tor_mm, cuda, osname)
PY
)"

if printf '%s\n' "$PY_OUT" | grep -q 'ERROR:TORCH_MISSING'; then
     echo "ERROR: torch is not importable in $PYTHON. Install PyTorch in this environment first."
  echo "Check with: $PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)'"
  exit 1
fi
if printf '%s\n' "$PY_OUT" | grep -q 'ERROR:CUDA_UNKNOWN'; then
  echo "ERROR: torch.version.cuda is None — looks like this is a CPU-only torch build."
  echo "FlashAttention requires a CUDA-capable torch build."
  exit 1
fi

read -r PY_TAG TORCH_TAG CUDA_VER OS_TYPE <<<"$PY_OUT"
echo "Detected: Python tag=$PY_TAG, torch tag=$TORCH_TAG, cuda=$CUDA_VER, os=$OS_TYPE"

# 2) map OS -> wheel platform substring used in release filenames
if [[ "$OS_TYPE" == "linux" ]]; then PLATFORM="linux_x86_64"
elif [[ "$OS_TYPE" == "windows" || "$OS_TYPE" == "mingw32" ]]; then PLATFORM="win_amd64"
else
  echo "Warning: OS '$OS_TYPE' may not have prebuilt wheels. Attempting linux_x86_64 search."
  PLATFORM="linux_x86_64"
fi

# 3) construct CUDA candidate tokens (e.g. cu121 for 12.1 and cu12)
CUDA_MAJOR="$(printf '%s' "$CUDA_VER" | cut -d. -f1)"
CUDA_MINOR="$(printf '%s' "$CUDA_VER" | cut -d. -f2 || true)"
CANDIDATES=()
if [[ -n "$CUDA_MAJOR" && -n "$CUDA_MINOR" ]]; then
  CANDIDATES+=("cu${CUDA_MAJOR}${CUDA_MINOR}")  # cu121, cu118, ...
fi
CANDIDATES+=("cu${CUDA_MAJOR}")                 # cu12, cu11, ...
# join for python code
CAND_JOIN="$(IFS=, ; echo "${CANDIDATES[*]}")"

# 4) query GitHub latest release and choose best asset
RELEASE_JSON="$(mktemp)"
curl -sL "https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest" -o "$RELEASE_JSON"

ASSET_URL="$(
  python - <<PY
import json,os,sys
js=json.load(open("$RELEASE_JSON"))
assets=js.get("assets",[])
py_tag=os.environ["PY_TAG"]
torch_tag=os.environ["TORCH_TAG"]
plat=os.environ["PLATFORM"]
cuda_candidates=os.environ["CANDS"].split(',')
def score(name):
    s=0
    n=name.lower()
    if py_tag.lower() in n: s+=4
    if torch_tag.lower() in n: s+=3
    if plat.lower() in n: s+=2
    for c in cuda_candidates:
        if c.lower() in n:
            s+=3
            break
    return s
best=None
best_score=0
for a in assets:
    n=a.get("name","")
    sc=score(n)
    if sc>best_score:
        best_score=sc
        best=a
if best and best_score>=7:
    print(best["browser_download_url"])
    sys.exit(0)
 else:
    # show top candidates for debugging if nothing found
    ranked=sorted(assets, key=lambda a: score(a.get("name","")), reverse=True)[:5]
    for a in ranked:
        print("#CAND", score(a.get("name","")), a.get("name",""))
    sys.exit(1)
PY
)" || true

if [[ -z "$ASSET_URL" ]]; then
     echo "No suitable prebuilt wheel found for:"
       echo "  python=$PY_TAG  torch=$TORCH_TAG  cuda candidates=${CANDIDATES[*]}  platform=$PLATFORM"
         echo "You can either:"
           echo "  * Try 'pip install flash-attn' (will compile from source if wheel missing)."
             echo "  * Or build from source following the repo README."
               echo "See the project's releases for available wheels: https://github.com/Dao-AILab/flash-attention/releases"
                 exit 2
fi

echo "Found wheel: $ASSET_URL"
echo "Installing with pip in the active environment..."
# prefer pip of the selected python
$PYTHON -m pip install --upgrade pip setuptools wheel
$PYTHON -m pip install --no-deps "$ASSET_URL"

echo "Done. Verify with:"
$PYTHON - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
from transformers.utils import is_flash_attn_available, is_flash_attn_2_available
print("is_flash_attn_available (FA1):", getattr(__import__('transformers.utils', fromlist=['is_flash_attn_available']), 'is_flash_attn_available', lambda: 'n/a')() )
print("is_flash_attn_2_available (FA2):", getattr(__import__('transformers.utils', fromlist=['is_flash_attn_2_available']), 'is_flash_attn_2_available', lambda: 'n/a')() )
PY

