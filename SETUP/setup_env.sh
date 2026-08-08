#!/bin/bash
set -e

echo "=== [0/4] Checking if the environment already works ==="
if python3 -W ignore -c "
import torch
assert torch.__version__.startswith('2.4.0'), 'wrong torch version'
from mamba_ssm.modules.mamba_simple import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 64, device='cuda', dtype=torch.float32)
m(x)
" 2>/dev/null; then
    echo "Already working -- skipping reinstall."
    exit 0
fi
echo "Environment needs (re)installing -- proceeding."

echo "=== [1/4] Removing any drifted torch/mamba stack ==="
pip uninstall -y torch torchvision torchaudio causal-conv1d mamba-ssm 2>/dev/null || true

echo "=== [2/4] Installing known-good torch (2.4.0+cu121) ==="
pip install torch==2.4.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/4] Installing causal-conv1d + mamba-ssm (prebuilt wheels, no dep resolution) ==="
pip install causal-conv1d==1.4.0 --no-build-isolation --no-deps --force-reinstall
pip install mamba-ssm==2.2.2 --no-build-isolation --no-deps --force-reinstall
pip install einops

echo "=== [4/4] Verifying (self-heals mamba_ssm/__init__.py first, same as the real pipeline does) ==="
python3 -W ignore -c "
import importlib.util
spec = importlib.util.find_spec('mamba_ssm')
if spec and spec.origin:
    lines = [l for l in open(spec.origin).read().splitlines() if 'MambaLMHeadModel' not in l]
    open(spec.origin, 'w').write('\n'.join(lines) + '\n')

import torch
from mamba_ssm.modules.mamba_simple import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 64, device='cuda', dtype=torch.float32)
y = m(x)
print(f'RESULT: PASS  torch={torch.__version__}  output_shape={tuple(y.shape)}  dtype={y.dtype}')
"

echo ""
echo "Environment ready."
