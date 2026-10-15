"""Quick check that the environment is set up correctly."""
import sys

import numpy, pandas, matplotlib, sklearn
import torch, transformers, accelerate

print(f"Python        {sys.version.split()[0]}")
for mod in (numpy, pandas, matplotlib, sklearn, torch, transformers, accelerate):
    print(f"{mod.__name__:<13} {mod.__version__}")

if torch.backends.mps.is_available():
    device = "mps (Apple GPU)"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Torch device  {device}")
