import re
import sys

import numpy as np

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def read_values(path: str) -> np.ndarray:
    with open(path) as f:
        values = [float(match) for match in NUMBER_RE.findall(f.read())]
    return np.array(values, dtype=np.float64)


vals1 = read_values(sys.argv[1])
vals2 = read_values(sys.argv[2])

if vals1.shape != vals2.shape:
    print(f"Outputs differ in length: {vals1.shape[0]} vs {vals2.shape[0]}")
    sys.exit(1)

if np.allclose(vals1, vals2, atol=1e-4):
    print("Outputs match")
    sys.exit(0)

print("Outputs differ")
print("C output:     ", vals1)
print("Python output:", vals2)
sys.exit(1)
