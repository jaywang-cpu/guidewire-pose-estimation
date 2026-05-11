import numpy as np

data = np.load("data/raw/GuidewireDataset.npz")
for k in data.keys():
    v = data[k]
    print(f"{k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.1f}, max={v.max():.1f}")
