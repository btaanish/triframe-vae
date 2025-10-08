# tools/upsample_sdf64_to_128.py
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

# --- paths ---
SRC = Path("data/Softnet/SDF_v1/resolution_64(before variations)")
DST = Path("data/Softnet/SDF_v1/resolution_128")
KEY = "pc_sdf_sample"
SRC_RES = 64
DST_RES = 128

def to_cube64(arr: np.ndarray) -> np.ndarray:
    """
    Normalize various stored shapes to (64, 64, 64).
    Accepts common cases:
      - (64,64,64)
      - (1,64,64,64) or (1,1,64,64,64)
      - (262144,) or (262144,1) or (1,262144)
    """
    a = np.asarray(arr)
    # Already cubic?
    if a.ndim == 3 and a.shape == (SRC_RES, SRC_RES, SRC_RES):
        return a.astype(np.float32, copy=False)

    # Remove any leading singleton dims
    while a.ndim > 3 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)

    # Now handle channel singleton on front
    if a.ndim == 4 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)

    # Handle final channel singleton
    if a.ndim == 4 and a.shape[-1] == 1:
        a = np.squeeze(a, axis=-1)

    # Flattened cases
    if a.ndim == 2:
        # e.g., (262144,1) or (1,262144)
        if 1 in a.shape and a.size == SRC_RES**3:
            a = a.reshape(-1)  # -> (262144,)
    if a.ndim == 1 and a.size == SRC_RES**3:
        a = a.reshape(SRC_RES, SRC_RES, SRC_RES)

    if a.ndim == 3 and a.shape == (SRC_RES, SRC_RES, SRC_RES):
        return a.astype(np.float32, copy=False)

    raise ValueError(f"Unrecognized SDF shape: {arr.shape}. "
                     f"Expected something equivalent to 64^3.")

def upsample_file(src_h5: Path, dst_h5: Path):
    with h5py.File(src_h5, "r") as f:
        if KEY not in f:
            raise KeyError(f"{KEY} not found in {src_h5}")
        sdf = f[KEY][:]

    sdf64 = to_cube64(sdf)                       # (64,64,64)
    t = torch.from_numpy(sdf64).float()[None,None]  # (1,1,64,64,64)
    t128 = F.interpolate(t, size=(DST_RES, DST_RES, DST_RES),
                         mode="trilinear", align_corners=False)
    out128 = t128[0,0].cpu().numpy().astype(np.float32)

    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_h5, "w") as f:
        f.create_dataset(KEY, data=out128, compression="gzip")
    print(f"[OK] {src_h5}  ->  {dst_h5}")

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source folder not found: {SRC}")
    for root, _, files in os.walk(SRC):
        for fn in files:
            if not fn.endswith(".h5"): 
                continue
            src = Path(root) / fn
            rel = src.relative_to(SRC)
            dst = DST / rel
            upsample_file(src, dst)

if __name__ == "__main__":
    main()

