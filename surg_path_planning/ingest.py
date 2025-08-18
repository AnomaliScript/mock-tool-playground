# ingest.py — read a DICOM series folder -> save a preview + numpy volume
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

DICOM_DIR = Path("dicom_series")  # folder of .dcm files from one study

def window01(arr, center=50, width=350):
    lo, hi = center - width/2, center + width/2
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-6)).astype(np.float32)  # -> [0,1]

# 1) find & read the series
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(str(DICOM_DIR))
if not series_ids:
    raise SystemExit("No DICOM series found in ./dicom_series")
files = reader.GetGDCMSeriesFileNames(str(DICOM_DIR), series_ids[0])
reader.SetFileNames(files)
img = reader.Execute()  # SimpleITK image (physical spacing/origin preserved)

# 2) to numpy (z,y,x)
vol = sitk.GetArrayFromImage(img).astype(np.float32)

# 3) window + normalize to [0,1] (CT soft tissue window)
vol01 = window01(vol, center=50, width=350)

# 4) quick sanity checks (acceptance)
assert vol01.ndim == 3 and vol01.shape[0] >= 1
assert 0.0 <= float(vol01.min()) + 1e-6
assert float(vol01.max()) <= 1.0 + 1e-6

# 5) save outputs
np.save("volume.npy", vol01)
mid = vol01.shape[0] // 2
plt.imshow(vol01[mid], cmap="gray"); plt.axis("off")
plt.savefig("preview.png", dpi=150, bbox_inches="tight")

# log a few useful things
sx, sy, sz = img.GetSpacing()  # SimpleITK spacing order is (x, y, z)
print("volume shape (z,y,x):", vol01.shape)
print("voxel spacing (z,y,x):", (sz, sy, sx))
print("saved: volume.npy and preview.png")