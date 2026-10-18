"""Genere des echantillons SYNTHETIQUES pour tests techniques (pas de vraies donnees)."""
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

out = Path("data/raw/synthetic")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(out / "rgb_uint8.png")


def write_tif(name, arr, dtype):
    c, h, w = arr.shape
    with rasterio.open(
        out / name, "w", driver="GTiff", height=h, width=w, count=c, dtype=dtype,
        crs="EPSG:32632", transform=from_origin(500000, 4500000, 10, 10),
    ) as dst:
        dst.write(arr.astype(dtype))


write_tif("s2_13b_uint16.tif", rng.integers(0, 10000, (13, 128, 128)), "uint16")
write_tif("single_float32.tif", rng.random((1, 200, 150)), "float32")
print("Echantillons crees dans", out)
