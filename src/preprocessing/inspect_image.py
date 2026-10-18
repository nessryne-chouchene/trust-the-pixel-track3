"""Inspection d'images EO : bandes, dimensions, dtype, plages de valeurs."""
import sys

import rasterio


def inspect(path):
    with rasterio.open(path) as src:
        print(f"Fichier : {path}")
        print(f"  bandes={src.count}  taille={src.height}x{src.width}  "
              f"dtype={src.dtypes[0]}  crs={src.crs}  nodata={src.nodata}")
        for i in range(1, src.count + 1):
            a = src.read(i).astype("float64")
            print(f"  bande {i:2d} : min={a.min():.3f}  max={a.max():.3f}  moy={a.mean():.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
