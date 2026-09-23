from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

TEST_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "tests"


def make_raster(path, width=512, height=512, crs="EPSG:32613", descriptions=("B04","B03","B02","B08"), scale=10000, masked=False):
    yy, xx = np.mgrid[:height, :width]
    data = np.stack([1000 + index * 1000 + (xx + yy) % 100 for index in range(4)]).astype("float32")
    data *= scale / 10000
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4, dtype="float32",
                       crs=crs, transform=from_origin(390000, 4108000, 10, 10)) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = descriptions
        if masked:
            mask = np.full((height, width), 255, dtype="uint8")
            mask[5:10, 5:10] = 0
            dst.write_mask(mask)
    return path


class RasterTestCase(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root = Path(self.temp.name).resolve()
        self.source = make_raster(self.root / "source.tif")

    def tearDown(self):
        assert self.root.parent == TEST_ROOT.resolve()
        self.temp.cleanup()
