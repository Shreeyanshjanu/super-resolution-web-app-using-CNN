import numpy as np
import rasterio
from affine import Affine
from pydantic import ValidationError
from rasterio.warp import transform_bounds

from inference.aoi import AOI, BANDS, inspect_raster, patch_count, plan_crop, write_crop, finish_output
from inference.preview import create_preview
from tests.helpers import RasterTestCase, make_raster


class AOITests(RasterTestCase):
    def test_patch_budget_matches_edge_windows(self):
        self.assertEqual(patch_count(512,512),25)
        self.assertEqual(patch_count(128,128),1)
        self.assertEqual(patch_count(129,128),2)
        self.assertEqual(patch_count(248,248),4)
        self.assertEqual(patch_count(256,256),9)
        with self.assertRaisesRegex(ValueError, "limit"):
            plan_crop(self.source, AOI(mode="pixel", x=0,y=0,width=256,height=256))

    def test_fixed_crop_preserves_pixels_and_transform(self):
        plan = plan_crop(self.source, AOI())
        self.assertEqual(plan["window"], dict(x=192,y=192,width=128,height=128))
        crop, model = write_crop(self.source, self.root/"job", plan, [1,2,3,4],10000)
        with rasterio.open(self.source) as source, rasterio.open(crop) as result:
            np.testing.assert_array_equal(result.read(), source.read()[:,192:320,192:320])
            self.assertEqual(result.transform, source.transform * Affine.translation(192,192))
            self.assertEqual(result.crs, source.crs)
            self.assertEqual(result.descriptions, BANDS)
        with rasterio.open(model) as src:
            self.assertEqual(src.shape,(128,128))

    def test_small_edge_crop_padding_is_removed(self):
        source = make_raster(self.root/"small.tif",width=63,height=77,masked=True)
        plan = plan_crop(source, AOI())
        crop, model = write_crop(source,self.root/"job",plan,[1,2,3,4],10000)
        with rasterio.open(model) as src:
            self.assertEqual(src.shape,(128,128))
            data = np.repeat(np.repeat(src.read(),4,axis=1),4,axis=2)
            profile = src.profile.copy()
            profile.update(width=512,height=512,transform=src.transform*Affine.scale(0.25))
        raw = self.root/"raw.tif"
        with rasterio.open(raw,"w",**profile) as dst:
            dst.write(data)
        result = finish_output(raw,crop,self.root/"result.tif")
        with rasterio.open(source) as original, rasterio.open(result) as sr:
            self.assertEqual(sr.shape,(308,252))
            self.assertEqual(sr.crs,original.crs)
            np.testing.assert_allclose(sr.bounds,original.bounds)
            np.testing.assert_allclose(sr.res,(2.5,2.5))
            self.assertFalse(sr.dataset_mask()[20:40,20:40].any())
            self.assertTrue((sr.dataset_mask()[0:10,0:10] == 255).all())

    def test_band_descriptions_reorder_bgrn(self):
        source = make_raster(self.root/"bgrn.tif",descriptions=("B02","B03","B04","B08"))
        meta = inspect_raster(source)
        self.assertEqual(meta["band_indexes"],[3,2,1,4])
        crop,_ = write_crop(source,self.root/"job",plan_crop(source,AOI()),meta["band_indexes"],10000)
        with rasterio.open(crop) as src:
            self.assertGreater(src.read(1).mean(), src.read(3).mean())

    def test_unlabelled_bands_need_explicit_order(self):
        source = make_raster(self.root/"unlabelled.tif",descriptions=None)
        with self.assertRaisesRegex(ValueError,"Band descriptions"):
            inspect_raster(source)
        self.assertEqual(inspect_raster(source,"rgbn")["band_indexes"],[1,2,3,4])

    def test_float_reflectance_is_scaled_once(self):
        source = make_raster(self.root/"reflectance.tif",scale=1)
        meta = inspect_raster(source,value_scale=1)
        crop,_ = write_crop(source,self.root/"job",plan_crop(source,AOI()),meta["band_indexes"],1)
        with rasterio.open(crop) as src:
            self.assertGreater(src.read().min(),999)
            self.assertLess(src.read().max(),4101)

    def test_pixel_only_source_has_no_metre_claim(self):
        source = make_raster(self.root/"pixel.tif",crs=None)
        meta = inspect_raster(source)
        self.assertIsNone(meta["crs"])
        self.assertTrue(meta["warnings"])
        plan = plan_crop(source,AOI())
        self.assertIsNone(plan["output_resolution_m"])
        with self.assertRaisesRegex(ValueError,"CRS"):
            plan_crop(source,AOI(mode="bbox",bbox=(-106,36,-105.99,36.01)))

    def test_bbox_projects_and_snaps_to_source_pixels(self):
        with rasterio.open(self.source) as src:
            native = rasterio.windows.bounds(rasterio.windows.Window(100,120,40,50),src.transform)
            bbox = transform_bounds(src.crs,"EPSG:4326",*native,densify_pts=21)
        plan = plan_crop(self.source,AOI(mode="bbox",bbox=bbox))
        w = plan["window"]
        self.assertLessEqual(w["x"],100)
        self.assertLessEqual(w["y"],120)
        self.assertGreaterEqual(w["x"]+w["width"],140)
        self.assertGreaterEqual(w["y"]+w["height"],170)
        self.assertLessEqual(w["width"],42)
        self.assertEqual(plan["patches"],1)

    def test_invalid_and_outside_aois_fail(self):
        with self.assertRaises(ValidationError):
            AOI(mode="bbox",bbox=(0,0,float("nan"),1))
        with self.assertRaises(ValueError):
            plan_crop(self.source,AOI(mode="bbox",bbox=(0,0,0.01,0.01)))
        with self.assertRaisesRegex(ValueError,"outside"):
            plan_crop(self.source,AOI(mode="pixel",x=500,y=0,width=128,height=128))

    def test_invalid_resolution_is_rejected(self):
        source = make_raster(self.root/"degrees.tif",crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError,"projected CRS"):
            inspect_raster(source)

    def test_map_preview_is_bounded_and_georeferenced(self):
        from PIL import Image
        result = create_preview(self.source,self.root/"preview.png",[1,2,3,4],max_size=128)
        with Image.open(self.root/"preview.png") as image:
            self.assertLessEqual(max(image.size),128)
        self.assertEqual(len(result["preview_bounds_wgs84"]),4)

    def test_out_of_range_crop_is_rejected(self):
        with rasterio.open(self.source,"r+") as src:
            bad = src.read()
            bad[:,192:320,192:320] = 20000
            src.write(bad)
        with self.assertRaisesRegex(ValueError,"reflectance"):
            write_crop(self.source,self.root/"job",plan_crop(self.source,AOI()),[1,2,3,4],10000)
