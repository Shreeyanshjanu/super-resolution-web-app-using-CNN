from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio

from backend.services.srm_service import SRMService
from frontend.components.results import detail_images
from inference.aoi import AOI, plan_crop, write_crop
from tests.helpers import RasterTestCase
from tests.test_api import StubService


class QualityTests(RasterTestCase):
    def test_each_job_uses_its_saved_steps_and_restores_model_config(self):
        config=SimpleNamespace(denoiser_settings=SimpleNamespace(sampling_steps=100))
        model=SimpleNamespace(config=config)
        with patch("backend.services.srm_service.load_model",return_value=(model,"cpu")):
            service=SRMService()
        observed=[]
        def infer(**kwargs):
            observed.append(kwargs["model"].config.denoiser_settings.sampling_steps)
            return self.source
        with patch("inference.geospatial.super_resolve",side_effect=infer), \
             patch("backend.services.srm_service.finish_output",return_value=self.source):
            for steps in (20,100,50):
                result=service.process(self.source,self.root,{"patches":1},sampling_steps=steps)
                self.assertEqual(result["sampling_steps"],steps)
                self.assertEqual(model.config.denoiser_settings.sampling_steps,100)
                with rasterio.open(self.source) as product:
                    self.assertEqual(product.tags()["sampling_steps"],str(steps))
        self.assertEqual(observed,[20,100,50])

    def test_failed_inference_restores_sampling_steps(self):
        model=SimpleNamespace(config=SimpleNamespace(denoiser_settings=SimpleNamespace(sampling_steps=100)))
        with patch("backend.services.srm_service.load_model",return_value=(model,"cpu")):
            service=SRMService()
        with patch("inference.geospatial.super_resolve",side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                service.process(self.source,self.root,{"patches":1},sampling_steps=20)
        self.assertEqual(model.config.denoiser_settings.sampling_steps,100)

    def test_detail_view_preserves_sr_pixels_without_smoothing(self):
        plan=plan_crop(self.source,AOI())
        crop,model_input=write_crop(self.source,self.root/"job",plan,[1,2,3,4],10000)
        product=StubService().process(model_input,self.root/"job",plan)
        _,detail=detail_images(crop.read_bytes(),Path(product["output_path"]).read_bytes(),10,12,32,32)
        pixels=np.asarray(detail)
        self.assertEqual(detail.size,(256,256))
        np.testing.assert_array_equal(pixels[::2,::2],pixels[1::2,1::2])
