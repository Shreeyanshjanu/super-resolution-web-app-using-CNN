import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import UploadLimit
from backend.routes.super_resolution import router
from backend.services.job_manager import JobManager, QueueFullError
from backend.services.storage import SceneStore
from inference.aoi import finish_output
from tests.helpers import RasterTestCase


class StubService:
    """Deterministic test double; production always uses ESA LDSR-S2."""
    def __init__(self):
        self.calls = []
        self.sampling_calls = []

    def process(self,input_path,job_directory,plan,progress_callback=None,sampling_steps=100):
        self.sampling_calls.append(sampling_steps)
        with rasterio.open(input_path) as src:
            self.calls.append(src.shape)
            data = np.repeat(np.repeat(src.read(),4,axis=1),4,axis=2)
            profile = src.profile.copy()
            profile.update(width=src.width*4,height=src.height*4,transform=src.transform*Affine.scale(0.25))
        raw = job_directory/"raw.tif"
        with rasterio.open(raw,"w",**profile) as dst:
            dst.write(data)
        if progress_callback:
            progress_callback(plan["patches"],plan["patches"])
        output = finish_output(raw,job_directory/"crop.tif",job_directory/"result.tif")
        return {"output_path":str(output),"inference_seconds":0.01}


class APITests(RasterTestCase):
    def setUp(self):
        super().setUp()
        self.service = StubService()
        self.manager = JobManager(self.service,self.root/"jobs",max_pending=2)
        self.app = FastAPI()
        self.app.state.job_manager = self.manager
        self.app.state.scenes = SceneStore(self.root/"scenes")
        self.app.include_router(router)
        self.app.add_middleware(UploadLimit)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.manager.shutdown()
        self.client.close()
        super().tearDown()

    def upload(self):
        response = self.client.post("/api/scenes",files={"file":("source.tif",self.source.read_bytes(),"image/tiff")})
        self.assertEqual(response.status_code,201,response.text)
        return response.json()

    def wait_job(self,job_id):
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            result=self.client.get(f"/api/jobs/{job_id}").json()
            if result["status"] in ("completed","failed"):
                return result
            time.sleep(0.02)
        self.fail("Test job timed out")

    def test_upload_plan_process_download_only_aoi(self):
        scene = self.upload()
        self.assertEqual(scene["full_image_patches"],25)
        self.assertEqual(self.client.get(f"/api/scenes/{scene['scene_id']}/preview").status_code,200)
        plan = self.client.post(f"/api/scenes/{scene['scene_id']}/plan",json={"mode":"center"})
        self.assertEqual(plan.json()["patches"],1)
        response=self.client.post("/api/super-resolve",json={"scene_id":scene["scene_id"],"aoi":{"mode":"center"}})
        self.assertEqual(response.status_code,202,response.text)
        job=self.wait_job(response.json()["job_id"])
        self.assertEqual(job["status"],"completed",job)
        self.assertEqual(self.service.calls,[(128,128)])
        self.assertEqual(job["sampling_steps"],100)
        self.assertEqual(self.service.sampling_calls,[100])
        self.assertEqual(job["patches_completed"],1)
        self.assertNotIn("input_path",job)
        data=self.client.get(f"/api/jobs/{job['job_id']}/download").content
        with rasterio.MemoryFile(data) as mem,mem.open() as src:
            self.assertEqual(src.shape,(512,512))
            self.assertEqual(src.res,(2.5,2.5))
            self.assertEqual(str(src.crs),"EPSG:32613")
        self.assertEqual(self.client.get(f"/api/jobs/{job['job_id']}/download?product=crop").status_code,200)

    def test_corrupt_or_wrong_extension_rejected(self):
        for filename,status in [("bad.tif",422),("bad.png",400)]:
            response=self.client.post("/api/scenes",files={"file":(filename,b"not a raster")})
            self.assertEqual(response.status_code,status,response.text)
        self.assertEqual(list((self.root/"scenes").iterdir()),[])

    def test_large_aoi_and_missing_selection_rejected(self):
        scene=self.upload()
        aoi={"mode":"pixel","x":0,"y":0,"width":512,"height":512}
        result=self.client.post("/api/super-resolve",json={"scene_id":scene["scene_id"],"aoi":aoi})
        self.assertEqual(result.status_code,422)
        self.assertEqual(self.service.calls,[])
        self.assertEqual(self.client.post("/api/super-resolve",json={"scene_id":scene["scene_id"]}).status_code,422)

    def test_sampling_choice_is_validated_and_sent_to_worker(self):
        scene=self.upload()
        submission={"scene_id":scene["scene_id"],"aoi":{"mode":"center"},"sampling_steps":50}
        result=self.client.post("/api/super-resolve",json=submission)
        self.assertEqual(result.status_code,202,result.text)
        job=self.wait_job(result.json()["job_id"])
        self.assertEqual(job["sampling_steps"],50)
        self.assertEqual(self.service.sampling_calls,[50])
        for invalid in (0, 21, 1000, "100", 50.5):
            submission["sampling_steps"]=invalid
            self.assertEqual(self.client.post("/api/super-resolve",json=submission).status_code,422)

    def test_upload_limits(self):
        with patch("backend.routes.super_resolution.MAX_UPLOAD_BYTES",10):
            result=self.client.post("/api/scenes",files={"file":("large.tif",b"x"*11)})
            self.assertEqual(result.status_code,413,result.text)
        result=self.client.post("/api/scenes",content=b"",headers={"content-length":str(1024**3)})
        self.assertEqual(result.status_code,413)
        self.assertEqual(list((self.root/"scenes").iterdir()),[])

    def test_queue_is_bounded(self):
        first=self.manager.create_job("one.tif")
        self.manager.create_job("two.tif")
        with self.assertRaises(QueueFullError):
            self.manager.create_job("three.tif")
        self.assertEqual(self.client.get(f"/api/jobs/{first}/download").status_code,409)

    def test_persistent_records_and_interrupted_recovery(self):
        identifier=self.manager.create_job("interrupted.tif")
        self.manager.update_job(identifier,status="processing")
        self.manager.shutdown()
        self.manager=JobManager(self.service,self.root/"jobs")
        self.manager.recover()
        result=self.manager.get_job(identifier)
        self.assertEqual(result["status"],"failed")
        self.assertIn("Server stopped",result["error"])

    def test_safe_cleanup(self):
        scene=self.upload()
        response=self.client.post("/api/super-resolve",json={"scene_id":scene["scene_id"],"aoi":{"mode":"center"}})
        job=self.wait_job(response.json()["job_id"])
        self.assertEqual(self.client.delete(f"/api/scenes/{scene['scene_id']}").status_code,409)
        self.assertEqual(self.client.delete(f"/api/jobs/{job['job_id']}").status_code,200)
        self.assertEqual(self.client.delete(f"/api/scenes/{scene['scene_id']}").status_code,200)
        self.assertEqual(self.client.get(f"/api/jobs/{job['job_id']}").status_code,404)
        self.assertTrue(self.source.exists())

    def test_get_job_returns_independent_snapshot(self):
        identifier=self.manager.create_job("a.tif")
        job=self.manager.get_job(identifier)
        job["status"]="completed"
        self.assertEqual(self.manager.get_job(identifier)["status"],"preparing")

    def test_queued_job_resumes_and_completed_download_survives_restart(self):
        scene=self.upload()
        with patch.object(self.manager,"enqueue"):
            response=self.client.post("/api/super-resolve",json={"scene_id":scene["scene_id"],"aoi":{"mode":"center"}})
        identifier=response.json()["job_id"]
        self.assertEqual(self.manager.get_job(identifier)["status"],"queued")
        self.manager.shutdown()
        self.manager=JobManager(self.service,self.root/"jobs")
        self.app.state.job_manager=self.manager
        self.manager.recover()
        self.assertEqual(self.wait_job(identifier)["status"],"completed")
        self.manager.shutdown()
        self.manager=JobManager(self.service,self.root/"jobs")
        self.app.state.job_manager=self.manager
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}/download").status_code,200)

    def test_chunked_request_limit_before_multipart(self):
        with patch("backend.main.MAX_UPLOAD_BYTES",1):
            chunks=(b"x"*(600*1024) for _ in range(2))
            response=self.client.post("/api/scenes",content=chunks,
                                      headers={"content-type":"multipart/form-data; boundary=example"})
        self.assertEqual(response.status_code,413,response.text)
