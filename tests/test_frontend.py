from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from inference.aoi import AOI, inspect_raster, plan_crop
from inference.preview import create_preview
from tests.helpers import RasterTestCase


class Upload(BytesIO):
    name = "same-name.tif"


class FrontendTests(RasterTestCase):
    def setUp(self):
        super().setUp()
        self.uploaded = Upload(self.source.read_bytes())
        self.scene = dict(scene_id="a"*32,filename=self.uploaded.name,**inspect_raster(self.source))
        self.scene.update(create_preview(self.source,self.root/"preview.png",[1,2,3,4]))
        self.scene["preview_bounds_wgs84"] = self.scene["bounds_wgs84"]
        self.preview=(self.root/"preview.png").read_bytes()
        self.patches=[
            patch("streamlit.file_uploader",side_effect=lambda *a,**k:self.uploaded),
            patch("frontend.api_client.SRMApiClient.health",return_value={"model":{"model":"ESA LDSR-S2","device":"cpu","sampling_steps":100,"sampling_options":[20,50,100],"max_patches":4}}),
            patch("frontend.api_client.SRMApiClient.upload_scene",return_value=self.scene),
            patch("frontend.api_client.SRMApiClient.preview",return_value=self.preview),
            patch("frontend.api_client.SRMApiClient.plan",side_effect=lambda scene_id,aoi:plan_crop(self.source,AOI.model_validate(aoi))),
            patch("frontend.components.map_view.st_folium",return_value={"all_drawings":[]}),
        ]
        for context in self.patches:
            context.start()

    def tearDown(self):
        for context in reversed(self.patches):
            context.stop()
        super().tearDown()

    def inspected_app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"frontend/app.py"),default_timeout=20).run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label=="Inspect image").click().run()
        self.assertFalse(app.exception)
        return app

    def test_center_crop_and_pixel_budget_ui(self):
        app=self.inspected_app()
        self.assertIn("1",[metric.value for metric in app.metric])
        app.radio[0].set_value("Pixel rectangle").run()
        inputs={item.label:item for item in app.number_input}
        inputs["Width (pixels)"].set_value(256)
        inputs["Height (pixels)"].set_value(256)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("9 patches" in error.value for error in app.error))
        run=next(button for button in app.button if button.label=="Super-resolve selected area")
        self.assertTrue(run.disabled)

    def test_changed_contents_with_same_filename_clear_old_result(self):
        app=self.inspected_app()
        app.session_state["products"]={"job_id":"old"}
        self.uploaded=Upload(self.source.read_bytes()+b"changed contents")
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("scene",app.session_state.filtered_state)
        self.assertNotIn("products",app.session_state.filtered_state)

    def test_drawn_rectangle_produces_a_plan(self):
        app=self.inspected_app()
        west,south,east,north=self.scene["bounds_wgs84"]
        dx,dy=(east-west)/20,(north-south)/20
        coordinates=[[west+dx,south+dy],[west+2*dx,south+dy],[west+2*dx,south+2*dy],[west+dx,south+2*dy],[west+dx,south+dy]]
        with patch("frontend.components.map_view.st_folium",return_value={"all_drawings":[{"geometry":{"type":"Polygon","coordinates":[coordinates]}}]}):
            app.radio[0].set_value("Draw on map").run()
        self.assertFalse(app.exception)
        self.assertIn("1",[metric.value for metric in app.metric])
        self.assertFalse(next(button for button in app.button if button.label=="Super-resolve selected area").disabled)

    def test_quality_defaults_to_100_and_follows_submission(self):
        app=self.inspected_app()
        quality=next(item for item in app.selectbox if item.label=="Reconstruction quality")
        self.assertEqual(quality.value,100)
        quality.set_value(50).run()
        queued={"job_id":"test-job","status":"queued","sampling_steps":50,
                "progress":0,"message":"Queued","patches_completed":0,"plan":{"patches":1}}
        with patch("frontend.api_client.SRMApiClient.submit_aoi",return_value=queued) as submit, \
             patch("frontend.api_client.SRMApiClient.get_job",return_value=queued):
            next(button for button in app.button if button.label=="Super-resolve selected area").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(submit.call_args.kwargs["sampling_steps"],50)

    def test_backend_upgrade_selects_new_default(self):
        with patch("frontend.api_client.SRMApiClient.health",return_value={"model":{"model":"ESA LDSR-S2","device":"cpu","sampling_steps":20,"max_patches":4}}):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"frontend/app.py"),default_timeout=20).run()
            self.assertEqual(app.session_state["sampling_steps"],20)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["sampling_steps"],100)
