import unittest

from inference.aoi import patch_count


class PipelineContractTests(unittest.TestCase):
    def test_patch_budget_matches_installed_opensr(self):
        from opensr_utils.pipeline import large_file_processing
        pipeline=large_file_processing.__new__(large_file_processing)
        pipeline.overlap=8
        pipeline.window_size=(128,128)
        for width,height in [(128,128),(129,128),(248,248),(256,256),(512,512),(488,128)]:
            pipeline.image_meta={"width":width,"height":height}
            windows=pipeline.create_image_windows()
            self.assertEqual(len(windows),patch_count(width,height),(width,height))
            self.assertTrue(all(window.width==128 and window.height==128 for window in windows))
