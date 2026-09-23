"""Run against tests.browser_server:18000 and Streamlit:18501 (test inference only)."""
from pathlib import Path
from time import monotonic
from playwright.sync_api import sync_playwright, expect, Error
from tests.helpers import make_raster

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/"outputs"/"browser-test"


def main():
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    source=make_raster(ARTIFACTS/"browser-input.tif")
    with sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        page=browser.new_page(viewport={"width":1440,"height":1050})
        try:
            page.goto("http://127.0.0.1:18501")
            page.locator("input[type=file]").wait_for(state="attached",timeout=30000)
            expect(page.get_by_role("combobox",name="Reconstruction quality")).to_have_value("Detailed — 100 steps")
            page.locator("input[type=file]").set_input_files(str(source))
            page.get_by_role("button",name="Inspect image",exact=True).click()
            page.get_by_text("2. Select the area to process",exact=True).wait_for(timeout=30000)
            page.get_by_text("Draw on map",exact=True).click()
            deadline=monotonic()+30
            map_frame=None
            while monotonic()<deadline:
                try:
                    map_frame=next((frame for frame in page.frames if frame.locator(".leaflet-draw-draw-rectangle").count()),None)
                except Error:
                    map_frame=None  # Streamlit replaces the iframe on a mode change.
                if map_frame:
                    break
                page.wait_for_timeout(250)
            assert map_frame, "Rectangle drawing control did not load"
            canvas=map_frame.locator(".leaflet-container")
            canvas.scroll_into_view_if_needed()
            map_frame.locator(".leaflet-draw-draw-rectangle").click()
            box=canvas.bounding_box()
            page.mouse.move(box["x"]+box["width"]*0.45,box["y"]+box["height"]*0.45)
            page.mouse.down()
            page.mouse.move(box["x"]+box["width"]*0.50,box["y"]+box["height"]*0.50,steps=10)
            page.mouse.up()
            run=page.get_by_role("button",name="Super-resolve selected area",exact=True)
            expect(run).to_be_enabled(timeout=20000)
            metric=page.get_by_test_id("stMetric").filter(has_text="Model patches")
            expect(metric).to_contain_text("1")
            page.screenshot(path=str(ARTIFACTS/"aoi-selected.png"),full_page=True)
            run.click()
            download=page.get_by_role("button",name="Download SR GeoTIFF",exact=True)
            download.wait_for(timeout=30000)
            expect(page.get_by_text("This result uses 100 sampling steps.",exact=True)).to_be_visible()
            page.get_by_text("Inspect fine detail",exact=True).click()
            expect(page.get_by_text("Original · bicubic enlarged",exact=True)).to_be_visible()
            expect(page.get_by_text("LDSR-S2 · 100 steps",exact=True)).to_be_visible()
            expect(page.get_by_text("Input: 10 m · Output grid: 2.5 m · Geographic extent preserved.",exact=True)).to_be_visible(timeout=20000)
            with page.expect_download() as download_info:
                download.click()
            download_info.value.save_as(str(ARTIFACTS/"downloaded-sr.tif"))
            page.screenshot(path=str(ARTIFACTS/"completed.png"),full_page=True)
            assert not page.get_by_test_id("stException").count()
            print("BROWSER PASS: upload, map drawing, one-patch plan, job completion, comparison, geospatial map, TIFF download.")
        finally:
            page.screenshot(path=str(ARTIFACTS/"latest.png"),full_page=True)
            browser.close()


if __name__=="__main__":
    main()
