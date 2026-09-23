# Satellite SRM

A FastAPI + Streamlit application for **selected-area** super-resolution of Sentinel-2 L2A RGB/NIR imagery with the pretrained ESA LDSR-S2 model.

Upload TIFF → inspect metadata and preview → select a center crop, pixel rectangle, or map rectangle → preview the patch budget → crop on the original grid → process only that crop → download the original AOI and 4× SR GeoTIFF.

## Setup on Windows

Tested with Python 3.13.1. Run these commands from the project root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m scripts.download_model
~~~

The checkpoint is approximately 1.13 GB. The download command verifies an existing file instead of downloading it again. The tracked model configuration comes from opensr-model 1.1.1; see [configs/README.md](configs/README.md).

In one terminal:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --workers 1
~~~

In a second terminal:

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
~~~

Open http://localhost:8501. API documentation: http://127.0.0.1:8000/docs. Wait for backend model loading to complete before using the frontend. Use one worker: each model instance consumes substantial memory, and a storage lock prevents conflicting workers. The lock is released by the OS if the process exits.

## Input requirements

- Exactly four bands: B04 (red), B03 (green), B02 (blue), B08 (NIR).
- Auto band detection uses those band descriptions. For unlabelled images, explicitly select RGBN or BGRN. Crops and outputs are always written as RGBN.
- Select either surface reflectance × 10,000 or reflectance in [0,1]. Values outside the selected range, unsupported embedded scale/offset metadata, and unmasked NaN/infinite values are rejected. A bounded sample is checked at upload; every selected pixel is checked before inference.
- Geographic processing requires a north-up 10 m projected raster with metre units. Reproject unsupported grids before upload.
- Images without a CRS are accepted for **pixel-only experiments**. Their coordinate system is preserved without inventing a location or a ground resolution; map selection and metre-based output claims are unavailable.

The 300 existing prepared test TIFFs have no CRS. They can exercise the CPU crop workflow but cannot demonstrate geographically located 2.5 m products. Obtain georeferenced originals to restore real coordinates. The existing SEN2NAIP demo at `data/raw/demo/demo/cross-sensor/ROI_0000/lr.tif` is georeferenced; select **RGBN** because its bands are unlabelled. Dataset files are not included in Git.

`scripts/prepare_test_images.py` now checks matching RGB/NIR grids and writes correctly labelled RGBN files. Existing prepared files are not overwritten; their BGRN descriptions are recognized and reordered when cropped.

## CPU workflow

The model configuration remains **128×128 inputs, factor 4, overlap 8, batch size 1, and num_workers 0**. Reconstruction now defaults to **100 sampling steps**, matching the upstream default. Choose Detailed (100), Balanced (50), or Quick preview (20) in the sidebar. Each job saves and uses its own selection, including after a restart.

100 steps performs five times as many denoising iterations as the original preview setting. It takes longer and may produce better reconstructed detail, but more steps do not guarantee better accuracy for every scene. Existing outputs must be regenerated; changing the selector does not alter a finished TIFF. Restart FastAPI after updating this code to enable the new options.

A real 100-step run on the same 128×128 test crop took **1325.7 seconds (22 minutes 6 seconds)** on this machine during development. Its visual change from the saved 20-step result was modest; extra sampling time alone did not produce a dramatic clarity improvement. This is not a controlled quality benchmark (the earlier job did not record its random seed). The report and TIFF are under `outputs/quality-comparison/steps100/`, with `outputs/quality-comparison/comparison.png` showing the shared-contrast comparison.

Use **Inspect fine detail** below the comparison to compare native SR pixels against a labelled bicubic baseline. Both use the same area and contrast. Integer magnification avoids browser smoothing; the TIFF is not sharpened or otherwise altered by the viewer.

- Center crop: at most 128×128 source pixels → one model patch.
- A 248×248 rectangle → four patches.
- A 256×256 rectangle → nine patches because of overlapping edge windows, so the default four-patch limit rejects it.
- A 512×512 image would need 25 patches. The app never sends it automatically.
- The plan and actual OpenSR patch count must agree before inference starts.
- Map rectangles are transformed from WGS84, snapped outward to native pixels, and intersected with the source extent. The displayed plan is the actual processed rectangle.
- Crops smaller than 128 pixels along an axis are edge-padded for inference. The final product removes this padding, retains the AOI extent, divides pixel spacing by four, and restores the valid-data mask.
- Map previews are downsampled and warped to Web Mercator for display only. The inference data is not reprojected or stretched.

Four CPU threads are the default, based on the local benchmark. Override `SRM_CPU_THREADS` to benchmark your hardware. This changes CPU scheduling, not the architecture or sampling count. Cropping reduces available image context, so boundary details can differ from a full-scene run.

### Measured locally

Historical **20-step** measurements on an Intel Core i5-13420H, using a center 128×128 crop from `001_10m_B2B3B4B8.tif`:

| Configuration | Patches | Inference + stitching |
|---|---:|---:|
| Original full-image workflow | 25 | User-reported 45–50 minutes |
| AOI, default PyTorch 8 threads | 1 | 195.6 seconds |
| AOI, 4 threads | 1 | 126.9 seconds |
| Geographic AOI via API, 4 threads (separate run) | 1 | 199.3 seconds |

Checkpoint/import loading took another 19.8 seconds in the four-thread benchmark; the backend pays this once at startup. Timings vary with CPU load and image content. The fixed-crop result is about 2 minutes 7 seconds; the geographic API run took 3 minutes 19 seconds. Sub-two-minute processing is not guaranteed.

Repeat the original 20-step actual-model benchmark (use `--sampling-steps 100` for the new detailed setting):

~~~powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_aoi data/test_images/prepared/001_10m_B2B3B4B8.tif --threads 4 --sampling-steps 20
~~~

For the georeferenced demo:

~~~powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_aoi data/raw/demo/demo/cross-sensor/ROI_0000/lr.tif --band-order rgbn
~~~

Reports and products are written under `outputs/benchmarks/`. The demo is 121×121 and exercises padding and removal.

## API

1. `POST /api/scenes`: multipart `file`, `band_order=auto|rgbn|bgrn`, `value_scale=10000|1`. Upload once; returns a scene ID, metadata and warnings.
2. `GET /api/scenes/{scene_id}/preview`: bounded PNG preview.
3. `POST /api/scenes/{scene_id}/plan`: an AOI JSON object.
4. `POST /api/super-resolve`: `{"scene_id":"...","aoi":{"mode":"center"},"sampling_steps":100}` → HTTP 202 and persistent job ID. Allowed sampling counts: 20, 50, 100; omitted defaults to 100.
5. `GET /api/jobs/{job_id}`: queue state, completed patches, plan and actual timing.
6. `GET /api/jobs/{job_id}/download`: SR TIFF; use `?product=crop` for the original selected area.

Other AOIs:

~~~json
{"mode":"pixel","x":192,"y":192,"width":128,"height":128}
~~~

~~~json
{"mode":"bbox","bbox":[-106.229,37.096,-106.222,37.102]}
~~~

Bboxes are west, south, east, north in WGS84. Arbitrary polygons and antimeridian crossing are unsupported. The old multipart `/api/super-resolve` upload contract was replaced: callers must inspect/upload a scene and explicitly select an AOI.

## Jobs and storage

SQLite job records and per-scene metadata live under `outputs/app/`. A single executor serializes model use; up to eight active/queued jobs are accepted. Queue overflow returns HTTP 429.

Completed jobs remain downloadable after restart. Queued jobs resume; interrupted preparation/inference is marked failed with a resubmission message. Shutdown finishes the current job and leaves pending jobs queued.

Use Saved jobs in the sidebar to reopen or delete completed/failed jobs. Delete a source through Source details after removing its jobs. Equivalent endpoints: `GET /api/jobs`, `DELETE /api/jobs/{id}`, `DELETE /api/scenes/{id}`. Downloads remain until explicitly deleted; there is no automatic expiry. Failed uploads are cleaned immediately.

This is a local single-user application. Keep it bound to localhost unless authentication and per-user access controls are added.

## Configuration

| Environment variable | Default |
|---|---|
| SRM_API_URL | http://127.0.0.1:8000 |
| SRM_DATA_ROOT | outputs/app under the project root |
| SRM_CHECKPOINT | project-root opensr-ldsrs2_v1_0_0.ckpt |
| SRM_CPU_THREADS | 4, capped to available logical CPUs |
| SRM_MAX_PATCHES | 4 |
| SRM_MAX_PENDING_JOBS | 8 |
| SRM_MAX_UPLOAD_MB | 256 |

The Streamlit upload limit is also set to 256 MB in `.streamlit/config.toml`; change both if needed. API body limits apply before multipart parsing and again while saving the actual file.

## Validation and scientific limits

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
~~~

Fast tests use a deterministic inference double to check input validation, patch budgets, metadata, padding, band order, queue limits, persistence and downloads. They do not establish model quality. The benchmark runs the actual pretrained model.

Automated tests include Streamlit state regressions, per-job sampling settings, native pixel detail inspection, and agreement with the installed OpenSR patch planner. A real-model API run also verified a WGS84 AOI, patch progress, padding removal, original extent/CRS and a 2.5 m output grid.

Optional browser test (requires installed Microsoft Edge): install `requirements-browser.txt`, run `python -m uvicorn tests.browser_server:app --port 18000`, then run Streamlit on port 18501 with `SRM_API_URL=http://127.0.0.1:18000`. Run `python -m tests.browser_smoke` in a third terminal. This test server is explicitly a fast interpolation double; it checks drawing, previews, job UI and downloads, not model inference. Screenshots and downloads go to `outputs/browser-test/`.

A denser 2.5 m output grid is a learned estimate, not a native 2.5 m observation. Existing saved demo evaluation scored slightly below bicubic (PSNR 18.120 vs 18.346 dB; SSIM 0.37530 vs 0.39278) under the original histogram-matched cross-sensor evaluation. Do not infer quality improvement from sharper previews. Validate more registered reference tiles and report uncertainty before making scientific claims.

The default was increased from 20 to 100 to prioritize reconstruction over CPU runtime. The pretrained weights and model architecture are unchanged. Lower-step modes remain available for previews.

LDSR-S2 sampling is stochastic. The current app does not fix random seeds, so repeated jobs need not be pixel-identical even with the same inputs and settings. The benchmark CLI accepts `--seed` (default 42) for controlled comparisons.

```
satellite-srm
├─ .streamlit
│  └─ config.toml
├─ backend
│  ├─ main.py
│  ├─ routes
│  │  ├─ health.py
│  │  ├─ super_resolution.py
│  │  └─ __init__.py
│  ├─ services
│  │  ├─ job_manager.py
│  │  ├─ srm_service.py
│  │  ├─ storage.py
│  │  └─ __init__.py
│  ├─ settings.py
│  └─ __init__.py
├─ configs
│  ├─ ldsrs2.yaml
│  └─ README.md
├─ evaluation
│  ├─ check_spatial_detail.py
│  ├─ evaluate.py
│  ├─ inspect_results.py
│  ├─ metrics.py
│  └─ visualize_results.py
├─ frontend
│  ├─ api_client.py
│  ├─ app.py
│  ├─ components
│  │  ├─ comparison.py
│  │  ├─ map_view.py
│  │  ├─ results.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ inference
│  ├─ aoi.py
│  ├─ geospatial.py
│  ├─ model.py
│  ├─ predict.py
│  ├─ preview.py
│  ├─ test_esa_model.py
│  ├─ test_pipeline.py
│  ├─ test_real_image.py
│  └─ __init__.py
├─ models
├─ notebooks
├─ preprocessing
│  └─ inspect_sen2naip.py
├─ README.md
├─ requirements-browser.txt
├─ requirements-dev.txt
├─ requirements-lock.txt
├─ requirements.txt
├─ scripts
│  ├─ benchmark_aoi.py
│  ├─ compare_quality.py
│  ├─ download_model.py
│  └─ prepare_test_images.py
├─ tests
│  ├─ browser_server.py
│  ├─ browser_smoke.py
│  ├─ helpers.py
│  ├─ test_aoi.py
│  ├─ test_api.py
│  ├─ test_frontend.py
│  ├─ test_pipeline_contract.py
│  ├─ test_quality.py
│  └─ __init__.py
└─ training

```