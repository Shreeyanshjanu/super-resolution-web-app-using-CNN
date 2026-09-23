"""Time one real 128x128 LDSR-S2 crop, including a geospatial output check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from inference.aoi import AOI, inspect_raster, plan_crop, write_crop, finish_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--band-order", choices=["auto", "rgbn", "bgrn"], default="auto")
    parser.add_argument("--value-scale", type=int, choices=[1, 10000], default=10000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, help="Optional CPU thread count.")
    parser.add_argument("--sampling-steps", type=int, choices=[20, 50, 100], default=100)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable comparisons.")
    args = parser.parse_args()
    if args.threads is not None:
        import os
        os.environ["SRM_CPU_THREADS"] = str(args.threads)
    directory = args.output_dir or Path("outputs/benchmarks") / uuid4().hex
    meta = inspect_raster(args.input, args.band_order, args.value_scale)
    plan = plan_crop(args.input, AOI(), max_patches=1)
    crop, model_input = write_crop(args.input, directory, plan, meta["band_indexes"], args.value_scale)
    print(json.dumps(plan, indent=2), flush=True)
    # Imports and checkpoint loading are timed separately from inference.
    loading = perf_counter()
    from inference.model import load_model
    from inference.geospatial import super_resolve
    model, device = load_model(device=args.device, sampling_steps=args.sampling_steps)
    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)
    load_seconds = perf_counter() - loading
    start = perf_counter()
    raw = super_resolve(model_input, model, device, expected_patches=1, max_patches=1)
    inference_seconds = perf_counter() - start
    output = finish_output(raw, crop, directory / "result.tif")
    import rasterio
    import torch
    import numpy as np
    with rasterio.open(output, "r+") as result:
        result.update_tags(sampling_steps=str(args.sampling_steps), seed=str(args.seed))
    with rasterio.open(crop) as source, rasterio.open(output) as result:
        assert result.width == source.width * 4 and result.height == source.height * 4
        assert result.crs == source.crs
        assert np.allclose(result.bounds, source.bounds)
        assert np.allclose(result.res, np.asarray(source.res) / 4)
    report = dict(input=str(args.input.resolve()), output=str(output.resolve()), device=device,
                  sampling_steps=args.sampling_steps, seed=args.seed, load_seconds=load_seconds, inference_seconds=inference_seconds,
                  cpu_threads=torch.get_num_threads(),
                  plan=plan, geospatial_checks="passed", warnings=meta["warnings"])
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
