"""
Voxel Occupancy Grid Backend
Depth Anything V2 Large (metric) → point cloud → voxelization
"""

import os
import sys
import time
import base64
import io
import logging
import traceback

import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxel-backend")

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Voxel Occupancy Grid API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model state ──────────────────────────────────────────────────────
depth_pipe = None
model_name: Optional[str] = None
model_load_error: Optional[str] = None

CANDIDATE_MODELS = [
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
]


@app.on_event("startup")
async def startup_event():
    global depth_pipe, model_name, model_load_error
    from transformers import pipeline as hf_pipeline

    device_id = 0 if torch.cuda.is_available() else -1
    device_label = f"CUDA:{device_id}" if device_id >= 0 else "CPU"
    logger.info(f"Device: {device_label}")

    # Log memory info if CUDA
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {total_mem:.1f} GB")
    else:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        logger.info(
            f"RAM: {ram:.1f} GB  — CPU inference will be slow (minutes per frame)"
        )
        logger.warning(
            "No GPU detected. The Large model needs ~4–6 GB RAM. "
            "Expect very slow inference on CPU."
        )

    for candidate in CANDIDATE_MODELS:
        logger.info(f"Attempting to load model: {candidate}")
        try:
            depth_pipe = hf_pipeline(
                task="depth-estimation",
                model=candidate,
                device=device_id,
            )
            model_name = candidate
            logger.info(f"✅ Model loaded: {candidate}")
            break
        except Exception as exc:
            logger.error(
                f"❌ Failed to load {candidate}: {exc}\n{traceback.format_exc()}"
            )
            depth_pipe = None

    if depth_pipe is None:
        model_load_error = (
            "All model candidates failed to load. "
            "Check RAM/GPU constraints and HuggingFace connectivity."
        )
        logger.error(model_load_error)


# ─── Request / response models ───────────────────────────────────────────────
class ProcessFrameRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    # Camera intrinsics (pinhole model)
    fx: float = Field(525.0, description="Focal length x (pixels)")
    fy: float = Field(525.0, description="Focal length y (pixels)")
    cx: float = Field(320.0, description="Principal point x (pixels)")
    cy: float = Field(240.0, description="Principal point y (pixels)")
    # Voxel grid parameters
    voxel_size: float = Field(0.25, gt=0, description="Voxel edge length (metres)")
    max_depth: float = Field(5.0, gt=0, description="Max depth range (metres)")
    min_points: int = Field(2, ge=1, description="Min points to mark voxel occupied")


class Voxel(BaseModel):
    x: float
    y: float
    z: float
    depth: float  # Z coordinate (for coloring)
    count: int


class DepthStats(BaseModel):
    min_m: float
    max_m: float
    mean_m: float
    valid_pixels: int
    total_pixels: int


class ProcessFrameResponse(BaseModel):
    voxels: list[Voxel]
    timings: dict[str, float]  # stage → seconds
    depth_stats: DepthStats
    total_points: int
    occupied_voxels: int


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/depth/health")
def health():
    return {
        "status": "ok" if depth_pipe is not None else "model_not_loaded",
        "model": model_name,
        "error": model_load_error,
        "cuda": torch.cuda.is_available(),
    }


@app.get("/depth/model_info")
def model_info():
    mem_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_info["gpu_total_gb"] = round(props.total_memory / 1e9, 2)
        mem_info["gpu_reserved_gb"] = round(
            torch.cuda.memory_reserved(0) / 1e9, 2
        )
        mem_info["gpu_allocated_gb"] = round(
            torch.cuda.memory_allocated(0) / 1e9, 2
        )
    else:
        import psutil
        vm = psutil.virtual_memory()
        mem_info["ram_total_gb"] = round(vm.total / 1e9, 2)
        mem_info["ram_available_gb"] = round(vm.available / 1e9, 2)
        mem_info["ram_used_gb"] = round(vm.used / 1e9, 2)

    return {
        "model_name": model_name,
        "model_loaded": depth_pipe is not None,
        "load_error": model_load_error,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory": mem_info,
        "candidate_models": CANDIDATE_MODELS,
    }


@app.post("/depth/process", response_model=ProcessFrameResponse)
async def process_frame(req: ProcessFrameRequest):
    if depth_pipe is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Depth model not loaded. Check startup logs.",
        )

    timings: dict[str, float] = {}

    # ── Stage 1: Decode image ────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        # Strip data-URL prefix if present
        b64 = req.image
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = pil_image.size
    except Exception as exc:
        logger.error(f"Image decode failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")
    timings["decode_s"] = round(time.perf_counter() - t0, 4)
    logger.info(
        f"[stage 1/5] decoded image {W}×{H}  ({timings['decode_s']*1000:.1f} ms)"
    )

    # ── Stage 2: Depth inference ─────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        result = depth_pipe(pil_image)
        # HF depth-estimation pipeline returns two outputs:
        #   result["predicted_depth"] — PyTorch tensor, raw metric values (metres for metric models)
        #   result["depth"]           — PIL Image, display-normalized (NOT metric)
        # We MUST use predicted_depth for correct 3-D back-projection.
        if "predicted_depth" in result:
            pd = result["predicted_depth"]
            # Shape may be (1, H, W) or (H, W); squeeze out any leading batch dim.
            if hasattr(pd, "squeeze"):
                pd = pd.squeeze()
            depth_array = pd.detach().cpu().numpy().astype(np.float32)
            logger.info(
                f"Using predicted_depth tensor  shape={depth_array.shape}  "
                f"range=[{depth_array.min():.3f}, {depth_array.max():.3f}] m"
            )
        else:
            # Fallback: older pipeline versions may only return "depth".
            # NOTE: this path gives display-normalized values — accuracy is degraded.
            logger.warning(
                "predicted_depth not found in pipeline output — falling back to 'depth' "
                "PIL image. Values may not be metric. Upgrade transformers if possible."
            )
            raw_depth = result["depth"]
            if isinstance(raw_depth, Image.Image):
                depth_array = np.array(raw_depth, dtype=np.float32)
            else:
                depth_array = np.asarray(raw_depth, dtype=np.float32)
    except Exception as exc:
        logger.error(f"Depth inference failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Depth inference error: {exc}")
    timings["depth_inference_s"] = round(time.perf_counter() - t1, 4)

    # Resize depth map to match image if needed
    if depth_array.shape != (H, W):
        depth_pil = Image.fromarray(depth_array)
        depth_pil = depth_pil.resize((W, H), Image.BILINEAR)
        depth_array = np.array(depth_pil, dtype=np.float32)

    # The metric model outputs metres directly.
    # Guard against zero/negative/NaN values.
    valid_mask_all = np.isfinite(depth_array) & (depth_array > 0)
    total_valid = int(np.sum(valid_mask_all))
    total_pixels = H * W

    d_min = float(np.min(depth_array[valid_mask_all])) if total_valid > 0 else 0.0
    d_max = float(np.max(depth_array[valid_mask_all])) if total_valid > 0 else 0.0
    d_mean = float(np.mean(depth_array[valid_mask_all])) if total_valid > 0 else 0.0

    logger.info(
        f"[stage 2/5] depth inference done  ({timings['depth_inference_s']*1000:.1f} ms)  "
        f"depth range: [{d_min:.2f} m – {d_max:.2f} m]  mean={d_mean:.2f} m"
    )

    # ── Stage 3: Back-project to 3D point cloud ──────────────────────────────
    t2 = time.perf_counter()
    uu = np.arange(W, dtype=np.float32)
    vv = np.arange(H, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(uu, vv)  # both (H, W)

    Z = depth_array
    range_mask = valid_mask_all & (Z <= req.max_depth)

    Zf = Z[range_mask]
    Xf = (grid_u[range_mask] - req.cx) * Zf / req.fx
    Yf = (grid_v[range_mask] - req.cy) * Zf / req.fy

    total_points = int(Xf.size)
    timings["backproject_s"] = round(time.perf_counter() - t2, 4)
    logger.info(
        f"[stage 3/5] back-projected {total_points:,} points within "
        f"{req.max_depth} m  ({timings['backproject_s']*1000:.1f} ms)"
    )

    if total_points == 0:
        logger.warning("No valid points after range filtering; returning empty grid.")
        return ProcessFrameResponse(
            voxels=[],
            timings=timings,
            depth_stats=DepthStats(
                min_m=d_min,
                max_m=d_max,
                mean_m=d_mean,
                valid_pixels=total_valid,
                total_pixels=total_pixels,
            ),
            total_points=0,
            occupied_voxels=0,
        )

    # ── Stage 4: Voxelization ────────────────────────────────────────────────
    t3 = time.perf_counter()
    vs = req.voxel_size
    xi = np.floor(Xf / vs).astype(np.int32)
    yi = np.floor(Yf / vs).astype(np.int32)
    zi = np.floor(Zf / vs).astype(np.int32)

    # Unique voxels + per-voxel point counts
    voxel_coords = np.column_stack([xi, yi, zi])
    # np.unique on (N,3) treats each row as a key
    unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)

    occupied_mask = counts >= req.min_points
    occupied_voxels_arr = unique_voxels[occupied_mask]
    occupied_counts = counts[occupied_mask]

    timings["voxelize_s"] = round(time.perf_counter() - t3, 4)
    logger.info(
        f"[stage 4/5] voxelized → {len(occupied_voxels_arr):,} occupied voxels "
        f"(voxel_size={vs} m, min_pts={req.min_points})  "
        f"({timings['voxelize_s']*1000:.1f} ms)"
    )

    # ── Stage 5: Serialise response ──────────────────────────────────────────
    t4 = time.perf_counter()
    voxels_out: list[Voxel] = []
    for (vx, vy, vz), cnt in zip(occupied_voxels_arr, occupied_counts):
        cx_m = (float(vx) + 0.5) * vs
        cy_m = (float(vy) + 0.5) * vs
        cz_m = (float(vz) + 0.5) * vs
        voxels_out.append(
            Voxel(x=cx_m, y=cy_m, z=cz_m, depth=cz_m, count=int(cnt))
        )

    timings["serialize_s"] = round(time.perf_counter() - t4, 4)
    timings["total_s"] = round(sum(timings.values()), 4)
    logger.info(
        f"[stage 5/5] serialised {len(voxels_out)} voxels  "
        f"total={timings['total_s']*1000:.0f} ms"
    )

    return ProcessFrameResponse(
        voxels=voxels_out,
        timings=timings,
        depth_stats=DepthStats(
            min_m=round(d_min, 3),
            max_m=round(d_max, 3),
            mean_m=round(d_mean, 3),
            valid_pixels=total_valid,
            total_pixels=total_pixels,
        ),
        total_points=total_points,
        occupied_voxels=len(voxels_out),
    )
