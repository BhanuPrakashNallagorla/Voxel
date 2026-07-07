"""
Voxel Occupancy Grid Backend — async job pattern
Depth Anything V2 Large (metric) → point cloud → voxelization

Endpoints:
  GET  /depth/health              — server + model liveness
  GET  /depth/model_info          — detailed memory / device info
  POST /depth/scan                — start a job, return job_id immediately
  GET  /depth/status/{job_id}     — poll job progress / result
"""

import os
import sys
import time
import base64
import io
import logging
import traceback
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxel-backend")

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Voxel Occupancy Grid API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Thread pool (max 1 worker — serialises GPU/CPU inference) ────────────────
_executor = ThreadPoolExecutor(max_workers=1)

# ─── In-memory job store (capped at 50 entries) ───────────────────────────────
MAX_JOBS = 50
jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
_jobs_lock = threading.Lock()   # guards _add_job eviction logic


def _add_job(job_id: str) -> dict:
    """Create a blank job.  Evict only *terminal* (done/error) jobs when over cap.
    Never evicts pending/running jobs — that would break in-flight polling.
    If cap is exceeded and no terminal jobs exist, raises RuntimeError (429 upstream).
    """
    job: dict[str, Any] = {
        "id": job_id,
        "status": "pending",          # pending | running | done | error
        "current_stage": None,
        "stages": [],                 # list of stage-event dicts (see _record)
        "result": None,
        "created_at": time.time(),
    }
    with _jobs_lock:
        jobs[job_id] = job
        if len(jobs) > MAX_JOBS:
            # Collect terminal job IDs in insertion order (oldest first)
            terminal = [jid for jid, j in jobs.items()
                        if j["status"] in ("done", "error") and jid != job_id]
            need_to_free = len(jobs) - MAX_JOBS
            if len(terminal) < need_to_free:
                # Cannot safely evict — remove the new job and raise
                del jobs[job_id]
                raise RuntimeError(
                    f"Job store full ({MAX_JOBS} entries, none terminal). "
                    "Wait for a scan to complete before starting another."
                )
            for jid in terminal[:need_to_free]:
                del jobs[jid]
    return job


def _record(job: dict, stage: str, state: str, msg: str = "", error: str = "") -> None:
    """Append a stage-event entry and update job.current_stage.

    state is one of: "start" | "done" | "error"
    """
    job["stages"].append({
        "stage": stage,
        "state": state,
        "ts": round(time.time(), 3),
        "msg": msg,
        "error": error,
    })
    job["current_stage"] = stage
    if state == "error":
        job["status"] = "error"


# ─── Global model state ───────────────────────────────────────────────────────
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

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {total_mem:.1f} GB")
    else:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        logger.info(f"RAM: {ram:.1f} GB  — CPU inference will be slow (minutes per frame)")
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
            logger.error(f"❌ Failed to load {candidate}: {exc}\n{traceback.format_exc()}")
            depth_pipe = None

    if depth_pipe is None:
        model_load_error = (
            "All model candidates failed to load. "
            "Check RAM/GPU constraints and HuggingFace connectivity."
        )
        logger.error(model_load_error)


# ─── Request models ───────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    fx: float = Field(525.0, description="Focal length x (pixels)")
    fy: float = Field(525.0, description="Focal length y (pixels)")
    cx: float = Field(320.0, description="Principal point x (pixels)")
    cy: float = Field(240.0, description="Principal point y (pixels)")
    voxel_size: float = Field(0.25, gt=0, description="Voxel edge length (metres)")
    max_depth: float = Field(5.0, gt=0, description="Max depth range (metres)")
    min_points: int = Field(2, ge=1, description="Min points to mark voxel occupied")


# ─── Pipeline (runs in a background thread) ───────────────────────────────────
def _run_pipeline(job_id: str, req: ScanRequest) -> None:
    """Execute all 5 pipeline stages, writing progress into jobs[job_id]."""
    job = jobs[job_id]
    job["status"] = "running"

    # ── Stage 1: Decode image ─────────────────────────────────────────────────
    _record(job, "DECODE", "start")
    t0 = time.perf_counter()
    try:
        b64 = req.image
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = pil_image.size
    except Exception as exc:
        _record(job, "DECODE", "error",
                error=f"{exc}\n{traceback.format_exc()}")
        return
    decode_s = time.perf_counter() - t0
    _record(job, "DECODE", "done", msg=f"{decode_s*1000:.1f} ms  |  {W}×{H} px")
    logger.info(f"[{job_id}] DECODE {decode_s*1000:.1f} ms")

    # ── Stage 2: Depth inference ──────────────────────────────────────────────
    _record(job, "DEPTH", "start")
    t1 = time.perf_counter()
    try:
        result = depth_pipe(pil_image)
        # Use predicted_depth tensor (metric metres).
        # result["depth"] is a display-normalised PIL image — do NOT use it for geometry.
        if "predicted_depth" in result:
            pd = result["predicted_depth"]
            if hasattr(pd, "squeeze"):
                pd = pd.squeeze()
            depth_array = pd.detach().cpu().numpy().astype(np.float32)
        else:
            logger.warning(f"[{job_id}] predicted_depth missing — falling back to depth PIL image")
            raw = result["depth"]
            depth_array = np.array(raw, dtype=np.float32) if isinstance(raw, Image.Image) \
                else np.asarray(raw, dtype=np.float32)
    except Exception as exc:
        _record(job, "DEPTH", "error",
                error=f"{exc}\n{traceback.format_exc()}")
        return
    depth_s = time.perf_counter() - t1

    # Resize depth map to image dimensions if needed
    if depth_array.shape != (H, W):
        depth_pil = Image.fromarray(depth_array)
        depth_array = np.array(depth_pil.resize((W, H), Image.BILINEAR), dtype=np.float32)

    valid_mask = np.isfinite(depth_array) & (depth_array > 0)
    total_valid = int(np.sum(valid_mask))
    d_min = float(depth_array[valid_mask].min()) if total_valid else 0.0
    d_max = float(depth_array[valid_mask].max()) if total_valid else 0.0
    d_mean = float(depth_array[valid_mask].mean()) if total_valid else 0.0

    _record(job, "DEPTH", "done",
            msg=f"{depth_s*1000:.0f} ms  |  range [{d_min:.2f}–{d_max:.2f} m]  mean {d_mean:.2f} m")
    logger.info(f"[{job_id}] DEPTH {depth_s*1000:.0f} ms  [{d_min:.2f}–{d_max:.2f} m]")

    # ── Stage 3: Back-project to 3-D point cloud ──────────────────────────────
    _record(job, "BACKPROJ", "start")
    t2 = time.perf_counter()
    try:
        uu = np.arange(W, dtype=np.float32)
        vv = np.arange(H, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(uu, vv)

        Z = depth_array
        range_mask = valid_mask & (Z <= req.max_depth)
        Zf = Z[range_mask]
        Xf = (grid_u[range_mask] - req.cx) * Zf / req.fx
        Yf = (grid_v[range_mask] - req.cy) * Zf / req.fy
        total_points = int(Xf.size)
    except Exception as exc:
        _record(job, "BACKPROJ", "error",
                error=f"{exc}\n{traceback.format_exc()}")
        return
    bp_s = time.perf_counter() - t2
    _record(job, "BACKPROJ", "done",
            msg=f"{bp_s*1000:.1f} ms  |  {total_points:,} pts within {req.max_depth} m")
    logger.info(f"[{job_id}] BACKPROJ {bp_s*1000:.1f} ms  {total_points:,} pts")

    if total_points == 0:
        job["result"] = {
            "voxels": [],
            "total_points": 0,
            "occupied_voxels": 0,
            "depth_stats": {"min_m": d_min, "max_m": d_max, "mean_m": d_mean,
                            "valid_pixels": total_valid, "total_pixels": H * W},
            "depth_map": [],
            "grid_h": 0,
            "grid_w": 0,
            "grid_fx": req.fx,
            "grid_fy": req.fy,
            "grid_cx": req.cx,
            "grid_cy": req.cy,
        }
        job["status"] = "done"
        _record(job, "SERIALIZE", "done", msg="0 voxels (no valid points in range)")
        return

    # ── Stage 4: Voxelization ─────────────────────────────────────────────────
    _record(job, "VOXELIZE", "start")
    t3 = time.perf_counter()
    try:
        vs = req.voxel_size
        xi = np.floor(Xf / vs).astype(np.int32)
        yi = np.floor(Yf / vs).astype(np.int32)
        zi = np.floor(Zf / vs).astype(np.int32)
        voxel_coords = np.column_stack([xi, yi, zi])
        unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)
        occupied_mask_occ = counts >= req.min_points
        occ_voxels = unique_voxels[occupied_mask_occ]
        occ_counts = counts[occupied_mask_occ]
    except Exception as exc:
        _record(job, "VOXELIZE", "error",
                error=f"{exc}\n{traceback.format_exc()}")
        return
    vox_s = time.perf_counter() - t3
    _record(job, "VOXELIZE", "done",
            msg=f"{vox_s*1000:.1f} ms  |  {len(occ_voxels):,} occupied voxels  (voxel={vs} m, min_pts={req.min_points})")
    logger.info(f"[{job_id}] VOXELIZE {vox_s*1000:.1f} ms  {len(occ_voxels):,} voxels")

    # ── Stage 5: Serialize result ─────────────────────────────────────────────
    _record(job, "SERIALIZE", "start")
    t4 = time.perf_counter()
    try:
        vs = req.voxel_size
        voxels_out = [
            {
                "x": float((vx + 0.5) * vs),
                "y": float((vy + 0.5) * vs),
                "z": float((vz + 0.5) * vs),
                "depth": float((vz + 0.5) * vs),  # Z = depth for coloring
                "count": int(cnt),
            }
            for (vx, vy, vz), cnt in zip(occ_voxels, occ_counts)
        ]

        # ── Subsample depth map for mesh reconstruction ───────────────────────
        # Downsample to at most 160×120 so JSON payload stays reasonable.
        # Depth values (metres) don't change with resolution; only pixel coords scale.
        MESH_TARGET_W, MESH_TARGET_H = 160, 120
        scale = min(1.0, MESH_TARGET_W / W, MESH_TARGET_H / H)
        mesh_w = max(1, round(W * scale))
        mesh_h = max(1, round(H * scale))
        if scale < 1.0:
            depth_small = np.array(
                Image.fromarray(depth_array).resize((mesh_w, mesh_h), Image.BILINEAR),
                dtype=np.float32,
            )
        else:
            depth_small = depth_array
            mesh_w, mesh_h = W, H

        # Clamp invalid/out-of-range values to 0 so frontend can skip them
        depth_small = np.where(
            np.isfinite(depth_small) & (depth_small > 0) & (depth_small <= req.max_depth),
            depth_small,
            0.0,
        )

        # Round to 2 decimal places (1 cm precision) to reduce JSON size
        depth_map_flat = depth_small.ravel().round(2).tolist()

        job["result"] = {
            "voxels": voxels_out,
            "total_points": total_points,
            "occupied_voxels": len(voxels_out),
            "depth_stats": {
                "min_m": round(d_min, 3),
                "max_m": round(d_max, 3),
                "mean_m": round(d_mean, 3),
                "valid_pixels": total_valid,
                "total_pixels": H * W,
            },
            # Mesh reconstruction data — flat 1D array, reshape to (grid_h, grid_w)
            "depth_map": depth_map_flat,
            "grid_h": mesh_h,
            "grid_w": mesh_w,
            # Scaled camera intrinsics matching the subsampled grid
            "grid_fx": round(req.fx * scale, 4),
            "grid_fy": round(req.fy * scale, 4),
            "grid_cx": round(req.cx * scale, 4),
            "grid_cy": round(req.cy * scale, 4),
        }
    except Exception as exc:
        _record(job, "SERIALIZE", "error",
                error=f"{exc}\n{traceback.format_exc()}")
        return
    ser_s = time.perf_counter() - t4
    total_s = (time.perf_counter() - t0)
    _record(job, "SERIALIZE", "done",
            msg=f"{ser_s*1000:.1f} ms  |  {len(voxels_out)} voxels  mesh={mesh_w}×{mesh_h}  |  total {total_s:.2f} s")
    logger.info(f"[{job_id}] SERIALIZE {ser_s*1000:.1f} ms  total={total_s:.2f} s")
    job["status"] = "done"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/depth/health")
def health():
    """Liveness check — confirms server + model are alive independently of any job."""
    return {
        "status": "ok" if depth_pipe is not None else "model_not_loaded",
        "model": model_name,
        "error": model_load_error,
        "cuda": torch.cuda.is_available(),
    }


@app.get("/depth/model_info")
def model_info():
    mem_info: dict = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_info["gpu_total_gb"] = round(props.total_memory / 1e9, 2)
        mem_info["gpu_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        mem_info["gpu_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
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
        "active_jobs": len(jobs),
    }


@app.post("/depth/scan")
async def start_scan(req: ScanRequest):
    """Start the depth pipeline in a background thread. Returns job_id immediately."""
    if depth_pipe is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Depth model not loaded. Check startup logs.",
        )

    job_id = uuid.uuid4().hex[:8]
    try:
        job = _add_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    logger.info(f"[{job_id}] scan started")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, req)

    return {"job_id": job_id}


@app.get("/depth/status/{job_id}")
def get_status(job_id: str):
    """Return the current state of a scan job — safe to call while pipeline is running."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    job = jobs[job_id]
    # Return a snapshot; never block on pipeline completion
    return {
        "id": job["id"],
        "status": job["status"],
        "current_stage": job["current_stage"],
        "stages": list(job["stages"]),   # copy to avoid race on list resize
        "result": job["result"],
        "elapsed_s": round(time.time() - job["created_at"], 1),
    }


# Keep /depth/process for backward compatibility — delegates to the new job system
# but blocks until done (use only for debugging, not production polling)
@app.post("/depth/process")
async def process_compat(req: ScanRequest):
    """Legacy blocking endpoint — polls internally until done. Use /scan + /status instead."""
    if depth_pipe is None:
        raise HTTPException(status_code=503, detail=model_load_error or "Model not loaded.")
    job_id = uuid.uuid4().hex[:8]
    job = _add_job(job_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _run_pipeline, job_id, req)
    if job["status"] == "error":
        last_err = next((s for s in reversed(job["stages"]) if s["state"] == "error"), None)
        raise HTTPException(status_code=500, detail=last_err["error"] if last_err else "Pipeline error")
    return job["result"]
