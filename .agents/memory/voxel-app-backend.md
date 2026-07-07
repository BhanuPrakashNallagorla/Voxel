---
name: Voxel App Backend
description: Key decisions and gotchas for the Tesla-style voxel occupancy grid app (artifacts/voxel-app + voxel-backend/).
---

## Async job pattern (POST /depth/scan → GET /depth/status/{job_id})

The backend uses an async job + polling pattern to avoid Replit proxy timeouts (60–120s) during CPU depth inference (which takes minutes).

- `POST /depth/scan` — returns `{job_id}` in ~17ms, runs pipeline in `ThreadPoolExecutor(max_workers=1)`
- `GET /depth/status/{job_id}` — returns live progress: stages list, current_stage, status, elapsed_s, result
- Each pipeline stage (DECODE/DEPTH/BACKPROJ/VOXELIZE/SERIALIZE) records `{stage, state, ts, msg, error}` events at start AND completion
- Job store is an `OrderedDict` capped at 50 entries; eviction only removes terminal (done/error) jobs — never pending/running
- `_jobs_lock` (threading.Lock) protects eviction logic in `_add_job`
- If cap full with no terminal jobs, returns HTTP 429

**Why:** The old single blocking POST would be killed by the proxy before depth inference completed.

## HuggingFace depth-estimation pipeline output

`result["predicted_depth"]` — PyTorch tensor with actual metric values in metres (use this).
`result["depth"]` — PIL Image, display-normalized — do NOT use for 3-D geometry.

Squeeze with `.squeeze()` before `.detach().cpu().numpy()` since shape can be `(1, H, W)`.

## Managed artifact workflow working directory

`artifacts/voxel-app: Depth API` must use absolute path in artifact.toml:
`run = "bash /home/runner/workspace/voxel-backend/start.sh"`

Relative `bash voxel-backend/start.sh` fails — the managed workflow's CWD is not the workspace root.

## Frontend loop architecture

- `loopActiveRef` = single-flight guard (prevents concurrent loops)
- Elapsed timer resets per scan cycle via `elapsedStartRef.current = Date.now()` at the top of each while iteration
- Stop→start race handled in finally: if `isRunningRef.current` is true when loop exits, calls `loop()` again
- `pollUntilDone` tracks `lastStageCount` to append only new stage events each poll cycle (2s interval)
- Voxels only rendered when `status === 'done' && result !== null`

## Model performance on this environment

Model: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` (cached after first download ~30s)
RAM: 8.4 GB, CPU only (no CUDA). Inference: ~10s on a 100×100 image, scales up with resolution.
