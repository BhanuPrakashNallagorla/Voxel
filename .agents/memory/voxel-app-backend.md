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

## Three.js InstancedMesh color fix

**NEVER use MeshLambertMaterial({ vertexColors: true }) with InstancedMesh.**
BoxGeometry has no `color` vertex attribute; with `vertexColors:true` the shader reads (0,0,0)
and multiplies the instance color by zero → black voxels.

**Correct approach:** `new THREE.MeshBasicMaterial()` (vertexColors defaults to false).
Per-instance colors set via `mesh.setColorAt(i, color)` + `mesh.instanceColor.needsUpdate = true`.
Three.js enables `USE_INSTANCING_COLOR` shader path independently of vertexColors — no lighting needed.

## Frontend loop architecture

- `loopActiveRef` = single-flight guard (prevents concurrent loops)
- Elapsed timer resets per scan cycle via `elapsedStartRef.current = Date.now()` at the top of each while iteration
- Stop→start race handled in finally: if `isRunningRef.current` is true when loop exits, calls `loop()` again
- `pollUntilDone` tracks `lastStageCount` to append only new stage events each poll cycle (2s interval)
- Voxels only rendered when `status === 'done' && result !== null`
- `lastResultRef` stores last ScanResult so color-mode toggle can recolor without a new scan
- `useEffect([colorMode])` calls `renderVoxels(lastResultRef.current)` on toggle

## Auto-fit camera

`autoFitCamera(voxels, vSize)` computes bounding box of all rendered voxel positions,
then derives required camera distance from BOTH vertical and horizontal FOV (uses the more
constraining axis — `Math.max(distV, distH)`) multiplied by 1.5 margin.
Must read `canvas.clientWidth/Height` for aspect ratio (not `camera.aspect`) to handle
cases where camera hasn't been updated yet after resize.

## Model performance on this environment

Model: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` (cached after first download ~30s)
RAM: 8.4 GB, CPU only (no CUDA). Inference: ~10–12s on a 640×480 frame, scales with resolution.
