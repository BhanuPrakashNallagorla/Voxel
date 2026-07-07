---
name: Voxel App Backend
description: Key decisions and gotchas for the Tesla-style voxel occupancy grid app (artifacts/voxel-app + voxel-backend/).
---

## HuggingFace depth-estimation pipeline output

The transformers depth-estimation pipeline returns TWO outputs:
- `result["predicted_depth"]` — PyTorch tensor, actual metric values in metres (use this for 3-D back-projection)
- `result["depth"]`           — PIL Image, display-normalized to 0-255 (NOT metric — do NOT use for geometry)

**Why:** Using `result["depth"]` produces display-normalized depth values, making all 3-D geometry fundamentally wrong at the wrong scale. Always use `result["predicted_depth"].squeeze().detach().cpu().numpy()`.

## Managed artifact workflow working directory

The `artifacts/voxel-app: Depth API` managed workflow runs with a working directory that is NOT the workspace root — `bash voxel-backend/start.sh` fails with "No such file or directory". Must use the absolute path: `bash /home/runner/workspace/voxel-backend/start.sh` in the artifact.toml service run command.

**How to apply:** Any time a managed artifact service needs to run a script outside its own artifact directory, use absolute paths in artifact.toml.

## Port conflict between configureWorkflow and managed artifact workflow

Adding a service to artifact.toml creates a managed workflow automatically. If you also call `configureWorkflow` for the same port, the second process fails with "Address already in use". Remove the `configureWorkflow`-based workflow after confirming the managed one works.

## Model loaded successfully

`depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` loads successfully on CPU in this environment (~30s download on first run, then cached). RAM is 8.4 GB — CPU inference only, no GPU. Expect minutes per frame on CPU.

## Frontend

- `loopActiveRef` single-flight guard prevents concurrent processing loops on rapid toggle
- InstancedMesh capped at 50000 voxels to match declared capacity
- `webglcontextlost` / `webglcontextrestored` handlers stop the loop and show an error message
- WebGL unavailable in Replit's headless preview (expected) — works fine in real browsers
