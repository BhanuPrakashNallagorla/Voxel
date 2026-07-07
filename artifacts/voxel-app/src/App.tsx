import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ModelInfo {
  status: string;
  model: string | null;
  error: string | null;
  cuda: boolean;
}

interface LogEntry {
  time: string;
  stage: string;
  message: string;
}

interface StageEvent {
  stage: string;
  state: 'start' | 'done' | 'error';
  ts: number;
  msg: string;
  error: string;
}

interface JobStatus {
  id: string;
  status: 'pending' | 'running' | 'done' | 'error';
  current_stage: string | null;
  stages: StageEvent[];
  result: ScanResult | null;
  elapsed_s: number;
}

interface Voxel {
  x: number;
  y: number;
  z: number;
  depth: number;
  count: number;
}

interface ScanResult {
  voxels: Voxel[];
  total_points: number;
  occupied_voxels: number;
  depth_stats: { min_m: number; max_m: number; mean_m: number; valid_pixels: number; total_pixels: number };
  // Mesh reconstruction — flat 1D depth map, reshape to (grid_h × grid_w)
  depth_map?: number[];
  grid_h?: number;
  grid_w?: number;
  grid_fx?: number;
  grid_fy?: number;
  grid_cx?: number;
  grid_cy?: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtElapsed(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// ─── Sub-component ───────────────────────────────────────────────────────────

const ControlSlider = ({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (val: number) => void;
}) => (
  <div className="flex flex-col gap-1.5 p-3 bg-muted rounded-lg border border-border/50 shadow-[inset_0_1px_3px_rgba(0,0,0,0.3)]">
    <div className="flex justify-between items-end">
      <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest">{label}</label>
      <span className="text-primary font-mono text-sm">{value.toFixed(step < 1 ? 2 : 0)}{unit}</span>
    </div>
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full accent-primary h-1.5 bg-black rounded-lg appearance-none cursor-pointer outline-none focus:ring-1 focus:ring-primary [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(0,255,128,0.8)]"
    />
  </div>
);

// ─── Main component ───────────────────────────────────────────────────────────

function Home() {
  // DOM refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Three.js refs
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const instancedMeshRef = useRef<THREE.InstancedMesh | null>(null);
  const meshSurfaceRef = useRef<THREE.Mesh | null>(null);
  const pointCloudRef = useRef<THREE.Points | null>(null);
  const frameIdRef = useRef<number>(0);

  // Last scan result — lets color-mode toggle recolor without a new scan
  const lastResultRef = useRef<ScanResult | null>(null);

  // Control state + refs (refs used inside async loops to avoid stale closures)
  const [voxelSize, setVoxelSize] = useState(0.25);
  const voxelSizeRef = useRef(0.25);
  const [maxDepth, setMaxDepth] = useState(5.0);
  const maxDepthRef = useRef(5.0);
  const [minPoints, setMinPoints] = useState(2);
  const minPointsRef = useRef(2);
  const [focalLength, setFocalLength] = useState(525);
  const focalLengthRef = useRef(525);
  const [depthJumpThreshold, setDepthJumpThreshold] = useState(0.3);
  const depthJumpThresholdRef = useRef(0.3);
  const [pointSize, setPointSize] = useState(2);
  const pointSizeRef = useRef(2);

  const handleVoxelSizeChange        = (v: number) => { voxelSizeRef.current = v;          setVoxelSize(v);          };
  const handleMaxDepthChange         = (v: number) => { maxDepthRef.current = v;            setMaxDepth(v);           };
  const handleMinPointsChange        = (v: number) => { minPointsRef.current = v;           setMinPoints(v);          };
  const handleFocalLengthChange      = (v: number) => { focalLengthRef.current = v;         setFocalLength(v);        };
  const handleDepthJumpThresholdChange = (v: number) => { depthJumpThresholdRef.current = v; setDepthJumpThreshold(v); };
  const handlePointSizeChange        = (v: number) => {
    pointSizeRef.current = v;
    setPointSize(v);
    if (pointCloudRef.current) (pointCloudRef.current.material as THREE.PointsMaterial).size = v;
  };

  // Running state
  const [isRunning, setIsRunning] = useState(false);
  const isRunningRef = useRef(false);
  const loopActiveRef = useRef(false); // single-flight guard

  // Elapsed timer state (ticks while a job is in flight)
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const elapsedStartRef = useRef<number | null>(null);

  // UI state
  const [webGLError, setWebGLError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);

  // View mode: voxel grid | smooth mesh surface | dense point cloud
  const [viewMode, setViewMode] = useState<'voxel' | 'mesh' | 'points'>('voxel');
  const viewModeRef = useRef<'voxel' | 'mesh' | 'points'>('voxel');

  const handleViewModeSelect = useCallback((mode: 'voxel' | 'mesh' | 'points') => {
    viewModeRef.current = mode;
    setViewMode(mode);
    if (instancedMeshRef.current) instancedMeshRef.current.visible = mode === 'voxel';
    if (meshSurfaceRef.current)   meshSurfaceRef.current.visible   = mode === 'mesh';
    if (pointCloudRef.current)    pointCloudRef.current.visible    = mode === 'points';
  }, []);

  // Color mode toggle (voxel mode only)
  const [colorMode, setColorMode] = useState<'depth' | 'solid'>('depth');
  const colorModeRef = useRef<'depth' | 'solid'>('depth');
  const handleColorModeToggle = () => {
    const next = colorModeRef.current === 'depth' ? 'solid' : 'depth';
    colorModeRef.current = next;
    setColorMode(next);
  };

  // Depth range for legend (updated after each successful scan)
  const [depthRange, setDepthRange] = useState<{ min: number; max: number } | null>(null);

  // ── Log helper ──────────────────────────────────────────────────────────────
  const appendLog = useCallback((stage: string, message: string) => {
    const t = new Date().toISOString().substring(11, 23);
    setLogs(prev => [...prev.slice(-99), { time: t, stage, message }]);
  }, []);

  // ── Elapsed counter ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isRunning) {
      setElapsedSeconds(0);
      elapsedStartRef.current = null;
      return;
    }
    elapsedStartRef.current = Date.now();
    const id = setInterval(() => {
      const start = elapsedStartRef.current;
      if (start != null) setElapsedSeconds(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  // ── Recolor existing voxels when color mode changes ─────────────────────────
  useEffect(() => {
    if (lastResultRef.current) renderVoxels(lastResultRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorMode]);

  // ── Three.js init ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || rendererRef.current) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60, canvasRef.current.clientWidth / canvasRef.current.clientHeight, 0.01, 100
    );
    camera.position.set(2, 3, 8);
    camera.lookAt(0, 0, 2);
    cameraRef.current = camera;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setWebGLError(`WebGL unavailable in this preview (${msg}). Open the app in a real browser — it will work fine there.`);
      return;
    }
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight, false);
    renderer.shadowMap.enabled = true;
    renderer.setClearColor(0x050508);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lighting — AmbientLight + DirectionalLight for MeshStandardMaterial shading
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x8899ff, 0.3);
    fillLight.position.set(-5, -2, -5);
    scene.add(fillLight);

    // Fog — matches bg color so distant geometry fades naturally
    scene.fog = new THREE.Fog(0x050508, 10, 28);

    // Grid / axes
    scene.add(new THREE.GridHelper(20, 20, 0x0c2a1e, 0x071510));
    scene.add(new THREE.AxesHelper(0.8));

    // ── Voxel InstancedMesh ──────────────────────────────────────────────────
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial();
    const instancedMesh = new THREE.InstancedMesh(geometry, material, 50000);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    instancedMesh.count = 0;
    // Fix disappearing-on-camera-move bug: disable frustum culling so the
    // InstancedMesh is never incorrectly culled when the bounding sphere is stale.
    instancedMesh.frustumCulled = false;
    scene.add(instancedMesh);
    instancedMeshRef.current = instancedMesh;

    // ── Mesh Surface (smooth triangulated depth reconstruction) ──────────────
    const meshSurface = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        side: THREE.DoubleSide,
        roughness: 0.75,
        metalness: 0.05,
      })
    );
    meshSurface.frustumCulled = false;
    meshSurface.visible = false; // voxel mode is default
    scene.add(meshSurface);
    meshSurfaceRef.current = meshSurface;

    // ── Dense Point Cloud ────────────────────────────────────────────────────
    const pointCloud = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        size: pointSizeRef.current,
        vertexColors: true,
        sizeAttenuation: true,
      })
    );
    pointCloud.frustumCulled = false;
    pointCloud.visible = false;
    scene.add(pointCloud);
    pointCloudRef.current = pointCloud;

    // WebGL context loss handling
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      cancelAnimationFrame(frameIdRef.current);
      isRunningRef.current = false;
      setIsRunning(false);
      setWebGLError('WebGL context was lost (GPU reset or tab backgrounded). Reload to restore 3D rendering.');
    };
    const handleContextRestored = () => setWebGLError(null);
    canvasRef.current.addEventListener('webglcontextlost', handleContextLost, false);
    canvasRef.current.addEventListener('webglcontextrestored', handleContextRestored, false);

    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      controlsRef.current?.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      rendererRef.current?.setSize(width, height, false);
      if (cameraRef.current) {
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
      }
    });
    if (canvasRef.current.parentElement) ro.observe(canvasRef.current.parentElement);

    return () => {
      cancelAnimationFrame(frameIdRef.current);
      ro.disconnect();
    };
  }, []);

  // ── Camera init ─────────────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
    }).then(stream => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    }).catch(err => appendLog("ERROR", `Camera: ${err.message}`));
  }, [appendLog]);

  // ── Health check on mount ────────────────────────────────────────────────────
  useEffect(() => {
    fetch("/depth/health")
      .then(r => r.json())
      .then(data => setModelInfo(data))
      .catch(err => setModelInfo({ status: "error", error: err.message, model: null, cuda: false }));
  }, []);

  // ── Auto-scroll terminal ────────────────────────────────────────────────────
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // ── Auto-fit camera to bounding box of rendered voxels ───────────────────────
  const autoFitCamera = useCallback((voxels: Voxel[], vSize: number) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const canvas = canvasRef.current;
    if (!camera || !controls || voxels.length === 0) return;
    const box = new THREE.Box3();
    const pt = new THREE.Vector3();
    for (const v of voxels) {
      pt.set(v.x, -v.y, v.z);
      box.expandByPoint(pt);
    }
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const radius = Math.max(size.x, size.y, size.z) * 0.5 + vSize * 2;
    const aspect = canvas ? canvas.clientWidth / canvas.clientHeight : camera.aspect;
    const vFovRad = camera.fov * (Math.PI / 180);
    const hFovRad = 2 * Math.atan(Math.tan(vFovRad / 2) * aspect);
    const distV = radius / Math.tan(vFovRad / 2);
    const distH = radius / Math.tan(hFovRad / 2);
    const distance = Math.max(distV, distH) * 1.5;
    camera.position.set(
      center.x + distance * 0.45,
      center.y + distance * 0.55,
      center.z + distance,
    );
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }, []);

  // ── Render voxels into Three.js InstancedMesh ────────────────────────────────
  const renderVoxels = useCallback((result: ScanResult) => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;
    const { voxels } = result;
    const MAX = 50000;
    const count = Math.min(voxels.length, MAX);
    mesh.count = count;
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const maxD = maxDepthRef.current;
    const vSize = voxelSizeRef.current;
    const scale = new THREE.Vector3(vSize, vSize, vSize);
    const isDepth = colorModeRef.current === 'depth';
    for (let i = 0; i < count; i++) {
      const v = voxels[i];
      matrix.makeTranslation(v.x, -v.y, v.z);
      matrix.scale(scale);
      mesh.setMatrixAt(i, matrix);
      if (isDepth) {
        color.setHSL((v.depth / maxD) * 0.67, 1.0, 0.55);
      } else {
        color.setHex(0x00ff80);
      }
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    // Auto-fit camera so the cluster is framed on every new scan
    autoFitCamera(voxels.slice(0, count), vSize);

    // Persist result so color-mode toggle can recolor without a new scan
    lastResultRef.current = result;

    // Update depth range for the legend
    const ds = result.depth_stats;
    setDepthRange({ min: ds.min_m, max: ds.max_m });

    appendLog("RENDER", `${voxels.length} voxels  depth=[${ds.min_m.toFixed(2)}–${ds.max_m.toFixed(2)} m]  pts=${result.total_points}`);
  }, [appendLog, autoFitCamera]);

  // ── Build smooth mesh from depth map via grid triangulation ─────────────────
  const renderMesh = useCallback((result: ScanResult) => {
    const surface = meshSurfaceRef.current;
    if (!surface) return;

    const { depth_map, grid_h, grid_w, grid_fx, grid_fy, grid_cx, grid_cy } = result;
    if (!depth_map || !grid_h || !grid_w || depth_map.length === 0) return;

    const H = grid_h;
    const W = grid_w;
    const fx = grid_fx ?? focalLengthRef.current;
    const fy = grid_fy ?? focalLengthRef.current;
    const cx = grid_cx ?? W / 2;
    const cy = grid_cy ?? H / 2;
    const threshold = depthJumpThresholdRef.current;

    // Back-project every pixel to 3-D (same pinhole formula as voxels, Y-flipped)
    const vX = new Float32Array(H * W);
    const vY = new Float32Array(H * W);
    const vZ = new Float32Array(H * W);
    const valid = new Uint8Array(H * W);

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const d = depth_map[r * W + c];
        const idx = r * W + c;
        if (d > 0) {
          vX[idx] = (c - cx) * d / fx;
          vY[idx] = -((r - cy) * d / fy); // flip Y to match voxel coordinate system
          vZ[idx] = d;
          valid[idx] = 1;
        }
      }
    }

    // Triangulate every 2×2 block of adjacent pixels into 2 triangles.
    // Skip quads where any vertex is invalid or the depth jump exceeds threshold
    // (prevents smearing across foreground/background boundaries).
    const positions: number[] = [];

    for (let r = 0; r < H - 1; r++) {
      for (let c = 0; c < W - 1; c++) {
        const i00 = r * W + c;
        const i01 = r * W + c + 1;
        const i10 = (r + 1) * W + c;
        const i11 = (r + 1) * W + c + 1;

        if (!valid[i00] || !valid[i01] || !valid[i10] || !valid[i11]) continue;

        const d00 = vZ[i00];
        const d01 = vZ[i01];
        const d10 = vZ[i10];
        const d11 = vZ[i11];

        const dMax = Math.max(d00, d01, d10, d11);
        const dMin = Math.min(d00, d01, d10, d11);
        if (dMax - dMin > threshold) continue;

        // Triangle 1: top-left → bottom-left → top-right
        positions.push(
          vX[i00], vY[i00], vZ[i00],
          vX[i10], vY[i10], vZ[i10],
          vX[i01], vY[i01], vZ[i01],
        );
        // Triangle 2: top-right → bottom-left → bottom-right
        positions.push(
          vX[i01], vY[i01], vZ[i01],
          vX[i10], vY[i10], vZ[i10],
          vX[i11], vY[i11], vZ[i11],
        );
      }
    }

    // Dispose old geometry to free GPU memory
    surface.geometry.dispose();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.computeVertexNormals(); // smooth shading from triangle normals
    surface.geometry = geo;

    const triCount = positions.length / 9;
    appendLog("MESH", `${triCount.toLocaleString()} triangles  grid=${W}×${H}`);
  }, [appendLog]);

  // ── Build dense point cloud from depth map ───────────────────────────────────
  const renderPointCloud = useCallback((result: ScanResult) => {
    const cloud = pointCloudRef.current;
    if (!cloud) return;

    const { depth_map, grid_h, grid_w, grid_fx, grid_fy, grid_cx, grid_cy } = result;
    if (!depth_map || !grid_h || !grid_w || depth_map.length === 0) return;

    const H = grid_h;
    const W = grid_w;
    const fx = grid_fx ?? focalLengthRef.current;
    const fy = grid_fy ?? focalLengthRef.current;
    const cx = grid_cx ?? W / 2;
    const cy = grid_cy ?? H / 2;
    const maxD = maxDepthRef.current;

    // Pre-allocate at max size — will slice if fewer valid pixels
    const maxPts = H * W;
    const positions = new Float32Array(maxPts * 3);
    const colors    = new Float32Array(maxPts * 3);
    const tmpColor  = new THREE.Color();
    let n = 0;

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const d = depth_map[r * W + c];
        if (d <= 0) continue;
        const base = n * 3;
        positions[base]     = (c - cx) * d / fx;
        positions[base + 1] = -((r - cy) * d / fy); // Y-flip matches voxel coord system
        positions[base + 2] = d;
        // Red→blue depth gradient (hue 0 = near, 0.67 = far) — same as voxel depth mode
        tmpColor.setHSL((d / maxD) * 0.67, 1.0, 0.55);
        colors[base]     = tmpColor.r;
        colors[base + 1] = tmpColor.g;
        colors[base + 2] = tmpColor.b;
        n++;
      }
    }

    cloud.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions.slice(0, n * 3), 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors.slice(0, n * 3), 3));
    cloud.geometry = geo;

    appendLog("POINTS", `${n.toLocaleString()} pts  grid=${W}×${H}`);
  }, [appendLog]);

  // ── Capture one frame from camera ────────────────────────────────────────────
  const captureFrame = (): string | null => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx || video.videoWidth === 0) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    appendLog("CAPTURE", `${canvas.width}×${canvas.height} px`);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    return dataUrl.replace(/^data:image\/jpeg;base64,/, "");
  };

  // ── POST /depth/scan → job_id ────────────────────────────────────────────────
  const startScan = async (): Promise<string | null> => {
    const b64 = captureFrame();
    if (!b64) {
      appendLog("ERROR", "camera not ready — no frame captured");
      return null;
    }
    const video = videoRef.current!;
    const cx = (captureCanvasRef.current?.width ?? video.videoWidth) / 2;
    const cy = (captureCanvasRef.current?.height ?? video.videoHeight) / 2;
    const payload = {
      image: b64,
      fx: focalLengthRef.current, fy: focalLengthRef.current,
      cx, cy,
      voxel_size: voxelSizeRef.current,
      max_depth: maxDepthRef.current,
      min_points: minPointsRef.current,
    };
    try {
      const res = await fetch("/depth/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /depth/scan → HTTP ${res.status}: ${text}`);
      }
      const { job_id } = await res.json() as { job_id: string };
      appendLog("START", `job_id=${job_id}`);
      setCurrentJobId(job_id);
      return job_id;
    } catch (err: unknown) {
      appendLog("ERROR", `scan start failed: ${(err as Error).message}`);
      return null;
    }
  };

  // ── Poll GET /depth/status/{job_id} until done ───────────────────────────────
  const pollUntilDone = async (jobId: string): Promise<void> => {
    let lastStageCount = 0;

    while (isRunningRef.current) {
      await new Promise(r => setTimeout(r, 2000));
      if (!isRunningRef.current) break;

      let status: JobStatus;
      try {
        const res = await fetch(`/depth/status/${jobId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        status = await res.json() as JobStatus;
      } catch (err: unknown) {
        appendLog("ERROR", `poll /depth/status/${jobId}: ${(err as Error).message}`);
        continue;
      }

      const stages = status.stages ?? [];
      for (let i = lastStageCount; i < stages.length; i++) {
        const s = stages[i];
        if (s.state === 'error') {
          appendLog("ERROR", `${s.stage}: ${s.error || s.msg}`);
        } else if (s.state === 'start') {
          appendLog(`${s.stage}`, `▶ started`);
        } else {
          appendLog(`${s.stage}`, s.msg);
        }
      }
      lastStageCount = stages.length;

      if (status.status === 'done' && status.result) {
        // Update all three representations — toggle switches between them instantly
        renderVoxels(status.result);
        renderMesh(status.result);
        renderPointCloud(status.result);
        appendLog("SUMMARY", `total=${status.elapsed_s}s | voxels=${status.result.occupied_voxels} | pts=${status.result.total_points}`);
        setCurrentJobId(null);
        return;
      }
      if (status.status === 'error') {
        appendLog("ERROR", `job ${jobId} failed at stage ${status.current_stage}`);
        setCurrentJobId(null);
        return;
      }
    }
  };

  // ── Main scan loop ───────────────────────────────────────────────────────────
  const loop = async () => {
    if (loopActiveRef.current) return;
    loopActiveRef.current = true;
    try {
      while (isRunningRef.current) {
        elapsedStartRef.current = Date.now();
        setElapsedSeconds(0);

        const jobId = await startScan();
        if (!jobId) {
          await new Promise(r => setTimeout(r, 2000));
          continue;
        }
        await pollUntilDone(jobId);
        if (isRunningRef.current) await new Promise(r => setTimeout(r, 500));
      }
    } finally {
      loopActiveRef.current = false;
      if (isRunningRef.current) {
        loop();
      } else {
        setCurrentJobId(null);
      }
    }
  };

  const toggleRunning = () => {
    const next = !isRunningRef.current;
    isRunningRef.current = next;
    setIsRunning(next);
    if (next) loop();
  };

  // ── Stage colour coding ──────────────────────────────────────────────────────
  const getStageColor = (stage: string) => {
    if (stage === 'ERROR')   return 'text-destructive';
    if (stage === 'DEPTH')   return 'text-yellow-400';
    if (stage === 'RENDER')  return 'text-green-400';
    if (stage === 'MESH')    return 'text-cyan-400';
    if (stage === 'POINTS')  return 'text-violet-400';
    if (stage === 'SUMMARY') return 'text-cyan-400';
    if (stage === 'CAPTURE') return 'text-primary/50';
    if (stage === 'START')   return 'text-primary/60';
    return 'text-primary';
  };

  // ─── JSX ────────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[100dvh] w-full bg-background text-foreground font-sans overflow-hidden">

      {/* ── LEFT PANEL ── */}
      <div className="w-[40%] h-full flex flex-col border-r border-border bg-card/50 shadow-[4px_0_24px_rgba(0,0,0,0.5)] z-10 relative">

        {/* Header + sliders */}
        <div className="p-5 border-b border-border bg-black/40 backdrop-blur-sm">
          <h1 className="text-3xl font-bold uppercase tracking-widest text-primary mb-6 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(0,255,128,0.5)]">
            <span className="w-3 h-3 bg-primary rounded-full animate-pulse shadow-[0_0_12px_rgba(0,255,128,0.8)]" />
            Grid Node
          </h1>
          <div className="grid grid-cols-2 gap-4">
            <ControlSlider label="Voxel Size"    value={voxelSize}          min={0.05} max={1.0}  step={0.05} unit="m"  onChange={handleVoxelSizeChange} />
            <ControlSlider label="Max Depth"     value={maxDepth}           min={1.0}  max={15.0} step={0.5}  unit="m"  onChange={handleMaxDepthChange} />
            <ControlSlider label="Min Points"    value={minPoints}          min={1}    max={20}   step={1}    unit=""   onChange={handleMinPointsChange} />
            <ControlSlider label="Focal Length"  value={focalLength}        min={200}  max={1000} step={25}   unit="px" onChange={handleFocalLengthChange} />
            <ControlSlider label="Mesh Edge Gap" value={depthJumpThreshold} min={0.05} max={2.0}  step={0.05} unit="m"  onChange={handleDepthJumpThresholdChange} />
            <ControlSlider label="Point Size"   value={pointSize}          min={1}    max={10}   step={0.5}  unit="px" onChange={handlePointSizeChange} />
          </div>
        </div>

        {/* Diagnostics + camera feed */}
        <div className="p-5 border-b border-border flex justify-between items-start gap-4 bg-gradient-to-b from-black/20 to-transparent">
          <div className="flex-1">
            <div className="text-muted-foreground uppercase text-[10px] font-bold tracking-widest mb-2 flex items-center gap-2">
              <div className="h-px bg-border flex-1" /> Diagnostics <div className="h-px bg-border flex-1" />
            </div>
            {modelInfo ? (
              <div className="flex flex-col gap-2">
                <div className="font-mono text-primary text-sm bg-primary/10 px-3 py-1.5 rounded-md border border-primary/20 shadow-[inset_0_0_10px_rgba(0,255,128,0.05)] w-fit">
                  {modelInfo.model || 'Unknown Engine'}
                </div>
                <div className="flex gap-4 font-mono text-[11px] text-muted-foreground bg-black/50 p-2 rounded border border-border/50">
                  <div>CUDA: <span className={modelInfo.cuda ? 'text-primary' : 'text-yellow-500'}>{modelInfo.cuda ? 'ONLINE' : 'OFFLINE'}</span></div>
                  <div>STAT: <span className={modelInfo.status === 'ok' ? 'text-primary' : 'text-destructive'}>{modelInfo.status.toUpperCase()}</span></div>
                </div>
                {modelInfo.error && (
                  <span className="text-destructive font-mono text-xs bg-destructive/10 border border-destructive/30 px-2 py-1 rounded">{modelInfo.error}</span>
                )}
              </div>
            ) : (
              <div className="text-muted-foreground font-mono text-xs animate-pulse bg-black/50 p-3 rounded border border-border/50 inline-block">Polling runtime link...</div>
            )}
          </div>

          <div className="relative w-40 aspect-video shrink-0 border border-border bg-black rounded-lg overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.8)] group">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
            <div className="absolute inset-0 shadow-[inset_0_0_20px_rgba(0,255,128,0.15)] pointer-events-none" />
            <div className="absolute bottom-1 left-1 bg-black/90 text-[9px] text-primary px-1.5 py-0.5 rounded-sm font-mono uppercase border border-primary/30 tracking-wider">Feed_Live</div>
            <div className="absolute top-1 right-1">
              <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_5px_rgba(255,0,0,0.8)]" />
            </div>
          </div>
        </div>

        {/* Scan button + elapsed timer */}
        <div className="p-5 border-b border-border bg-black/20 space-y-2">
          <button
            onClick={toggleRunning}
            className={`w-full py-4 font-bold uppercase tracking-[0.2em] text-sm rounded-lg transition-all duration-300 border-2 relative overflow-hidden ${
              isRunning
                ? 'bg-destructive/20 text-destructive border-destructive shadow-[0_0_25px_rgba(255,0,0,0.4)] hover:bg-destructive/30'
                : 'bg-primary/10 text-primary border-primary shadow-[0_0_20px_rgba(0,255,128,0.2)] hover:bg-primary/20 hover:shadow-[0_0_30px_rgba(0,255,128,0.5)]'
            }`}
          >
            {isRunning ? 'Halt Process' : 'Initialize Scan'}
          </button>

          <div className={`flex justify-between items-center font-mono text-[11px] transition-opacity duration-300 ${isRunning ? 'opacity-100' : 'opacity-0'}`}>
            <span className="text-primary/60">
              {currentJobId ? `JOB: ${currentJobId}` : 'QUEUING...'}
            </span>
            <span className={`tabular-nums ${elapsedSeconds > 60 ? 'text-yellow-400' : 'text-primary'}`}>
              ELAPSED: {fmtElapsed(elapsedSeconds)}
            </span>
          </div>
        </div>

        {/* Terminal output */}
        <div className="flex-1 p-5 overflow-hidden flex flex-col bg-[#020204] relative">
          <div className="text-muted-foreground uppercase text-[10px] font-bold tracking-widest mb-3 flex items-center gap-2">
            <div className="h-px bg-border flex-1" /> Terminal Output <div className="h-px bg-border flex-1" />
          </div>
          <div ref={logContainerRef} className="flex-1 overflow-y-auto font-mono text-[11px] space-y-1 pb-4 custom-scrollbar relative z-10">
            {logs.length === 0 && (
              <div className="text-muted-foreground/50 italic">Awaiting telemetry...</div>
            )}
            {logs.map((log, i) => (
              <div key={i} className="flex gap-3 hover:bg-white/5 px-2 py-0.5 rounded transition-colors group">
                <span className="text-gray-500/70 shrink-0 group-hover:text-gray-400">[{log.time}]</span>
                <span className={`w-20 shrink-0 text-right font-bold ${getStageColor(log.stage)}`}>{log.stage}</span>
                <span className="text-gray-300 break-all">{log.message}</span>
              </div>
            ))}
          </div>
          <div className="absolute inset-0 pointer-events-none border-t border-primary/10 bg-gradient-to-b from-primary/5 to-transparent opacity-50 translate-y-[-100%] animate-scan mix-blend-screen" />
        </div>
      </div>

      {/* ── RIGHT PANEL — Three.js canvas ── */}
      <div className="w-[60%] h-full relative bg-[#030305]">
        {webGLError && (
          <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/90 p-8">
            <div className="font-mono border border-yellow-400/30 bg-yellow-400/5 rounded-lg p-6 max-w-md text-center space-y-3">
              <div className="text-yellow-400 font-bold text-sm tracking-widest uppercase">WebGL Context Unavailable</div>
              <div className="text-gray-400 text-xs leading-relaxed">{webGLError}</div>
              <div className="text-primary/60 text-xs pt-2 border-t border-primary/10">
                The backend pipeline (depth inference + voxelization) will still work — only the 3D render requires a GPU-capable browser context.
              </div>
            </div>
          </div>
        )}
        <div className="absolute inset-0 w-full h-full">
          <canvas ref={canvasRef} className="block w-full h-full outline-none" />
        </div>

        {/* HUD overlay */}
        <div className="absolute top-5 right-5 pointer-events-none z-10">
          <div className="text-primary/40 font-mono text-xs text-right space-y-1.5 bg-black/40 p-3 rounded border border-primary/10 backdrop-blur-sm">
            <div className="flex justify-end gap-2 items-center">
              <div className="w-1.5 h-1.5 bg-primary rounded-full" />SYS_RENDERER_ACTIVE
            </div>
            <div>COORD_SYS: RIGHT_HANDED</div>
            <div>MODE: <span className="text-primary/70">{viewMode === 'voxel' ? 'VOXEL_GRID' : viewMode === 'mesh' ? 'MESH_SURFACE' : 'POINT_CLOUD'}</span></div>
            <div>VOXEL_MAX: <span className="text-primary/70">50 000</span></div>
            <div>PATTERN: ASYNC_JOB_POLL</div>
            <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-primary/10">
              drag to orbit · scroll to zoom
            </div>
          </div>
        </div>

        {/* View mode toggle + color mode toggle — bottom-right */}
        <div className="absolute bottom-5 right-5 z-10 flex flex-col gap-2 items-end">
          {/* View mode segmented control — 3 options */}
          <div className="flex font-mono text-[10px] uppercase tracking-widest bg-black/60 border border-primary/20 rounded backdrop-blur-sm overflow-hidden">
            <button
              onClick={() => handleViewModeSelect('voxel')}
              className={`px-3 py-1.5 transition-colors ${
                viewMode === 'voxel'
                  ? 'bg-primary/20 text-primary'
                  : 'text-primary/40 hover:text-primary/70 hover:bg-primary/10'
              }`}
            >
              ⬛ Voxel
            </button>
            <div className="w-px bg-primary/20" />
            <button
              onClick={() => handleViewModeSelect('mesh')}
              className={`px-3 py-1.5 transition-colors ${
                viewMode === 'mesh'
                  ? 'bg-primary/20 text-primary'
                  : 'text-primary/40 hover:text-primary/70 hover:bg-primary/10'
              }`}
            >
              🔷 Mesh
            </button>
            <div className="w-px bg-primary/20" />
            <button
              onClick={() => handleViewModeSelect('points')}
              className={`px-3 py-1.5 transition-colors ${
                viewMode === 'points'
                  ? 'bg-primary/20 text-primary'
                  : 'text-primary/40 hover:text-primary/70 hover:bg-primary/10'
              }`}
            >
              ✦ Points
            </button>
          </div>

          {/* Color mode toggle — only relevant in voxel mode */}
          {viewMode === 'voxel' && (
            <button
              onClick={handleColorModeToggle}
              className="font-mono text-[10px] uppercase tracking-widest bg-black/60 border border-primary/20 text-primary/70 px-3 py-1.5 rounded hover:bg-primary/10 hover:text-primary transition-colors backdrop-blur-sm"
            >
              {colorMode === 'depth' ? '⬛ DEPTH GRADIENT' : '🟩 SOLID FILL'}
            </button>
          )}
        </div>

        {/* Depth legend — bottom-centre, shown after first scan */}
        {depthRange && (
          <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
            <div className="bg-black/60 backdrop-blur-sm border border-primary/15 rounded-lg px-4 py-2.5 text-center">
              <div className="text-[9px] text-primary/50 uppercase tracking-widest mb-1.5 font-bold font-mono">
                Depth Scale
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-red-400 tabular-nums">{depthRange.min.toFixed(1)}m</span>
                <div
                  className="w-28 h-2 rounded-full"
                  style={{ background: 'linear-gradient(to right, #ff2200, #ffaa00, #00ff80, #0099ff, #3300ff)' }}
                />
                <span className="font-mono text-[10px] text-blue-400 tabular-nums">{depthRange.max.toFixed(1)}m</span>
              </div>
              <div className="flex justify-between mt-0.5">
                <span className="font-mono text-[8px] text-red-400/50">NEAR</span>
                <span className="font-mono text-[8px] text-blue-400/50">FAR</span>
              </div>
            </div>
          </div>
        )}

        {/* Dot-grid overlay */}
        <div className="absolute inset-0 pointer-events-none bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMCwyNTUsMTI4LDAuMDUpIi8+PC9zdmc+')] opacity-50 mix-blend-screen" />
      </div>

      {/* Hidden capture canvas */}
      <canvas ref={captureCanvasRef} className="hidden" />
    </div>
  );
}

function App() {
  return (
    <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
      <Switch>
        <Route path="/" component={Home} />
        <Route>
          <div className="min-h-screen flex items-center justify-center bg-background text-primary font-mono">
            404_SYSTEM_NOT_FOUND
          </div>
        </Route>
      </Switch>
    </WouterRouter>
  );
}

export default App;
