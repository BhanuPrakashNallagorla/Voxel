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

interface ScanResult {
  total_points: number;
  occupied_voxels: number;
  depth_stats: { min_m: number; max_m: number; mean_m: number; valid_pixels: number; total_pixels: number };
  depth_map?: number[];
  seg_map?: number[];
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

// ─── Sub-components ───────────────────────────────────────────────────────────

const ControlSlider = ({
  label, value, min, max, step, unit, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (val: number) => void;
}) => (
  <div className="flex flex-col gap-1.5 p-3 bg-muted rounded-lg border border-border/50 shadow-[inset_0_1px_3px_rgba(0,0,0,0.3)]">
    <div className="flex justify-between items-end">
      <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest">{label}</label>
      <span className="text-primary font-mono text-sm font-semibold">
        {value.toFixed(step < 1 ? 2 : 0)}{unit}
      </span>
    </div>
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full accent-primary h-1.5 bg-black rounded-lg appearance-none cursor-pointer outline-none focus:ring-1 focus:ring-primary [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(0,255,128,0.8)]"
    />
  </div>
);

const SectionLabel = ({ title }: { title: string }) => (
  <div className="text-muted-foreground/70 uppercase text-[9px] font-bold tracking-[0.2em] flex items-center gap-2 mt-1">
    <div className="h-px bg-border/60 flex-1" />
    {title}
    <div className="h-px bg-border/60 flex-1" />
  </div>
);

// ─── Main component ───────────────────────────────────────────────────────────

function Home() {
  // DOM refs
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const videoRef         = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const logContainerRef  = useRef<HTMLDivElement>(null);

  // Three.js refs
  const sceneRef       = useRef<THREE.Scene | null>(null);
  const cameraRef      = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef    = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef    = useRef<OrbitControls | null>(null);
  const meshSurfaceRef = useRef<THREE.Mesh | null>(null);
  const frameIdRef     = useRef<number>(0);

  // ── Control state + refs ──────────────────────────────────────────────────
  const [maxDepth, setMaxDepth]                     = useState(5.0);
  const maxDepthRef                                 = useRef(5.0);
  const [focalLength, setFocalLength]               = useState(525);
  const focalLengthRef                              = useRef(525);
  const [depthJumpThreshold, setDepthJumpThreshold] = useState(0.3);
  const depthJumpThresholdRef                       = useRef(0.3);

  const handleMaxDepthChange = (v: number) => {
    maxDepthRef.current = v; setMaxDepth(v);
  };
  const handleFocalLengthChange = (v: number) => {
    focalLengthRef.current = v; setFocalLength(v);
  };
  const handleDepthJumpThresholdChange = (v: number) => {
    depthJumpThresholdRef.current = v; setDepthJumpThreshold(v);
  };

  // ── Running / job state ──────────────────────────────────────────────────
  const [isRunning, setIsRunning]       = useState(false);
  const isRunningRef                    = useRef(false);
  const loopActiveRef                   = useRef(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const elapsedStartRef                 = useRef<number | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<string | null>(null);

  // ── UI state ─────────────────────────────────────────────────────────────
  const [webGLError, setWebGLError] = useState<string | null>(null);
  const [modelInfo, setModelInfo]   = useState<ModelInfo | null>(null);
  const [logs, setLogs]             = useState<LogEntry[]>([]);

  // ── Log helper ───────────────────────────────────────────────────────────
  const appendLog = useCallback((stage: string, message: string) => {
    const t = new Date().toISOString().substring(11, 23);
    setLogs(prev => [...prev.slice(-99), { time: t, stage, message }]);
  }, []);

  // ── Elapsed timer ────────────────────────────────────────────────────────
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

  // ── Three.js init ─────────────────────────────────────────────────────────
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
    renderer.setClearColor(0x050508);
    rendererRef.current = renderer;

    // OrbitControls — smooth, controlled movement
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.08;
    controls.rotateSpeed    = 0.5;
    controls.zoomSpeed      = 0.7;
    controls.panSpeed       = 0.6;
    controlsRef.current = controls;

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x8899ff, 0.3);
    fillLight.position.set(-5, -2, -5);
    scene.add(fillLight);

    // Fog — subtle depth perception
    scene.fog = new THREE.Fog(0x050508, 8, 25);

    // Grid / axes — very dim so mesh stands out
    const gridHelper = new THREE.GridHelper(20, 20, 0x061208, 0x040c06);
    (gridHelper.material as THREE.Material).opacity = 0.4;
    (gridHelper.material as THREE.Material).transparent = true;
    scene.add(gridHelper);
    const axesHelper = new THREE.AxesHelper(0.5);
    (axesHelper.material as THREE.Material).opacity = 0.25;
    (axesHelper.material as THREE.Material).transparent = true;
    scene.add(axesHelper);

    // Mesh surface — vertex colors driven by walkability seg map
    const meshSurface = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        color: 0xffffff,
        side: THREE.DoubleSide,
        roughness: 0.72,
        metalness: 0.04,
        flatShading: false,
      })
    );
    meshSurface.frustumCulled = false;
    scene.add(meshSurface);
    meshSurfaceRef.current = meshSurface;

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

  // ── Camera init ───────────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
    }).then(stream => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    }).catch(err => appendLog("ERROR", `Camera: ${err.message}`));
  }, [appendLog]);

  // ── Health check on mount ─────────────────────────────────────────────────
  useEffect(() => {
    fetch("/depth/health")
      .then(r => r.json())
      .then(data => setModelInfo(data))
      .catch(err => setModelInfo({ status: "error", error: err.message, model: null, cuda: false }));
  }, []);

  // ── Auto-scroll terminal ──────────────────────────────────────────────────
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // ── Auto-fit camera to mesh bounding box ─────────────────────────────────
  const autoFitMesh = useCallback((geo: THREE.BufferGeometry) => {
    const camera   = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    geo.computeBoundingBox();
    const box = geo.boundingBox;
    if (!box) return;

    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const diagRadius = size.length() * 0.5;

    controls.target.copy(center);

    const vFovRad = camera.fov * (Math.PI / 180);
    const fitDist = (diagRadius / Math.tan(vFovRad / 2)) * 1.35;
    camera.position.set(
      center.x + fitDist * 0.3,
      center.y + fitDist * 0.35,
      center.z - fitDist * 0.9,
    );
    camera.lookAt(center);

    controls.minDistance = Math.max(0.4, diagRadius * 0.15);
    controls.maxDistance = diagRadius * 5;
    controls.update();
  }, []);

  // ── Render mesh with walkability vertex colors ────────────────────────────
  const renderMesh = useCallback((result: ScanResult) => {
    const surface = meshSurfaceRef.current;
    if (!surface) return;

    const { depth_map, seg_map, grid_h, grid_w, grid_fx, grid_fy, grid_cx, grid_cy } = result;
    if (!depth_map || !grid_h || !grid_w || depth_map.length === 0) return;

    const H  = grid_h;
    const W  = grid_w;
    const fx = grid_fx ?? focalLengthRef.current;
    const fy = grid_fy ?? focalLengthRef.current;
    const cx = grid_cx ?? W / 2;
    const cy = grid_cy ?? H / 2;
    const threshold = depthJumpThresholdRef.current;
    const hasSeg    = !!(seg_map && seg_map.length === H * W);

    const vX    = new Float32Array(H * W);
    const vY    = new Float32Array(H * W);
    const vZ    = new Float32Array(H * W);
    const valid = new Uint8Array(H * W);

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const d   = depth_map[r * W + c];
        const idx = r * W + c;
        if (d > 0) {
          vX[idx] = -((c - cx) * d / fx);
          vY[idx] = -((r - cy) * d / fy);
          vZ[idx] = d;
          valid[idx] = 1;
        }
      }
    }

    // Per-pixel RGB: green = walkable (road/sidewalk/terrain), red = non-walkable
    const WALKABLE = new Set([0, 1, 9]);
    const pixRGB   = new Float32Array(H * W * 3);
    if (hasSeg) {
      for (let i = 0; i < H * W; i++) {
        if (WALKABLE.has(seg_map![i])) {
          pixRGB[i * 3] = 0.13; pixRGB[i * 3 + 1] = 0.72; pixRGB[i * 3 + 2] = 0.27;
        } else {
          pixRGB[i * 3] = 0.82; pixRGB[i * 3 + 1] = 0.17; pixRGB[i * 3 + 2] = 0.17;
        }
      }
    } else {
      pixRGB.fill(0.65);
    }

    const positions: number[] = [];
    const colors: number[]    = [];

    for (let r = 0; r < H - 1; r++) {
      for (let c = 0; c < W - 1; c++) {
        const i00 = r * W + c;
        const i01 = r * W + c + 1;
        const i10 = (r + 1) * W + c;
        const i11 = (r + 1) * W + c + 1;

        if (!valid[i00] || !valid[i01] || !valid[i10] || !valid[i11]) continue;

        const dMax = Math.max(vZ[i00], vZ[i01], vZ[i10], vZ[i11]);
        const dMin = Math.min(vZ[i00], vZ[i01], vZ[i10], vZ[i11]);
        if (dMax - dMin > threshold) continue;

        positions.push(
          vX[i00], vY[i00], vZ[i00], vX[i10], vY[i10], vZ[i10], vX[i01], vY[i01], vZ[i01],
          vX[i01], vY[i01], vZ[i01], vX[i10], vY[i10], vZ[i10], vX[i11], vY[i11], vZ[i11],
        );
        colors.push(
          pixRGB[i00*3], pixRGB[i00*3+1], pixRGB[i00*3+2],
          pixRGB[i10*3], pixRGB[i10*3+1], pixRGB[i10*3+2],
          pixRGB[i01*3], pixRGB[i01*3+1], pixRGB[i01*3+2],
          pixRGB[i01*3], pixRGB[i01*3+1], pixRGB[i01*3+2],
          pixRGB[i10*3], pixRGB[i10*3+1], pixRGB[i10*3+2],
          pixRGB[i11*3], pixRGB[i11*3+1], pixRGB[i11*3+2],
        );
      }
    }

    surface.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.computeVertexNormals();
    surface.geometry = geo;

    autoFitMesh(geo);

    const triCount = positions.length / 9;
    const ds = result.depth_stats;
    appendLog("MESH",
      `${triCount.toLocaleString()} tris  ·  depth [${ds.min_m.toFixed(2)}–${ds.max_m.toFixed(2)} m]  ·  seg ${hasSeg ? '✓' : '—'}`
    );
  }, [appendLog, autoFitMesh]);

  // ── Capture frame from camera ─────────────────────────────────────────────
  const captureFrame = (): string | null => {
    const video  = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx || video.videoWidth === 0) return null;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    appendLog("CAPTURE", `${canvas.width}×${canvas.height} px`);
    return canvas.toDataURL('image/jpeg', 0.85).replace(/^data:image\/jpeg;base64,/, "");
  };

  // ── POST /depth/scan ──────────────────────────────────────────────────────
  const startScan = async (): Promise<string | null> => {
    const b64 = captureFrame();
    if (!b64) {
      appendLog("ERROR", "camera not ready — no frame captured");
      return null;
    }
    const video = videoRef.current!;
    const cx = (captureCanvasRef.current?.width  ?? video.videoWidth)  / 2;
    const cy = (captureCanvasRef.current?.height ?? video.videoHeight) / 2;
    try {
      const res = await fetch("/depth/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image:      b64,
          fx:         focalLengthRef.current,
          fy:         focalLengthRef.current,
          cx, cy,
          max_depth:  maxDepthRef.current,
          voxel_size: 0.25,   // kept for API compatibility — not used in UI
          min_points: 2,
        }),
      });
      if (!res.ok) throw new Error(`POST /depth/scan → HTTP ${res.status}: ${await res.text()}`);
      const { job_id } = await res.json() as { job_id: string };
      appendLog("START", `job_id=${job_id}`);
      setCurrentJobId(job_id);
      return job_id;
    } catch (err: unknown) {
      appendLog("ERROR", `scan start failed: ${(err as Error).message}`);
      return null;
    }
  };

  // ── Poll GET /depth/status/{job_id} until done ────────────────────────────
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
        appendLog("ERROR", `poll: ${(err as Error).message}`);
        continue;
      }

      const stages = status.stages ?? [];
      for (let i = lastStageCount; i < stages.length; i++) {
        const s = stages[i];
        if (s.state === 'error') {
          appendLog("ERROR", `${s.stage}: ${s.error || s.msg}`);
        } else if (s.state === 'start') {
          appendLog(s.stage, `▶ started`);
          setCurrentStage(s.stage);
        } else {
          appendLog(s.stage, s.msg);
        }
      }
      lastStageCount = stages.length;

      if (status.status === 'done' && status.result) {
        renderMesh(status.result);
        appendLog("DONE", `${status.elapsed_s}s  ·  ${status.result.total_points.toLocaleString()} pts`);
        setCurrentJobId(null);
        setCurrentStage(null);
        return;
      }
      if (status.status === 'error') {
        appendLog("ERROR", `job failed at ${status.current_stage}`);
        setCurrentJobId(null);
        setCurrentStage(null);
        return;
      }
    }
  };

  // ── Main scan loop ────────────────────────────────────────────────────────
  const loop = async () => {
    if (loopActiveRef.current) return;
    loopActiveRef.current = true;
    try {
      while (isRunningRef.current) {
        elapsedStartRef.current = Date.now();
        setElapsedSeconds(0);
        const jobId = await startScan();
        if (!jobId) { await new Promise(r => setTimeout(r, 2000)); continue; }
        await pollUntilDone(jobId);
        if (isRunningRef.current) await new Promise(r => setTimeout(r, 500));
      }
    } finally {
      loopActiveRef.current = false;
      if (isRunningRef.current) loop();
      else setCurrentJobId(null);
    }
  };

  const toggleRunning = () => {
    const next = !isRunningRef.current;
    isRunningRef.current = next;
    setIsRunning(next);
    if (next) loop();
  };

  // ── Stage colour coding ───────────────────────────────────────────────────
  const getStageColor = (stage: string) => {
    if (stage === 'ERROR')   return 'text-destructive';
    if (stage === 'DEPTH')   return 'text-yellow-400';
    if (stage === 'SEGMENT') return 'text-emerald-400';
    if (stage === 'BACKPROJ') return 'text-primary/70';
    if (stage === 'VOXELIZE') return 'text-primary/70';
    if (stage === 'SERIALIZE') return 'text-primary/50';
    if (stage === 'MESH')    return 'text-cyan-400';
    if (stage === 'DONE')    return 'text-green-400';
    if (stage === 'CAPTURE') return 'text-primary/50';
    if (stage === 'START')   return 'text-primary/50';
    return 'text-primary/80';
  };

  const isBusy = isRunning && currentJobId !== null;

  // ─── JSX ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[100dvh] w-full bg-background text-foreground font-sans overflow-hidden">

      {/* ── LEFT PANEL ── */}
      <div className="w-[38%] h-full flex flex-col border-r border-border bg-card/50 shadow-[4px_0_24px_rgba(0,0,0,0.5)] z-10">

        {/* Header */}
        <div className="px-5 pt-5 pb-4 border-b border-border bg-black/40 backdrop-blur-sm">
          <h1 className="text-2xl font-bold uppercase tracking-widest text-primary flex items-center gap-3 drop-shadow-[0_0_8px_rgba(0,255,128,0.5)]">
            <span className="w-2.5 h-2.5 bg-primary rounded-full animate-pulse shadow-[0_0_12px_rgba(0,255,128,0.8)]" />
            Depth Scan
          </h1>
          <p className="text-muted-foreground/60 font-mono text-[10px] uppercase tracking-widest mt-1">
            Walkable surface reconstruction
          </p>
        </div>

        {/* Sliders — grouped by function */}
        <div className="px-5 py-4 border-b border-border bg-black/30 space-y-4">
          <SectionLabel title="Capture" />
          <div className="grid grid-cols-2 gap-3">
            <ControlSlider label="Max Depth"    value={maxDepth}    min={1.0}  max={15.0} step={0.5}  unit="m"  onChange={handleMaxDepthChange} />
            <ControlSlider label="Focal Length" value={focalLength} min={200}  max={1000} step={25}   unit="px" onChange={handleFocalLengthChange} />
          </div>

          <SectionLabel title="Mesh" />
          <ControlSlider label="Edge Gap Threshold" value={depthJumpThreshold} min={0.05} max={2.0} step={0.05} unit="m" onChange={handleDepthJumpThresholdChange} />
        </div>

        {/* Diagnostics + camera feed */}
        <div className="px-5 py-4 border-b border-border flex justify-between items-start gap-4 bg-gradient-to-b from-black/20 to-transparent">
          <div className="flex-1 min-w-0">
            <SectionLabel title="Diagnostics" />
            <div className="mt-2 space-y-2">
              {modelInfo ? (
                <>
                  <div className="font-mono text-primary text-xs bg-primary/10 px-2.5 py-1.5 rounded border border-primary/20 truncate">
                    {modelInfo.model ?? 'Unknown Engine'}
                  </div>
                  <div className="flex gap-3 font-mono text-[11px] text-muted-foreground">
                    <span>CUDA: <span className={modelInfo.cuda ? 'text-primary' : 'text-yellow-500'}>{modelInfo.cuda ? 'ONLINE' : 'OFFLINE'}</span></span>
                    <span>STAT: <span className={modelInfo.status === 'ok' ? 'text-primary' : 'text-destructive'}>{modelInfo.status.toUpperCase()}</span></span>
                  </div>
                  {modelInfo.error && (
                    <span className="text-destructive font-mono text-xs bg-destructive/10 border border-destructive/30 px-2 py-1 rounded block">{modelInfo.error}</span>
                  )}
                </>
              ) : (
                <div className="text-muted-foreground font-mono text-xs animate-pulse">Polling runtime…</div>
              )}
            </div>
          </div>

          <div className="relative w-36 aspect-video shrink-0 border border-border bg-black rounded-lg overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.8)] group">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
            <div className="absolute inset-0 shadow-[inset_0_0_20px_rgba(0,255,128,0.15)] pointer-events-none" />
            <div className="absolute bottom-1 left-1 bg-black/90 text-[9px] text-primary px-1.5 py-0.5 rounded-sm font-mono uppercase border border-primary/30">LIVE</div>
            <div className="absolute top-1 right-1 w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_5px_rgba(255,0,0,0.8)]" />
          </div>
        </div>

        {/* Scan button */}
        <div className="px-5 py-4 border-b border-border bg-black/20 space-y-2">
          <button
            onClick={toggleRunning}
            className={`w-full py-3.5 font-bold uppercase tracking-[0.2em] text-sm rounded-lg transition-all duration-300 border-2 ${
              isBusy
                ? 'bg-yellow-500/5 text-yellow-400/60 border-yellow-400/20 cursor-not-allowed opacity-60'
                : isRunning
                ? 'bg-destructive/20 text-destructive border-destructive shadow-[0_0_25px_rgba(255,0,0,0.3)] hover:bg-destructive/30'
                : 'bg-primary/10 text-primary border-primary shadow-[0_0_20px_rgba(0,255,128,0.2)] hover:bg-primary/20 hover:shadow-[0_0_30px_rgba(0,255,128,0.5)]'
            }`}
          >
            {isBusy
              ? `⟳  Processing…  ${currentStage ?? ''}`
              : isRunning
              ? 'Halt Scan'
              : 'Initialize Scan'}
          </button>

          <div className={`flex justify-between items-center font-mono text-[11px] transition-opacity duration-300 ${isRunning ? 'opacity-100' : 'opacity-0'}`}>
            <span className="text-primary/50">{currentJobId ? `JOB: ${currentJobId}` : 'QUEUING…'}</span>
            <span className={`tabular-nums ${elapsedSeconds > 60 ? 'text-yellow-400' : 'text-primary/70'}`}>
              {fmtElapsed(elapsedSeconds)}
            </span>
          </div>
        </div>

        {/* Terminal log */}
        <div className="flex-1 flex flex-col overflow-hidden bg-[#020204]">
          <div className="px-5 pt-4 pb-2">
            <SectionLabel title="Terminal Output" />
          </div>
          <div
            ref={logContainerRef}
            className="flex-1 overflow-y-auto px-4 pb-4 font-mono text-[11.5px] space-y-0.5"
          >
            {logs.length === 0 && (
              <div className="text-muted-foreground/40 italic px-1 pt-1">Awaiting telemetry…</div>
            )}
            {logs.map((log, i) => (
              <div key={i} className="flex gap-2 hover:bg-white/[0.03] px-1 py-[3px] rounded transition-colors group">
                <span className="text-muted-foreground/40 shrink-0 text-[10px] pt-px tabular-nums group-hover:text-muted-foreground/60">
                  [{log.time}]
                </span>
                <span className={`w-[4.5rem] shrink-0 text-right font-bold text-[10px] pt-px ${getStageColor(log.stage)}`}>
                  {log.stage}
                </span>
                <span className="text-gray-300/90 break-all leading-relaxed">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL — Three.js canvas ── */}
      <div className="flex-1 h-full relative bg-[#030305]">

        {/* WebGL error */}
        {webGLError && (
          <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/90 p-8">
            <div className="font-mono border border-yellow-400/30 bg-yellow-400/5 rounded-lg p-6 max-w-md text-center space-y-3">
              <div className="text-yellow-400 font-bold text-sm tracking-widest uppercase">WebGL Context Unavailable</div>
              <div className="text-gray-400 text-xs leading-relaxed">{webGLError}</div>
              <div className="text-primary/60 text-xs pt-2 border-t border-primary/10">
                The backend pipeline will still work — only the 3D render requires a GPU-capable browser context.
              </div>
            </div>
          </div>
        )}

        {/* Canvas */}
        <div className="absolute inset-0">
          <canvas ref={canvasRef} className="block w-full h-full outline-none" />
        </div>

        {/* Processing overlay — shown while a job is in flight */}
        {isBusy && (
          <div className="absolute inset-0 z-10 pointer-events-none flex items-center justify-center">
            <div className="bg-black/70 backdrop-blur-sm border border-primary/20 rounded-xl px-8 py-5 flex flex-col items-center gap-3 shadow-[0_0_40px_rgba(0,0,0,0.8)]">
              <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <div className="font-mono text-primary text-sm tracking-widest uppercase">
                {currentStage ?? 'Processing'}
              </div>
              <div className="font-mono text-primary/50 text-xs tabular-nums">
                {fmtElapsed(elapsedSeconds)} elapsed  ·  {currentJobId ?? '…'}
              </div>
            </div>
          </div>
        )}

        {/* HUD — top-right */}
        <div className="absolute top-4 right-4 pointer-events-none z-10">
          <div className="text-primary/40 font-mono text-[10px] text-right space-y-1 bg-black/40 px-3 py-2.5 rounded border border-primary/10 backdrop-blur-sm">
            <div className="flex justify-end gap-2 items-center">
              <div className="w-1.5 h-1.5 bg-primary rounded-full" />
              SYS_RENDERER_ACTIVE
            </div>
            <div>COORD_SYS: RIGHT_HANDED</div>
            <div className="text-[9px] text-muted-foreground/50 mt-1.5 pt-1.5 border-t border-primary/10">
              drag to orbit · scroll to zoom
            </div>
          </div>
        </div>

        {/* Walkability legend — bottom-right corner */}
        <div className="absolute bottom-5 right-5 z-10 pointer-events-none">
          <div className="bg-black/70 backdrop-blur-sm border border-primary/15 rounded-xl px-5 py-3.5">
            <div className="text-[9px] text-primary/50 uppercase tracking-[0.18em] font-bold font-mono mb-2.5 text-center">
              Walkability
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2.5">
                <div className="w-3.5 h-3.5 rounded-sm bg-[#21b845] shadow-[0_0_8px_rgba(33,184,69,0.7)] shrink-0" />
                <span className="font-mono text-[11px] text-[#21b845] font-semibold">Walkable</span>
              </div>
              <div className="flex items-center gap-2.5">
                <div className="w-3.5 h-3.5 rounded-sm bg-[#d12b2b] shadow-[0_0_8px_rgba(209,43,43,0.7)] shrink-0" />
                <span className="font-mono text-[11px] text-[#d12b2b] font-semibold">Non-walkable</span>
              </div>
            </div>
            <div className="font-mono text-[8.5px] text-primary/25 mt-2.5 pt-2 border-t border-primary/10 text-center">
              road · sidewalk · terrain
            </div>
          </div>
        </div>

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
            404_NOT_FOUND
          </div>
        </Route>
      </Switch>
    </WouterRouter>
  );
}

export default App;
