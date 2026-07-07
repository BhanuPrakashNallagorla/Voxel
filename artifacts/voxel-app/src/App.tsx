import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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

const ControlSlider = ({ label, value, min, max, step, unit, onChange }: { 
  label: string; 
  value: number; 
  min: number; 
  max: number; 
  step: number; 
  unit: string; 
  onChange: (val: number) => void 
}) => {
  return (
    <div className="flex flex-col gap-1.5 p-3 bg-muted rounded-lg border border-border/50 shadow-[inset_0_1px_3px_rgba(0,0,0,0.3)]">
      <div className="flex justify-between items-end">
        <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest">{label}</label>
        <span className="text-primary font-mono text-sm">{value.toFixed(step < 1 ? 2 : 0)}{unit}</span>
      </div>
      <input 
        type="range" 
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-primary h-1.5 bg-black rounded-lg appearance-none cursor-pointer outline-none focus:ring-1 focus:ring-primary [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(0,255,128,0.8)]"
      />
    </div>
  );
};

function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Three.js Refs
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const instancedMeshRef = useRef<THREE.InstancedMesh | null>(null);
  const frameIdRef = useRef<number>(0);

  // Control State & Refs
  const [voxelSize, setVoxelSize] = useState(0.25);
  const voxelSizeRef = useRef(0.25);

  const [maxDepth, setMaxDepth] = useState(5.0);
  const maxDepthRef = useRef(5.0);

  const [minPoints, setMinPoints] = useState(2);
  const minPointsRef = useRef(2);

  const [focalLength, setFocalLength] = useState(525);
  const focalLengthRef = useRef(525);

  const [isRunning, setIsRunning] = useState(false);
  const isRunningRef = useRef(false);
  const loopActiveRef = useRef(false); // single-flight guard: prevents concurrent loops

  const [webGLError, setWebGLError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const appendLog = useCallback((stage: string, message: string) => {
    const time = new Date().toISOString().substring(11, 23);
    setLogs(prev => [...prev.slice(-49), { time, stage, message }]);
  }, []);

  // Sync state to refs for the loop
  const handleVoxelSizeChange = (v: number) => { voxelSizeRef.current = v; setVoxelSize(v); };
  const handleMaxDepthChange = (v: number) => { maxDepthRef.current = v; setMaxDepth(v); };
  const handleMinPointsChange = (v: number) => { minPointsRef.current = v; setMinPoints(v); };
  const handleFocalLengthChange = (v: number) => { focalLengthRef.current = v; setFocalLength(v); };

  // Three.js Initialization
  useEffect(() => {
    if (!canvasRef.current || rendererRef.current) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, canvasRef.current.clientWidth / canvasRef.current.clientHeight, 0.01, 100);
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
    renderer.setClearColor(0x050508); // Deep space background
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    // Cyberpunk Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x165a4c, 0x0f2a24);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);

    // Voxel Mesh
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, 50000);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(instancedMesh);
    instancedMeshRef.current = instancedMesh;

    // Handle WebGL context loss gracefully
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      cancelAnimationFrame(frameIdRef.current);
      isRunningRef.current = false;
      setIsRunning(false);
      setWebGLError('WebGL context was lost (GPU reset or tab backgrounded). Reload to restore 3D rendering.');
    };
    const handleContextRestored = () => {
      setWebGLError(null);
    };
    canvasRef.current.addEventListener('webglcontextlost', handleContextLost, false);
    canvasRef.current.addEventListener('webglcontextrestored', handleContextRestored, false);

    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    const resizeObserver = new ResizeObserver(entries => {
      if (!entries || !entries.length) return;
      const { width, height } = entries[0].contentRect;
      if (rendererRef.current && cameraRef.current) {
        rendererRef.current.setSize(width, height, false);
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
      }
    });
    
    if (canvasRef.current.parentElement) {
      resizeObserver.observe(canvasRef.current.parentElement);
    }

    return () => {
      cancelAnimationFrame(frameIdRef.current);
      resizeObserver.disconnect();
    };
  }, []);

  // Camera Capture Initialization
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ 
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } } 
    }).then(stream => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    }).catch(err => {
      appendLog("ERROR", `Camera access failed: ${err.message}`);
    });
  }, [appendLog]);

  // Initial Health Check
  useEffect(() => {
    fetch("/depth/health")
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => setModelInfo({ status: "error", error: err.message, model: null, cuda: false }));
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const processFrame = async () => {
    if (!videoRef.current || !captureCanvasRef.current) return;
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    appendLog("CAPTURE", `frame ${canvas.width}x${canvas.height}`);
    
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const base64Image = dataUrl.replace(/^data:image\/jpeg;base64,/, "");
    
    appendLog("UPLOAD", "sending frame to /depth/process...");
    
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    const payload = {
      image: base64Image,
      fx: focalLengthRef.current,
      fy: focalLengthRef.current,
      cx: cx,
      cy: cy,
      voxel_size: voxelSizeRef.current,
      max_depth: maxDepthRef.current,
      min_points: minPointsRef.current
    };
    
    try {
      const res = await fetch("/depth/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      appendLog("DECODE", `${data.timings.decode_s.toFixed(3)}s`);
      appendLog("DEPTH", `${data.timings.depth_inference_s.toFixed(3)}s`);
      appendLog("BACKPROJ", `${data.timings.backproject_s.toFixed(3)}s`);
      appendLog("VOXELIZE", `${data.timings.voxelize_s.toFixed(3)}s`);
      appendLog("SERIALIZE", `${data.timings.serialize_s.toFixed(3)}s`);
      
      if (instancedMeshRef.current) {
        const mesh = instancedMeshRef.current;
        const voxels = data.voxels;
        const MAX_VOXELS = 50000;
        mesh.count = Math.min(voxels.length, MAX_VOXELS);
        
        const matrix = new THREE.Matrix4();
        const color = new THREE.Color();
        const maxD = maxDepthRef.current;
        const vSize = voxelSizeRef.current;
        const vec3 = new THREE.Vector3(vSize, vSize, vSize);
        
        for (let i = 0; i < voxels.length; i++) {
          const v = voxels[i];
          matrix.makeTranslation(v.x, -v.y, v.z);
          matrix.scale(vec3);
          mesh.setMatrixAt(i, matrix);
          
          const hue = (v.depth / maxD) * 0.67;
          color.setHSL(hue, 1.0, 0.55);
          mesh.setColorAt(i, color);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        
        appendLog("RENDER", `${voxels.length} voxels drawn voxel_size=${vSize}m`);
      }
      
      appendLog("SUMMARY", `total=${data.timings.total_s.toFixed(3)}s | depth=[${data.depth_stats.min_m.toFixed(2)}m-${data.depth_stats.max_m.toFixed(2)}m] | pts=${data.total_points} | voxels=${data.occupied_voxels}`);
      
    } catch (err: any) {
      appendLog("ERROR", err.message);
      // Wait slightly on error to avoid tight spin loop if endpoint is down
      await new Promise(r => setTimeout(r, 1000));
    }
  };

  const loop = async () => {
    // Single-flight guard: bail immediately if a loop is already active
    if (loopActiveRef.current) return;
    loopActiveRef.current = true;
    try {
      while (isRunningRef.current) {
        await processFrame();
        // Yield to the browser paint loop before the next frame
        await new Promise<void>(r => requestAnimationFrame(() => r()));
      }
    } finally {
      loopActiveRef.current = false;
    }
  };

  const toggleRunning = () => {
    const next = !isRunningRef.current;
    isRunningRef.current = next;
    setIsRunning(next);
    if (next) {
      loop();
    }
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'DEPTH': return 'text-yellow-400';
      case 'ERROR': return 'text-destructive';
      case 'SUMMARY': return 'text-cyan-400 text-shadow-[0_0_5px_rgba(34,211,238,0.5)]';
      case 'CAPTURE': return 'text-primary/70';
      case 'RENDER': return 'text-green-400';
      default: return 'text-primary';
    }
  };

  return (
    <div className="flex h-[100dvh] w-full bg-background text-foreground font-sans overflow-hidden">
      {/* LEFT PANEL */}
      <div className="w-[40%] h-full flex flex-col border-r border-border bg-card/50 shadow-[4px_0_24px_rgba(0,0,0,0.5)] z-10 relative">
        <div className="p-5 border-b border-border bg-black/40 backdrop-blur-sm">
           <h1 className="text-3xl font-bold uppercase tracking-widest text-primary mb-6 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(0,255,128,0.5)]">
             <span className="w-3 h-3 bg-primary rounded-full animate-pulse shadow-[0_0_12px_rgba(0,255,128,0.8)]"></span>
             Grid Node
           </h1>
           <div className="grid grid-cols-2 gap-4">
             <ControlSlider label="Voxel Size" value={voxelSize} min={0.05} max={1.0} step={0.05} unit="m" onChange={handleVoxelSizeChange} />
             <ControlSlider label="Max Depth" value={maxDepth} min={1.0} max={15.0} step={0.5} unit="m" onChange={handleMaxDepthChange} />
             <ControlSlider label="Min Points" value={minPoints} min={1} max={20} step={1} unit="" onChange={handleMinPointsChange} />
             <ControlSlider label="Focal Length" value={focalLength} min={200} max={1000} step={25} unit="px" onChange={handleFocalLengthChange} />
           </div>
        </div>
        
        <div className="p-5 border-b border-border flex justify-between items-start gap-4 bg-gradient-to-b from-black/20 to-transparent">
          <div className="flex-1">
             <div className="text-muted-foreground uppercase text-[10px] font-bold tracking-widest mb-2 flex items-center gap-2">
                <div className="h-px bg-border flex-1"></div>
                Diagnostics
                <div className="h-px bg-border flex-1"></div>
             </div>
             {modelInfo ? (
               <div className="flex flex-col gap-2">
                 <div className="font-mono text-primary text-sm bg-primary/10 px-3 py-1.5 rounded-md border border-primary/20 shadow-[inset_0_0_10px_rgba(0,255,128,0.05)] w-fit">
                   {modelInfo.model || 'Unknown Engine'}
                 </div>
                 <div className="flex gap-4 font-mono text-[11px] text-muted-foreground bg-black/50 p-2 rounded border border-border/50">
                   <div>CUDA: <span className={modelInfo.cuda ? 'text-primary drop-shadow-[0_0_4px_rgba(0,255,128,0.5)]' : 'text-yellow-500'}>{modelInfo.cuda ? 'ONLINE' : 'OFFLINE'}</span></div>
                   <div>STAT: <span className={modelInfo.status === 'ok' ? 'text-primary' : 'text-destructive'}>{modelInfo.status.toUpperCase()}</span></div>
                 </div>
                 {modelInfo.error && <span className="text-destructive font-mono text-xs mt-1 bg-destructive/10 border border-destructive/30 px-2 py-1 rounded">{modelInfo.error}</span>}
               </div>
             ) : (
               <div className="text-muted-foreground font-mono text-xs animate-pulse bg-black/50 p-3 rounded border border-border/50 inline-block">Polling runtime link...</div>
             )}
          </div>
          
          <div className="relative w-40 aspect-video shrink-0 border border-border bg-black rounded-lg overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.8)] group">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
            <div className="absolute inset-0 shadow-[inset_0_0_20px_rgba(0,255,128,0.15)] pointer-events-none" />
            <div className="absolute bottom-1 left-1 bg-black/90 text-[9px] text-primary px-1.5 py-0.5 rounded-sm font-mono uppercase border border-primary/30 tracking-wider">Feed_Live</div>
            <div className="absolute top-1 right-1 flex gap-1">
              <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_5px_rgba(255,0,0,0.8)]"></div>
            </div>
          </div>
        </div>
        
        <div className="p-5 border-b border-border bg-black/20">
           <button 
             onClick={toggleRunning}
             className={`w-full py-4 font-bold uppercase tracking-[0.2em] text-sm rounded-lg transition-all duration-300 border-2 relative overflow-hidden ${
               isRunning 
                 ? 'bg-destructive/20 text-destructive border-destructive shadow-[0_0_25px_rgba(255,0,0,0.4)] hover:bg-destructive/30' 
                 : 'bg-primary/10 text-primary border-primary shadow-[0_0_20px_rgba(0,255,128,0.2)] hover:bg-primary/20 hover:shadow-[0_0_30px_rgba(0,255,128,0.5)]'
             }`}
           >
             <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full hover:animate-[shimmer_1.5s_infinite]"></div>
             {isRunning ? 'Halt Process' : 'Initialize Scan'}
           </button>
        </div>
        
        <div className="flex-1 p-5 overflow-hidden flex flex-col bg-[#020204] relative">
           <div className="text-muted-foreground uppercase text-[10px] font-bold tracking-widest mb-3 flex items-center gap-2">
              <div className="h-px bg-border flex-1"></div>
              Terminal Output
              <div className="h-px bg-border flex-1"></div>
           </div>
           
           <div ref={logContainerRef} className="flex-1 overflow-y-auto font-mono text-[11px] space-y-1.5 pb-4 custom-scrollbar relative z-10">
             {logs.length === 0 && (
               <div className="text-muted-foreground/50 italic">Awaiting telemetry...</div>
             )}
             {logs.map((log, i) => (
               <div key={i} className="flex gap-4 hover:bg-white/5 px-2 py-0.5 rounded transition-colors group">
                 <span className="text-gray-500/70 shrink-0 group-hover:text-gray-400 transition-colors">[{log.time}]</span>
                 <span className={`w-20 shrink-0 text-right font-bold ${getStageColor(log.stage)}`}>{log.stage}</span>
                 <span className="text-gray-300 break-all">{log.message}</span>
               </div>
             ))}
           </div>
           
           {/* Visual scanning line over terminal */}
           <div className="absolute inset-0 pointer-events-none border-t border-primary/10 bg-gradient-to-b from-primary/5 to-transparent opacity-50 translate-y-[-100%] animate-scan mix-blend-screen" />
        </div>
      </div>
      
      {/* RIGHT PANEL - Three.js Canvas */}
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
         
         <div className="absolute top-5 right-5 pointer-events-none z-10">
           <div className="text-primary/40 font-mono text-xs text-right space-y-1.5 bg-black/40 p-3 rounded border border-primary/10 backdrop-blur-sm">
             <div className="flex justify-end gap-2 items-center"><div className="w-1.5 h-1.5 bg-primary rounded-full"></div>SYS_RENDERER_ACTIVE</div>
             <div>COORD_SYS: RIGHT_HANDED</div>
             <div>VOXEL_MAX_CAPACITY: <span className="text-primary/70">50000</span></div>
             <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-primary/10">Use mouse/touch to orbit & zoom</div>
           </div>
         </div>

         {/* Grid decorative overlay */}
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
