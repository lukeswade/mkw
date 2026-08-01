import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Eye, RotateCcw, Box, Layers } from 'lucide-react';
import { MaterialType } from '../types';

interface ThreeCanvasProps {
  geometry: THREE.BufferGeometry;
  material: MaterialType;
  onMaterialChange: (mat: MaterialType) => void;
  widthMm: number;
  depthMm: number;
  heightMm: number;
}

export const ThreeCanvas: React.FC<ThreeCanvasProps> = ({
  geometry,
  material,
  onMaterialChange,
  widthMm,
  depthMm,
  heightMm,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [wireframe, setWireframe] = useState(false);
  const [isRotating, setIsRotating] = useState(true);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Material color definitions
  const materialStyles: Record<MaterialType, { color: number; roughness: number; metalness: number; transmission: number; opacity: number }> = {
    PLA: { color: 0x4f46e5, roughness: 0.35, metalness: 0.1, transmission: 0, opacity: 1 },       // Vivid Indigo
    PETG: { color: 0x06b6d4, roughness: 0.15, metalness: 0.05, transmission: 0.4, opacity: 0.9 },   // Translucent Cyan
    TPU: { color: 0x10b981, roughness: 0.5, metalness: 0.0, transmission: 0, opacity: 1 },        // Emerald Flex
    ABS: { color: 0xf59e0b, roughness: 0.4, metalness: 0.2, transmission: 0, opacity: 1 },        // Tough Amber
  };

  useEffect(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    scene.fog = new THREE.FogExp2(0x090d16, 0.003);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    camera.position.set(120, 110, 140);
    camera.lookAt(0, heightMm / 2, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.5);
    dirLight1.position.set(100, 150, 100);
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.width = 1024;
    dirLight1.shadow.mapSize.height = 1024;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x22d3ee, 1.5);
    dirLight2.position.set(-100, 80, -100);
    scene.add(dirLight2);

    // Print Bed Grid (220mm x 220mm)
    const gridHelper = new THREE.GridHelper(220, 22, 0x6366f1, 0x1e293b);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // Bed Outer Boundary Box
    const bedGeo = new THREE.PlaneGeometry(220, 220);
    const bedMat = new THREE.MeshBasicMaterial({ color: 0x1e1b4b, side: THREE.DoubleSide, transparent: true, opacity: 0.4 });
    const bedMesh = new THREE.Mesh(bedGeo, bedMat);
    bedMesh.rotateX(Math.PI / 2);
    bedMesh.position.y = -0.1;
    scene.add(bedMesh);

    // 3D Mesh
    const matProps = materialStyles[material];
    const meshMaterial = new THREE.MeshStandardMaterial({
      color: matProps.color,
      roughness: matProps.roughness,
      metalness: matProps.metalness,
      transparent: matProps.opacity < 1,
      opacity: matProps.opacity,
      wireframe: wireframe,
    });

    const mesh = new THREE.Mesh(geometry, meshMaterial);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    meshRef.current = mesh;

    // Simple Mouse Rotation Interaction
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const domElem = renderer.domElement;

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true;
      setIsRotating(false);
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !meshRef.current) return;

      const deltaMove = {
        x: e.clientX - previousMousePosition.x,
        y: e.clientY - previousMousePosition.y,
      };

      meshRef.current.rotation.y += deltaMove.x * 0.01;
      meshRef.current.rotation.x += deltaMove.y * 0.01;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    // Touch support for mobile PWA
    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDragging = true;
        setIsRotating(false);
        previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDragging || !meshRef.current || e.touches.length !== 1) return;
      const deltaMove = {
        x: e.touches[0].clientX - previousMousePosition.x,
        y: e.touches[0].clientY - previousMousePosition.y,
      };

      meshRef.current.rotation.y += deltaMove.x * 0.01;
      meshRef.current.rotation.x += deltaMove.y * 0.01;

      previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    };

    const handleTouchEnd = () => {
      isDragging = false;
    };

    domElem.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    domElem.addEventListener('touchstart', handleTouchStart);
    window.addEventListener('touchmove', handleTouchMove);
    window.addEventListener('touchend', handleTouchEnd);

    // Animation Loop
    let reqId: number;
    const animate = () => {
      reqId = requestAnimationFrame(animate);

      if (meshRef.current && isRotating && !isDragging) {
        meshRef.current.rotation.y += 0.005;
      }

      renderer.render(scene, camera);
    };
    animate();

    // Handle Resize
    const handleResize = () => {
      if (!mountRef.current || !renderer || !camera) return;
      const newW = mountRef.current.clientWidth;
      const newH = mountRef.current.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(reqId);
      window.removeEventListener('resize', handleResize);
      domElem.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);

      domElem.removeEventListener('touchstart', handleTouchStart);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);

      if (mountRef.current) {
        mountRef.current.innerHTML = '';
      }
    };
  }, [geometry, material, wireframe]);

  // Update wireframe & material properties on prop change
  useEffect(() => {
    if (meshRef.current) {
      const matProps = materialStyles[material];
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      mat.color.setHex(matProps.color);
      mat.roughness = matProps.roughness;
      mat.metalness = matProps.metalness;
      mat.wireframe = wireframe;
      mat.needsUpdate = true;
    }
  }, [material, wireframe]);

  const resetCamera = () => {
    if (meshRef.current) {
      meshRef.current.rotation.set(0, 0, 0);
      setIsRotating(true);
    }
  };

  return (
    <div className="relative w-full h-[400px] sm:h-[480px] lg:h-[540px] rounded-2xl glass-panel overflow-hidden group">
      
      {/* Three.js Canvas Container */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Top Left Status & Material Selector */}
      <div className="absolute top-4 left-4 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs font-mono text-slate-300 backdrop-blur-md">
          <Box className="w-3.5 h-3.5 text-indigo-400" />
          <span>{widthMm.toFixed(1)} x {depthMm.toFixed(1)} x {heightMm.toFixed(1)} mm</span>
        </div>

        {/* Material Selector Buttons */}
        <div className="flex items-center gap-1 p-1 rounded-lg bg-slate-900/90 border border-slate-800 backdrop-blur-md">
          {(['PLA', 'PETG', 'TPU', 'ABS'] as MaterialType[]).map((mat) => (
            <button
              key={mat}
              onClick={() => onMaterialChange(mat)}
              className={`px-2.5 py-1 rounded-md text-xs font-bold transition-all ${
                material === mat
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {mat}
            </button>
          ))}
        </div>
      </div>

      {/* Top Right View Controls */}
      <div className="absolute top-4 right-4 flex items-center gap-2">
        <button
          onClick={() => setWireframe(!wireframe)}
          title="Toggle Wireframe View"
          className={`p-2 rounded-lg border backdrop-blur-md transition-all ${
            wireframe
              ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300'
              : 'bg-slate-900/90 border-slate-800 text-slate-300 hover:text-white hover:border-slate-700'
          }`}
        >
          <Layers className="w-4 h-4" />
        </button>

        <button
          onClick={resetCamera}
          title="Reset View & Rotation"
          className="p-2 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300 hover:text-white hover:border-slate-700 backdrop-blur-md transition-all"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Bottom Hint Overlay */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-slate-800/80 text-[11px] text-slate-400 backdrop-blur-md pointer-events-none flex items-center gap-2">
        <Eye className="w-3.5 h-3.5 text-cyan-400" />
        <span>Drag to rotate 3D view | 220x220mm Print Bed Grid</span>
      </div>

    </div>
  );
};
