import React, { useState, useEffect, lazy, Suspense } from 'react';
import { Header } from './components/Header';
import { PromptSection } from './components/PromptSection';
import { ModelControls } from './components/ModelControls';
import { PrintStats } from './components/PrintStats';
import { Footer } from './components/Footer';
import { PRESET_IDEAS } from './lib/preset-ideas';
import { downloadFile } from './lib/download';
// Type-only: erased at build, so it costs nothing in the bundle.
import type * as THREE from 'three';
import { ModelParams, MaterialType, PrintAnalytics } from './types';
import { AlertCircle, Loader2 } from 'lucide-react';

// three.js + the CSG evaluator are ~176KB gzipped and jszip another ~30KB.
// Loading them on demand lets the header, prompt box and hero paint first;
// the viewport fades in a moment later.
const ThreeCanvas = lazy(() =>
  import('./components/ThreeCanvas').then((m) => ({ default: m.ThreeCanvas }))
);

// One shared promise so the engine is fetched and evaluated exactly once,
// however many callers ask for it.
let enginePromise: Promise<typeof import('./lib/three-engine')> | null = null;
const loadEngine = () => (enginePromise ??= import('./lib/three-engine'));

/** Shown while three.js is still downloading, so the shell paints immediately. */
const ViewportLoading: React.FC = () => (
  <div className="w-full h-full flex flex-col items-center justify-center gap-3 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
    <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
    <p className="text-sm text-slate-400 animate-pulse">Loading 3D viewport…</p>
  </div>
);

export const App: React.FC = () => {
  // Default initial model: SD Card Organizer Tray preset
  const [modelParams, setModelParams] = useState<ModelParams>(PRESET_IDEAS[0].params);
  const [material, setMaterial] = useState<MaterialType>('PLA');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversation, setConversation] = useState<{role: string, content: string}[]>([]);
  const [xrayMode, setXrayMode] = useState(false);

  // Auto-dismiss error
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const [currentGeometry, setCurrentGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [printAnalytics, setPrintAnalytics] = useState<PrintAnalytics | null>(null);

  // Generate or load 3D BufferGeometry on parameter change. The engine is
  // dynamically imported, so the very first run also fetches three.js.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const engine = await loadEngine();
        if (cancelled) return;

        if (modelParams.type === 'external' && modelParams.externalUrl) {
          setIsGenerating(true);
          try {
            const geometry = await engine.loadExternalStl(modelParams.externalUrl, modelParams);
            if (!cancelled) setCurrentGeometry(geometry);
          } catch (err) {
            console.error('Error loading external STL:', err);
            if (!cancelled) setError('Failed to load official 3D model.');
          } finally {
            if (!cancelled) setIsGenerating(false);
          }
        } else {
          setCurrentGeometry(engine.buildProceduralGeometry(modelParams));
        }
      } catch (err) {
        console.error('Failed to load the 3D engine:', err);
        if (!cancelled) setError('Could not load the 3D engine. Please reload.');
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [modelParams]);

  // Calculate slicer analytics once geometry exists.
  useEffect(() => {
    if (!currentGeometry) return;
    let cancelled = false;

    loadEngine()
      .then((engine) => {
        if (!cancelled) setPrintAnalytics(engine.calculatePrintAnalytics(currentGeometry, material));
      })
      .catch(() => {/* the geometry effect above already surfaced this */});

    return () => {
      cancelled = true;
    };
  }, [currentGeometry, material]);

  const handleGenerate = async (promptText: string) => {
    setIsGenerating(true);
    setError(null);

    const updatedConversation = [...conversation, { role: 'user', content: promptText }];
    setConversation(updatedConversation);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, messages: updatedConversation }),
      });

      const data = await response.json();

      if (data.success && data.params) {
        setModelParams(data.params);
        setConversation([...updatedConversation, { role: 'assistant', content: JSON.stringify(data.params) }]);
      } else {
        throw new Error(data.error || 'Failed to generate 3D model.');
      }
    } catch (err: any) {
      console.warn('API Generation fallback:', err);
      // Heuristic client fallback on network error
      const words = promptText.toLowerCase();
      let matchedPreset = PRESET_IDEAS[0].params;
      if (words.includes('hook') || words.includes('headphone')) matchedPreset = PRESET_IDEAS[1].params;
      else if (words.includes('cable') || words.includes('clip')) matchedPreset = PRESET_IDEAS[2].params;
      else if (words.includes('key') || words.includes('tag')) matchedPreset = PRESET_IDEAS[3].params;
      else if (words.includes('phone') || words.includes('stand')) matchedPreset = PRESET_IDEAS[4].params;
      else if (words.includes('hex') || words.includes('tray')) matchedPreset = PRESET_IDEAS[5].params;

      setModelParams({
        ...matchedPreset,
        title: promptText.slice(0, 24) || matchedPreset.title,
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const filenameFor = (ext: 'stl' | '3mf') =>
    `${(modelParams.title || '3d_model').toLowerCase().replace(/[^a-z0-9]/g, '_')}.${ext}`;

  // Download STL
  const handleDownloadSTL = async () => {
    if (!currentGeometry) return;
    try {
      const { exportBinarySTL } = await import('./lib/stl-exporter');
      const buffer = exportBinarySTL(currentGeometry);
      downloadFile(new Blob([buffer], { type: 'model/stl' }), filenameFor('stl'));
    } catch (err) {
      console.error('Error exporting STL:', err);
      setError('Failed to generate STL file.');
    }
  };

  // Download 3MF
  const handleDownload3MF = async () => {
    if (!currentGeometry) return;
    try {
      const { export3MF } = await import('./lib/3mf-exporter');
      const blob = await export3MF(currentGeometry, modelParams.title);
      downloadFile(blob, filenameFor('3mf'));
    } catch (err) {
      console.error('Error exporting 3MF:', err);
      setError('Failed to generate 3MF package.');
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col relative bg-slate-950">
      
      {/* 3D Viewport - Background Fullscreen */}
      <div className="absolute inset-0 z-0 pointer-events-auto">
        {currentGeometry ? (
          <Suspense fallback={<ViewportLoading />}>
            <ThreeCanvas
              geometry={currentGeometry}
              material={material}
              onMaterialChange={setMaterial}
              widthMm={modelParams.width}
              depthMm={modelParams.depth}
              heightMm={modelParams.height}
              isGenerating={isGenerating}
              xrayMode={xrayMode}
            />
          </Suspense>
        ) : (
          <ViewportLoading />
        )}
      </div>

      {/* Floating UI Container */}
      <div className="relative z-10 h-full flex flex-col pointer-events-none overflow-y-auto">
        
        {/* Header - Top Left */}
        <div className="pointer-events-auto shrink-0 w-full">
          <Header />
        </div>

        {/* Prompt Section - Top Center floating */}
        <div className="pointer-events-auto shrink-0 w-full mt-4">
          <PromptSection
            conversation={conversation}
            onGenerate={handleGenerate}
            isGenerating={isGenerating}
          />
        </div>

        {/* Error Alert */}
        <div className={`pointer-events-auto max-w-7xl mx-auto px-4 lg:px-8 mt-2 transition-all duration-300 w-full ${error ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none hidden'}`}>
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between gap-2 shadow-lg shadow-rose-500/10 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="p-1 hover:bg-rose-500/20 rounded-md transition-colors text-rose-400">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
        </div>

        {/* Main Floating Content (Controls + Stats) */}
        <div className="flex-1 w-full max-w-7xl mx-auto px-4 lg:px-8 py-6 flex flex-col justify-end pointer-events-none gap-6 pb-20">
          
          {/* Controls & Export Floating Panel */}
          <div className="w-full lg:w-[400px] pointer-events-auto self-end mt-auto">
            <ModelControls
              params={modelParams}
              onChangeParams={setModelParams}
              onDownloadSTL={handleDownloadSTL}
              onDownload3MF={handleDownload3MF}
            />
          </div>

          {/* Slicer Print Analytics Floating Panel */}
          {printAnalytics && (
            <div className="w-full pointer-events-auto">
              <PrintStats analytics={printAnalytics} material={material} />
            </div>
          )}
        </div>

      </div>

      {/* Footer - Bottom */}
      <div className="absolute bottom-0 w-full z-20 pointer-events-auto">
        <Footer />
      </div>

      {/* Floating X-Ray Toggle */}
      <div className="absolute top-24 right-4 lg:right-8 z-20 pointer-events-auto">
        <button
          onClick={() => setXrayMode(!xrayMode)}
          className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all shadow-lg backdrop-blur-md border ${
            xrayMode 
              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-cyan-500/20' 
              : 'bg-slate-900/60 text-slate-300 border-slate-700/50 hover:bg-slate-800/80 hover:text-white'
          }`}
        >
          {xrayMode ? '👁 X-Ray Mode: ON' : '👀 X-Ray Mode: OFF'}
        </button>
      </div>

    </div>
  );
};

export default App;
