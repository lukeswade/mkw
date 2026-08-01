import React, { useState, useMemo, useEffect } from 'react';
import { Header } from './components/Header';
import { PromptSection } from './components/PromptSection';
import { ThreeCanvas } from './components/ThreeCanvas';
import { ModelControls } from './components/ModelControls';
import { PrintStats } from './components/PrintStats';
import { Footer } from './components/Footer';
import { PRESET_IDEAS } from './lib/preset-ideas';
import { buildProceduralGeometry, calculatePrintAnalytics } from './lib/procedural-3d';
import { exportBinarySTL, downloadFile } from './lib/stl-exporter';
import { export3MF } from './lib/3mf-exporter';
import { ModelParams, MaterialType } from './types';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  // Default initial model: SD Card Organizer Tray preset
  const [modelParams, setModelParams] = useState<ModelParams>(PRESET_IDEAS[0].params);
  const [material, setMaterial] = useState<MaterialType>('PLA');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-dismiss error
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Generate 3D BufferGeometry on parameter change
  const currentGeometry = useMemo(() => {
    return buildProceduralGeometry(modelParams);
  }, [modelParams]);

  // Calculate slicer analytics
  const printAnalytics = useMemo(() => {
    return calculatePrintAnalytics(currentGeometry, material);
  }, [currentGeometry, material]);

  // Handle AI Prompt Submit
  const handleGenerate = async (promptText: string) => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText }),
      });

      const data = await response.json();

      if (data.success && data.params) {
        setModelParams(data.params);
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

  // Download STL
  const handleDownloadSTL = () => {
    try {
      const buffer = exportBinarySTL(currentGeometry);
      const blob = new Blob([buffer], { type: 'model/stl' });
      const filename = `${(modelParams.title || '3d_model').toLowerCase().replace(/[^a-z0-9]/g, '_')}.stl`;
      downloadFile(blob, filename);
    } catch (err) {
      console.error('Error exporting STL:', err);
      setError('Failed to generate STL file.');
    }
  };

  // Download 3MF
  const handleDownload3MF = async () => {
    try {
      const blob = await export3MF(currentGeometry, modelParams.title);
      const filename = `${(modelParams.title || '3d_model').toLowerCase().replace(/[^a-z0-9]/g, '_')}.3mf`;
      downloadFile(blob, filename);
    } catch (err) {
      console.error('Error exporting 3MF:', err);
      setError('Failed to generate 3MF package.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        
        {/* Header */}
        <Header />

        {/* Hero & Prompt Section */}
        <PromptSection
          onGenerate={handleGenerate}
          onSelectPreset={(presetParams) => setModelParams(presetParams)}
          isGenerating={isGenerating}
        />

        {/* Error Alert */}
        <div className={`max-w-7xl mx-auto px-4 lg:px-8 mt-4 transition-all duration-300 ${error ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none absolute'}`}>
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between gap-2 shadow-lg shadow-rose-500/10">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="p-1 hover:bg-rose-500/20 rounded-md transition-colors text-rose-400">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
        </div>

        {/* Main 3D Studio Content */}
        <main className="max-w-7xl mx-auto px-4 lg:px-8 py-6">
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
            
            {/* 3D Viewport Column (7 cols on desktop) */}
            <div className="lg:col-span-7">
              <ThreeCanvas
                geometry={currentGeometry}
                material={material}
                onMaterialChange={setMaterial}
                widthMm={modelParams.width}
                depthMm={modelParams.depth}
                heightMm={modelParams.height}
                isGenerating={isGenerating}
              />
            </div>

            {/* Parametric Controls & Export Column (5 cols on desktop) */}
            <div className="lg:col-span-5">
              <ModelControls
                params={modelParams}
                onChangeParams={setModelParams}
                onDownloadSTL={handleDownloadSTL}
                onDownload3MF={handleDownload3MF}
              />
            </div>

          </div>

          {/* Slicer Print Analytics */}
          <PrintStats analytics={printAnalytics} material={material} />

        </main>
      </div>

      {/* Footer (featuring filtracker.com link) */}
      <Footer />
    </div>
  );
};

export default App;
