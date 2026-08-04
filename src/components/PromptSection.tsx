import React, { useState } from 'react';
import { Sparkles, Wand2, ArrowRight } from 'lucide-react';

interface PromptSectionProps {
  onGenerate: (prompt: string) => void;
  isGenerating: boolean;
  conversation: {role: string, content: string}[];
}

export const PromptSection: React.FC<PromptSectionProps> = ({
  onGenerate,
  isGenerating,
  conversation,
}) => {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isGenerating) {
      onGenerate(prompt.trim());
      setPrompt('');
    }
  };



  return (
    <section className="relative pt-6 pb-4">
      {/* Background ambient lighting */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-indigo-600/10 blur-[80px] pointer-events-none rounded-full" />

      <div className="max-w-4xl mx-auto text-center px-4 relative z-10">
        
        {/* Prominent App Logo */}
        <div className="mb-6 flex justify-center">
          <div className="relative group animate-float">
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400 rounded-[28px] blur-xl opacity-40 group-hover:opacity-60 transition-opacity duration-500"></div>
            <div className="relative w-24 h-24 sm:w-32 sm:h-32 p-1 rounded-[28px] bg-gradient-to-br from-indigo-500/50 to-cyan-500/50 backdrop-blur-sm border border-white/10 shadow-2xl">
              <img src="/icon-512.png" alt="MKW 3D App Icon" className="w-full h-full object-cover rounded-[24px]" />
            </div>
          </div>
        </div>

        {/* Title */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-medium text-indigo-300 mb-4 shadow-lg">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Powered by Cloudflare Workers AI</span>
        </div>

        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-4 drop-shadow-sm">
          Turn Ideas into <span className="bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-lg">Printable 3D</span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto mb-6">
          Describe what you want to print. AI generates parametric 3D geometry with live WebGL preview & slicer estimates.
        </p>

        {/* Chat History */}
        {conversation.length > 0 && (
          <div className="max-w-2xl mx-auto mb-4 flex flex-col gap-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
            {conversation.filter(m => m.role === 'user').map((msg, i) => (
              <div key={i} className="self-end bg-indigo-500/20 border border-indigo-500/30 text-indigo-100 text-sm px-4 py-2.5 rounded-2xl rounded-tr-sm shadow-sm inline-block max-w-[85%] text-left">
                {msg.content}
              </div>
            ))}
          </div>
        )}

        {/* Input Box */}
        <form onSubmit={handleSubmit} className="relative max-w-2xl mx-auto mb-6">
          <div className="relative flex items-center glass-panel-glow rounded-2xl p-1.5 transition-all">
            <div className="pl-3 text-slate-400">
              <Wand2 className="w-5 h-5 text-indigo-400 animate-pulse" />
            </div>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. A wall mount headphone hook or SD card desk organizer..."
              className="w-full bg-transparent px-3 py-3 text-sm sm:text-base text-white placeholder-slate-500 focus:outline-none"
              disabled={isGenerating}
            />
            <button
              type="submit"
              disabled={!prompt.trim() || isGenerating}
              className="flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 disabled:opacity-50 text-white font-semibold text-sm shadow-lg shadow-indigo-600/30 transition-all active:scale-95 whitespace-nowrap"
            >
              {isGenerating ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Generating 3D...</span>
                </>
              ) : (
                <>
                  <span>Generate Model</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </form>

        {/* Quick Maker Utility Chips */}
        <div className="flex flex-wrap items-center justify-center gap-2 max-w-2xl mx-auto mb-6">
          <button
            type="button"
            onClick={() => onGenerate('A calibration filament swatch card with stepped thickness windows')}
            disabled={isGenerating}
            className="text-xs px-3 py-1.5 rounded-full bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition-all shadow-sm"
          >
            🧪 Filament Swatch
          </button>
          <button
            type="button"
            onClick={() => onGenerate('A spool rim tag clip for labeling filament spools')}
            disabled={isGenerating}
            className="text-xs px-3 py-1.5 rounded-full bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition-all shadow-sm"
          >
            🏷️ Spool Tag Clip
          </button>
          <button
            type="button"
            onClick={() => onGenerate('A 20mm 3D printing calibration cube')}
            disabled={isGenerating}
            className="text-xs px-3 py-1.5 rounded-full bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition-all shadow-sm"
          >
            🎲 Calibration Cube
          </button>
          <button
            type="button"
            onClick={() => onGenerate('SD and MicroSD card desk organizer tray')}
            disabled={isGenerating}
            className="text-xs px-3 py-1.5 rounded-full bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition-all shadow-sm"
          >
            💾 SD Card Tray
          </button>
        </div>

        {/* FilTracker Integration Banner */}
        <div className="max-w-2xl mx-auto p-3.5 rounded-xl bg-gradient-to-r from-emerald-950/60 via-slate-900 to-indigo-950/60 border border-emerald-500/30 flex items-center justify-between gap-3 text-left shadow-lg">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">FilTracker Bridge</span>
              <span className="text-[10px] px-2 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Free Inventory Tool</span>
            </div>
            <p className="text-xs text-slate-300 mt-0.5">
              Managing 3D printing spools? Track your filament inventory, spool weights & print costs at <strong className="text-emerald-400 font-semibold">FilTracker.com</strong>
            </p>
          </div>
          <a
            href="https://filtracker.com"
            target="_blank"
            rel="noreferrer"
            className="shrink-0 px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs shadow-md transition-all whitespace-nowrap"
          >
            Open FilTracker ↗
          </a>
        </div>

      </div>
    </section>
  );
};
