import React, { useState } from 'react';
import { Sparkles, Wand2, ArrowRight, Lightbulb, Box, Grid, Anchor, Paperclip, Key, Smartphone, Hexagon } from 'lucide-react';
import { PRESET_IDEAS } from '../lib/preset-ideas';
import { ModelParams } from '../types';

interface PromptSectionProps {
  onGenerate: (prompt: string) => void;
  onSelectPreset: (params: ModelParams) => void;
  isGenerating: boolean;
}

export const PromptSection: React.FC<PromptSectionProps> = ({
  onGenerate,
  onSelectPreset,
  isGenerating,
}) => {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isGenerating) {
      onGenerate(prompt.trim());
    }
  };

  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'Grid': return <Grid className="w-4 h-4 text-cyan-400" />;
      case 'Anchor': return <Anchor className="w-4 h-4 text-indigo-400" />;
      case 'Paperclip': return <Paperclip className="w-4 h-4 text-emerald-400" />;
      case 'Key': return <Key className="w-4 h-4 text-amber-400" />;
      case 'Smartphone': return <Smartphone className="w-4 h-4 text-purple-400" />;
      case 'Hexagon': return <Hexagon className="w-4 h-4 text-pink-400" />;
      default: return <Box className="w-4 h-4 text-cyan-400" />;
    }
  };

  return (
    <section className="relative pt-6 pb-4">
      {/* Background ambient lighting */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-indigo-600/10 blur-[80px] pointer-events-none rounded-full" />

      <div className="max-w-4xl mx-auto text-center px-4">
        
        {/* Title */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-medium text-indigo-300 mb-4">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Powered by Cloudflare Workers AI</span>
        </div>

        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-white mb-3">
          Turn 3D Ideas into Printable <span className="bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">STL & 3MF Files</span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto mb-6">
          Describe what you want to print. AI generates parametric 3D geometry with live WebGL preview & slicer estimates.
        </p>

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

        {/* Presets & Ideas Header */}
        <div className="flex items-center justify-center gap-2 mb-3">
          <Lightbulb className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Or pick a 3D Print Idea Preset
          </span>
        </div>

        {/* Presets Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 max-w-4xl mx-auto">
          {PRESET_IDEAS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => {
                setPrompt(preset.prompt);
                onSelectPreset(preset.params);
              }}
              className="flex flex-col items-center justify-center p-3 rounded-xl glass-panel hover:border-indigo-500/50 hover:bg-slate-900/90 transition-all text-left group"
            >
              <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 group-hover:border-indigo-500/40 mb-2 transition-colors">
                {getIcon(preset.icon)}
              </div>
              <span className="text-xs font-medium text-slate-200 text-center line-clamp-1 group-hover:text-indigo-300">
                {preset.title}
              </span>
            </button>
          ))}
        </div>

      </div>
    </section>
  );
};
