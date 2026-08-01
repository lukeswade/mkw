import React from 'react';
import { ExternalLink, Cpu, Box, Zap } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-16 border-t border-slate-800/80 glass-panel py-8 px-4 lg:px-8">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        
        {/* Left Brand info */}
        <div className="flex flex-col items-center md:items-start gap-2">
          <div className="flex items-center gap-2">
            <Box className="w-4 h-4 text-cyan-400" />
            <span className="font-bold text-white tracking-tight text-sm">
              MKW 3D AI Studio
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
              v1.0.0
            </span>
          </div>
          <p className="text-xs text-slate-400 text-center md:text-left">
            Parametric 3D printable STL & 3MF model generation served on <span className="text-indigo-300">mattkwade.com</span>.
          </p>
        </div>

        {/* Center / Featured Links (CRITICAL: filtracker.com link) */}
        <div className="flex flex-wrap items-center justify-center gap-6">
          <a
            href="https://filtracker.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-slate-900 to-indigo-950/60 border border-indigo-500/40 text-cyan-300 hover:text-white font-semibold text-xs transition-all shadow-md hover:border-indigo-400 group"
          >
            <Zap className="w-4 h-4 text-amber-400 group-hover:scale-110 transition-transform" />
            <span>Track your filament at <strong>filtracker.com</strong></span>
            <ExternalLink className="w-3.5 h-3.5 opacity-70 group-hover:translate-x-0.5 transition-transform" />
          </a>
        </div>

        {/* Right Cloudflare & PWA Badges */}
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-1.5 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Cloudflare Workers AI</span>
          </div>
          <span>&copy; {new Date().getFullYear()} mattkwade.com</span>
        </div>

      </div>
    </footer>
  );
};
