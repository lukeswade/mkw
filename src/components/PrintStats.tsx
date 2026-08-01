import React from 'react';
import { Gauge, Clock, Scale, Ruler, CheckCircle2, AlertTriangle, Printer } from 'lucide-react';
import { PrintAnalytics, MaterialType } from '../types';

interface PrintStatsProps {
  analytics: PrintAnalytics;
  material: MaterialType;
}

export const PrintStats: React.FC<PrintStatsProps> = ({ analytics, material }) => {
  return (
    <div className="glass-panel rounded-2xl p-5 lg:p-6 mb-6">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Gauge className="w-5 h-5 text-indigo-400" />
          <h3 className="text-base font-bold text-white">Slicer Print Analytics</h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Material: <strong className="text-cyan-400">{material}</strong>
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        
        {/* Volume */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Mesh Volume</span>
            <Ruler className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <p className="text-lg font-bold text-white font-mono">
            {analytics.volumeCm3} <span className="text-xs font-normal text-slate-400">cm³</span>
          </p>
        </div>

        {/* Weight */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Est. Weight</span>
            <Scale className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <p className="text-lg font-bold text-white font-mono">
            {analytics.weightGrams} <span className="text-xs font-normal text-slate-400">g</span>
          </p>
        </div>

        {/* Print Time */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Print Time</span>
            <Clock className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <p className="text-lg font-bold text-white font-mono">
            {analytics.estimatedTimeMin} <span className="text-xs font-normal text-slate-400">min</span>
          </p>
        </div>

        {/* Filament Length */}
        <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Filament Used</span>
            <Printer className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <p className="text-lg font-bold text-white font-mono">
            {analytics.filamentLengthMeters} <span className="text-xs font-normal text-slate-400">m</span>
          </p>
        </div>

      </div>

      {/* Bed Compatibility */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/80 text-xs">
        <span className="text-slate-400 font-medium">Printer Bed Fit:</span>

        <div className="flex flex-wrap items-center gap-2">
          
          {/* Bambu Lab */}
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-lg border font-mono ${
            analytics.bedCompatibility.bambuLab
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
          }`}>
            {analytics.bedCompatibility.bambuLab ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            )}
            <span>Bambu Lab (256mm)</span>
          </div>

          {/* Ender 3 */}
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-lg border font-mono ${
            analytics.bedCompatibility.ender3
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
          }`}>
            {analytics.bedCompatibility.ender3 ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            )}
            <span>Ender 3 (220mm)</span>
          </div>

          {/* Prusa Mini */}
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-lg border font-mono ${
            analytics.bedCompatibility.mini
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
          }`}>
            {analytics.bedCompatibility.mini ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            )}
            <span>Prusa Mini (180mm)</span>
          </div>

        </div>
      </div>

    </div>
  );
};
