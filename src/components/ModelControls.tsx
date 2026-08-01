import React from 'react';
import { Download, Sliders, FileCode } from 'lucide-react';
import { ModelParams } from '../types';

interface ModelControlsProps {
  params: ModelParams;
  onChangeParams: (newParams: ModelParams) => void;
  onDownloadSTL: () => void;
  onDownload3MF: () => void;
}

export const ModelControls: React.FC<ModelControlsProps> = ({
  params,
  onChangeParams,
  onDownloadSTL,
  onDownload3MF,
}) => {
  const updateField = (field: keyof ModelParams, value: any) => {
    onChangeParams({
      ...params,
      [field]: value,
    });
  };

  return (
    <div className="glass-panel rounded-2xl p-5 lg:p-6 flex flex-col justify-between h-full">
      <div>
        
        {/* Model Title & Details */}
        <div className="flex items-start justify-between mb-4 border-b border-slate-800 pb-3">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span>{params.title || 'Parametric 3D Model'}</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1 line-clamp-2">
              {params.description || 'AI-generated 3D printable geometry.'}
            </p>
          </div>
          <span className="text-[11px] font-mono font-semibold px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 uppercase">
            {params.type}
          </span>
        </div>

        {/* Parametric Sliders Header */}
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-300 mb-4">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span>Parametric Controls</span>
        </div>

        {/* Sliders Grid */}
        <div className="space-y-4 mb-6">
          
          {/* Width (X) */}
          <div>
            <div className="flex justify-between text-xs text-slate-300 font-mono mb-1">
              <span>Width (X)</span>
              <span className="text-cyan-400 font-bold">{params.width} mm</span>
            </div>
            <input
              type="range"
              min="15"
              max="200"
              step="1"
              value={params.width}
              onChange={(e) => updateField('width', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Depth (Y) */}
          <div>
            <div className="flex justify-between text-xs text-slate-300 font-mono mb-1">
              <span>Depth (Z)</span>
              <span className="text-cyan-400 font-bold">{params.depth} mm</span>
            </div>
            <input
              type="range"
              min="15"
              max="200"
              step="1"
              value={params.depth}
              onChange={(e) => updateField('depth', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Height (Z) */}
          <div>
            <div className="flex justify-between text-xs text-slate-300 font-mono mb-1">
              <span>Height (Y)</span>
              <span className="text-cyan-400 font-bold">{params.height} mm</span>
            </div>
            <input
              type="range"
              min="3"
              max="150"
              step="1"
              value={params.height}
              onChange={(e) => updateField('height', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Wall Thickness */}
          <div>
            <div className="flex justify-between text-xs text-slate-300 font-mono mb-1">
              <span>Wall Thickness</span>
              <span className="text-indigo-400 font-bold">{params.wallThickness} mm</span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              step="0.5"
              value={params.wallThickness}
              onChange={(e) => updateField('wallThickness', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Hole Diameter (if applicable) */}
          {params.holeDiameter !== undefined && (
            <div>
              <div className="flex justify-between text-xs text-slate-300 font-mono mb-1">
                <span>Hole / Screw Cutout Dia.</span>
                <span className="text-indigo-400 font-bold">{params.holeDiameter} mm</span>
              </div>
              <input
                type="range"
                min="0"
                max="12"
                step="0.5"
                value={params.holeDiameter}
                onChange={(e) => updateField('holeDiameter', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
            </div>
          )}

          {/* Text Label Input (for Keychains) */}
          {params.type === 'keychain' && (
            <div>
              <label className="block text-xs text-slate-300 font-mono mb-1">
                Custom Embossed Text
              </label>
              <input
                type="text"
                maxLength={12}
                value={params.textLabel || ''}
                onChange={(e) => updateField('textLabel', e.target.value.toUpperCase())}
                placeholder="MKW"
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white uppercase font-mono focus:border-indigo-500 focus:outline-none"
              />
            </div>
          )}

        </div>
      </div>

      {/* Export Buttons */}
      <div className="space-y-2.5 pt-4 border-t border-slate-800">
        
        {/* Download STL */}
        <button
          onClick={onDownloadSTL}
          className="w-full flex items-center justify-center gap-2.5 px-4 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-bold text-sm shadow-lg shadow-indigo-600/30 transition-all transform active:scale-95 group"
        >
          <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
          <span>Download .STL File</span>
          <span className="text-[10px] font-normal px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-200">
            Binary
          </span>
        </button>

        {/* Download 3MF */}
        <button
          onClick={onDownload3MF}
          className="w-full flex items-center justify-center gap-2.5 px-4 py-3 rounded-xl bg-slate-900 hover:bg-slate-800 border border-cyan-500/40 text-cyan-300 hover:text-white font-bold text-sm shadow-md transition-all transform active:scale-95 group"
        >
          <FileCode className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" />
          <span>Download .3MF Package</span>
          <span className="text-[10px] font-normal px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-200">
            Bambu / Prusa
          </span>
        </button>

      </div>
    </div>
  );
};
