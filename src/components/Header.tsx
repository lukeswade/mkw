import React, { useState, useEffect } from 'react';
import { Cpu, Smartphone, ExternalLink } from 'lucide-react';

export const Header: React.FC = () => {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [isInstallable, setIsInstallable] = useState(false);

  useEffect(() => {
    const handleBeforeInstall = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setIsInstallable(true);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstall);
    return () => window.removeEventListener('beforeinstallprompt', handleBeforeInstall);
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setIsInstallable(false);
    }
    setDeferredPrompt(null);
  };

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-4 lg:px-8 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Logo & Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20">
            <img src="/icon-192.png" alt="MKW 3D Logo" className="w-full h-full rounded-[10px] object-cover" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                MKW 3D
              </span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                PWA
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono hidden sm:block">
              mattkwade.com
            </p>
          </div>
        </div>

        {/* Status Indicators & PWA Install */}
        <div className="flex items-center gap-3">
          
          {/* Cloudflare AI Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-indigo-500/30 text-xs text-slate-300 shadow-inner">
            <Cpu className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>Cloudflare Workers AI</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          </div>

          {/* PWA Install Button */}
          {isInstallable && (
            <button
              onClick={handleInstallClick}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-indigo-600 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-medium text-xs shadow-lg shadow-indigo-500/25 transition-all transform active:scale-95"
            >
              <Smartphone className="w-3.5 h-3.5" />
              <span>Install App</span>
            </button>
          )}

          {/* FilTracker Bridge Link */}
          <a
            href="https://filtracker.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300 transition-colors bg-emerald-950/40 px-3 py-1.5 rounded-lg border border-emerald-500/30 hover:border-emerald-500/60 shadow-lg shadow-emerald-500/10"
          >
            <span>FilTracker.com</span>
            <ExternalLink className="w-3 h-3 opacity-80" />
          </a>

          {/* Domain Tag */}
          <a
            href="https://mattkwade.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-indigo-300 transition-colors bg-slate-900/60 px-3 py-1.5 rounded-lg border border-slate-800"
          >
            <span>mattkwade.com</span>
            <ExternalLink className="w-3 h-3 opacity-60" />
          </a>
        </div>

      </div>
    </header>
  );
};
