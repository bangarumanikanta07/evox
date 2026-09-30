import React from 'react';
import { Play, Sparkles, SlidersHorizontal, Layers, Activity } from 'lucide-react';

interface HeaderProps {
  currentView: string;
  onNavigate: (view: string) => void;
  onLaunchDemo: () => void;
  onOpenPresentation: () => void;
  isDemoLoading: boolean;
  systemStatus: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentView,
  onNavigate,
  onLaunchDemo,
  onOpenPresentation,
  isDemoLoading,
  systemStatus,
}) => {
  const navLinks = [
    { id: 'overview', label: 'Overview' },
    { id: 'dataset', label: 'Dataset Lab' },
    { id: 'automl', label: 'AutoML' },
    { id: 'optimization', label: 'Optimization' },
    { id: 'models', label: 'Model Arena' },
    { id: 'anomalies', label: 'Anomaly Radar' },
    { id: 'reports', label: 'Intelligence' },
  ];

  return (
    <header className="sticky top-0 z-40 w-full bg-slate-950/85 backdrop-blur-md border-b border-slate-800/80 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Zone 1: Single text element wordmark strictly adhering to Top Bar Contract */}
        <button
          onClick={() => onNavigate('landing')}
          className="text-xl font-bold tracking-tight text-white hover:text-cyan-400 transition-colors shrink-0 flex items-center gap-2"
        >
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.8)]"></span>
          <span>EVOX</span>
        </button>

        {/* Zone 2: 4-6 clean text navigation links with subtle hover underline */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-400">
          {navLinks.map((link) => (
            <button
              key={link.id}
              onClick={() => onNavigate(link.id)}
              className={`transition-colors whitespace-nowrap py-1 relative ${
                currentView === link.id
                  ? 'text-white font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {link.label}
              {currentView === link.id && (
                <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400 rounded-full" />
              )}
            </button>
          ))}
        </nav>

        {/* Zone 3: 1-2 primary actions */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={onOpenPresentation}
            title="Interactive Presentation Tour for Judges"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-700/80 rounded-md hover:bg-slate-800 hover:text-white transition-colors whitespace-nowrap"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Presentation Mode</span>
          </button>

          <button
            onClick={onLaunchDemo}
            disabled={isDemoLoading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 rounded-md shadow-sm transition-all whitespace-nowrap"
          >
            <Play className={`w-3.5 h-3.5 fill-current ${isDemoLoading ? 'animate-spin' : ''}`} />
            <span>{isDemoLoading ? 'Running Demo...' : 'Launch Demo'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
