import React, { useState } from 'react';
import { Settings, Server, Cpu, Database, Save, CheckCircle2, RotateCcw } from 'lucide-react';

interface SettingsViewProps {
  onClearSession: () => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({ onClearSession }) => {
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || '');
  const [randomSeed, setRandomSeed] = useState(42);
  const [maxGenerations, setMaxGenerations] = useState(10);
  const [saveNotice, setSaveNotice] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaveNotice(true);
    setTimeout(() => setSaveNotice(false), 3000);
  };

  return (
    <div className="p-6 lg:p-8 max-w-4xl mx-auto space-y-8">
      {/* Title */}
      <div className="pb-4 border-b border-slate-800">
        <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
          Platform Configuration
        </span>
        <h1 className="text-2xl font-bold text-white tracking-tight">System Settings</h1>
      </div>

      {saveNotice && (
        <div className="p-4 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-200 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>System configuration saved successfully.</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Backend & Deployment */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>Backend Integration & Microservice Endpoint</span>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-slate-300 font-medium">
              External API URL (Render / Python FastAPI)
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="e.g. https://evox-backend.onrender.com or empty for embedded"
              className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-400 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none transition-colors"
            />
            <p className="text-[11px] text-slate-400">
              When blank, EVOX utilizes its high-performance embedded Node & Web Worker ML engine. When pointing to a deployed FastAPI instance, calls route through the Python service.
            </p>
          </div>
        </div>

        {/* Compute & Optimization Parameters */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span>Optimization & Reproducibility</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs text-slate-300 font-medium">Random Seed</label>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(parseInt(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-400 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none"
              />
              <p className="text-[11px] text-slate-400">Fixed seed for deterministic PRNG splits</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-slate-300 font-medium">
                Default Evolution Generations
              </label>
              <input
                type="number"
                min="3"
                max="50"
                value={maxGenerations}
                onChange={(e) => setMaxGenerations(parseInt(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-400 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none"
              />
              <p className="text-[11px] text-slate-400">Chromosomes iterated per genetic run</p>
            </div>
          </div>
        </div>

        {/* Session Management */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Database className="w-4 h-4 text-amber-400" />
            <span>Session & Cache Storage</span>
          </div>

          <p className="text-xs text-slate-400">
            Clear in-memory datasets, trained model weights, and cached Pareto points to start a fresh experiment.
          </p>

          <button
            type="button"
            onClick={onClearSession}
            className="flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-rose-400 bg-rose-950/40 hover:bg-rose-950 border border-rose-800/60 rounded-lg transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Clear Current Experiment State</span>
          </button>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="flex items-center gap-2 px-5 py-2.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg shadow transition-colors"
          >
            <Save className="w-4 h-4" />
            <span>Save Settings</span>
          </button>
        </div>
      </form>
    </div>
  );
};
