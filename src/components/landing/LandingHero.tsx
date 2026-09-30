import React, { useState } from 'react';
import { ArrowRight, Play, Database, Search, GitFork, BrainCircuit, CheckCircle2, Sparkles } from 'lucide-react';
import { NeuralCanvas } from '../common/NeuralCanvas';

interface LandingHeroProps {
  onLaunchLab: () => void;
  onLaunchDemo: () => void;
  onExploreWorkflow: () => void;
  isDemoLoading: boolean;
}

export const LandingHero: React.FC<LandingHeroProps> = ({
  onLaunchLab,
  onLaunchDemo,
  onExploreWorkflow,
  isDemoLoading,
}) => {
  const [activeStage, setActiveStage] = useState(2);

  const pipelineStages = [
    { id: 0, label: 'DATA', desc: 'Ingestion & Quality Profiling', icon: Database },
    { id: 1, label: 'ANALYZE', desc: 'Autonomous Problem Detection', icon: Search },
    { id: 2, label: 'OPTIMIZE', desc: 'AutoML & Multi-Objective Pareto', icon: GitFork },
    { id: 3, label: 'EXPLAIN', desc: 'SHAP & Permutation Attribution', icon: BrainCircuit },
    { id: 4, label: 'DECIDE', desc: 'Actionable Intelligence & Edge Deploy', icon: CheckCircle2 },
  ];

  return (
    <div className="relative min-h-[calc(100vh-65px)] flex flex-col justify-between overflow-hidden bg-slate-950 px-4 sm:px-6 lg:px-8 py-12">
      {/* Background Neural Canvas */}
      <div className="absolute inset-0 pointer-events-none opacity-40 z-0">
        <NeuralCanvas activeStage={activeStage} />
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950/40 via-transparent to-slate-950"></div>
      </div>

      {/* Main Content Area */}
      <div className="relative z-10 max-w-5xl mx-auto my-auto text-center space-y-8 pt-6">
        {/* Subtle kicker text (Anti-slop: clean text, no mechanical prefixes) */}
        <div className="inline-flex items-center gap-2 text-xs font-semibold text-cyan-400 tracking-wider uppercase">
          <span>Adaptive Machine Learning Laboratory</span>
          <span aria-hidden="true">·</span>
          <span>Automated Pareto Fronts</span>
        </div>

        {/* Hero Title & Tagline with text-wrap: balance */}
        <div className="space-y-4">
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white leading-none [text-wrap:balance]">
            EVOX
          </h1>
          <p className="text-2xl sm:text-3xl lg:text-4xl font-semibold tracking-tight text-slate-200 [text-wrap:balance]">
            Adaptive Intelligence. <span className="text-cyan-400">Optimized Decisions.</span>
          </p>
        </div>

        {/* Hero Subtitle */}
        <p className="max-w-2xl mx-auto text-base sm:text-lg text-slate-400 leading-relaxed [text-wrap:balance]">
          An intelligent machine-learning laboratory that transforms raw data into
          optimized, explainable, and actionable intelligence.
        </p>

        {/* Primary Action Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
          <button
            onClick={onLaunchLab}
            className="flex items-center gap-2 px-6 py-3 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg shadow-lg shadow-cyan-500/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Launch Intelligence Lab</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={onLaunchDemo}
            disabled={isDemoLoading}
            className="flex items-center gap-2 px-6 py-3 text-sm font-semibold text-slate-200 bg-slate-900 border border-slate-700/80 hover:bg-slate-800 hover:text-white rounded-lg transition-all"
          >
            <Play className={`w-4 h-4 text-cyan-400 ${isDemoLoading ? 'animate-spin' : ''}`} />
            <span>{isDemoLoading ? 'Synthesizing Pipeline...' : 'Run Demo Pipeline'}</span>
          </button>

          <button
            onClick={onExploreWorkflow}
            className="px-4 py-3 text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors"
          >
            Explore Workflow &rarr;
          </button>
        </div>

        {/* Visual Pipeline Progression: DATA → ANALYZE → OPTIMIZE → EXPLAIN → DECIDE */}
        <div className="pt-12 max-w-4xl mx-auto">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
            {pipelineStages.map((stage, idx) => {
              const Icon = stage.icon;
              const isSelected = activeStage === stage.id;
              return (
                <button
                  key={stage.id}
                  onClick={() => setActiveStage(stage.id)}
                  className={`p-3 text-left rounded-lg transition-all border ${
                    isSelected
                      ? 'bg-slate-800/90 border-cyan-500/40 shadow-sm'
                      : 'border-transparent hover:bg-slate-900/40 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <Icon className={`w-4 h-4 ${isSelected ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <span className="text-[10px] text-slate-400 font-mono">0{idx + 1}</span>
                  </div>
                  <h4 className={`text-xs font-semibold tracking-wider ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                    {stage.label}
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                    {stage.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Hero Footer Meta (Anti-slop: clean text, no mechanical fake telemetry tickers) */}
      <div className="relative z-10 max-w-5xl mx-auto w-full pt-8 border-t border-slate-800/60 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-4">
        <div className="flex items-center gap-3">
          <span>Multi-Objective Pareto Engine</span>
          <span aria-hidden="true">·</span>
          <span>Genetic Evolutionary Search</span>
          <span aria-hidden="true">·</span>
          <span>Isolation Forest Radar</span>
        </div>
        <span>Production-Ready Monorepo Architecture</span>
      </div>
    </div>
  );
};
