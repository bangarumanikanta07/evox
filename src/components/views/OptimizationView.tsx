import React, { useState } from 'react';
import {
  GitFork,
  SlidersHorizontal,
  Dna,
  Play,
  CheckCircle2,
  Trophy,
  ArrowRight,
  Info,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { CandidateModel, Dataset, ObjectiveWeights, EvolutionaryOptimizationRun } from '../../types';
import { calculateWeightedScores } from '../../lib/ml-engine';
import { api } from '../../services/api';

interface OptimizationViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  evolutionRun: EvolutionaryOptimizationRun | null;
  onUpdateEvolutionRun: (run: EvolutionaryOptimizationRun) => void;
  onNavigate: (view: string) => void;
}

export const OptimizationView: React.FC<OptimizationViewProps> = ({
  dataset,
  models,
  evolutionRun,
  onUpdateEvolutionRun,
  onNavigate,
}) => {
  const [weights, setWeights] = useState<ObjectiveWeights>({
    performance: 0.5,
    latency: 0.25,
    simplicity: 0.25,
    robustness: 0.1,
  });

  const [optimizerEngine, setOptimizerEngine] = useState<'evolutionary' | 'optuna'>('evolutionary');
  const [isEvolving, setIsEvolving] = useState(false);
  const [activeTab, setActiveTab] = useState<'pareto' | 'evolution'>('pareto');

  if (models.length === 0) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <GitFork className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Models to Optimize</h2>
        <p className="text-xs text-slate-400">
          Train models in the AutoML engine before running multi-objective Pareto or evolutionary searches.
        </p>
        <button
          onClick={() => onNavigate('automl')}
          className="px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
        >
          Train Candidate Models
        </button>
      </div>
    );
  }

  // Calculate dynamic weighted ranking based on sliders
  const scoredRanks = calculateWeightedScores(models, weights);
  const topCandidateId = scoredRanks[0]?.modelId;
  const topCandidate = models.find((m) => m.id === topCandidateId) || models[0];

  // Pareto Scatter Data: Performance (%) vs Latency (ms)
  const isClass = dataset?.problemDetection.detectedProblem === 'classification';
  const paretoScatterData = models.map((m) => ({
    id: m.id,
    name: m.name,
    performance: isClass ? Math.round((m.metrics.accuracy || 0) * 1000) / 10 : m.metrics.r2 || 0,
    latency: m.metrics.inferenceLatencyMs,
    complexity: m.metrics.modelComplexityScore,
    isPareto: m.isParetoOptimal,
  }));

  const paretoPoints = paretoScatterData.filter((d) => d.isPareto);
  const dominatedPoints = paretoScatterData.filter((d) => !d.isPareto);

  // Run new evolutionary or Optuna search
  const handleRunEvolution = async () => {
    setIsEvolving(true);
    try {
      let run: EvolutionaryOptimizationRun;
      if (optimizerEngine === 'optuna') {
        run = await api.runOptunaOptimization(dataset?.id || 'demo', 10);
      } else {
        run = await api.runOptimization(dataset?.id || 'demo', 8);
      }
      onUpdateEvolutionRun(run);
    } catch (err) {
      console.error('Optimization error:', err);
    } finally {
      setIsEvolving(false);
    }
  };

  const evoGenerations = evolutionRun?.generations || [];

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Pareto Frontiers & Genetic Algorithms
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Multi-Objective Optimization
          </h1>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-1 p-1 bg-slate-900 border border-slate-800 rounded-lg self-start">
          <button
            onClick={() => setActiveTab('pareto')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              activeTab === 'pareto'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Pareto Frontier
          </button>
          <button
            onClick={() => setActiveTab('evolution')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              activeTab === 'evolution'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Evolutionary Search
          </button>
        </div>
      </div>

      {activeTab === 'pareto' ? (
        <div className="space-y-6">
          {/* Section 13: Concept Callout */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs text-slate-300 flex items-start gap-3">
            <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <strong className="text-white font-semibold">Pareto-Optimality Invariant:</strong>
              <p className="text-slate-400 leading-relaxed">
                A model is Pareto-optimal when no objective (performance, inference latency, or simplicity) can be improved without worsening another objective. The highlighted points represent non-dominated candidates across the decision boundary.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Pareto Chart */}
            <div className="lg:col-span-2 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    Pareto Frontier (Performance vs Latency)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Optimal trade-off frontier. Top-left represents peak Pareto efficiency.
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                    <span className="text-slate-300">Pareto Optimal</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-600" />
                    <span className="text-slate-400">Dominated</span>
                  </div>
                </div>
              </div>

              <div className="h-72 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: -10 }}>
                    <XAxis
                      type="number"
                      dataKey="latency"
                      name="Inference Latency"
                      unit="ms"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      label={{ value: 'Inference Latency (ms)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="performance"
                      name="Performance"
                      unit={isClass ? '%' : ''}
                      domain={['auto', 'auto']}
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      label={{ value: isClass ? 'Accuracy (%)' : 'R²', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                    />
                    <Tooltip
                      content={({ payload }) => {
                        if (payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="p-3 bg-slate-900 border border-slate-700 rounded-lg shadow-xl text-xs space-y-1">
                              <p className="font-bold text-white">{data.name}</p>
                              <p className="text-cyan-400">Performance: {data.performance}{isClass ? '%' : ''}</p>
                              <p className="text-slate-300">Latency: {data.latency} ms</p>
                              <p className="text-slate-400">Complexity: {data.complexity}/10</p>
                              <p className={data.isPareto ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
                                {data.isPareto ? '✓ Pareto-Optimal Solution' : 'Dominated Candidate'}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Scatter name="Pareto Optimal" data={paretoPoints} fill="#22d3ee" />
                    <Scatter name="Dominated" data={dominatedPoints} fill="#475569" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Dynamic Objective Weight Sliders */}
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-white">Objective Weight Tuning</h3>
                <p className="text-xs text-slate-400">
                  Dynamically rank models according to your operational constraints
                </p>
              </div>

              {/* Slider 1: Performance */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Predictive Performance</span>
                  <span className="text-cyan-400 font-mono tabular-nums">
                    {Math.round(weights.performance * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.performance}
                  onChange={(e) =>
                    setWeights({ ...weights, performance: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Slider 2: Speed / Latency */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Inference Latency (Speed)</span>
                  <span className="text-cyan-400 font-mono tabular-nums">
                    {Math.round(weights.latency * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.latency}
                  onChange={(e) =>
                    setWeights({ ...weights, latency: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Slider 3: Simplicity / Complexity */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Simplicity (Low Complexity)</span>
                  <span className="text-cyan-400 font-mono tabular-nums">
                    {Math.round(weights.simplicity * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.simplicity}
                  onChange={(e) =>
                    setWeights({ ...weights, simplicity: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Dynamic Winner Callout */}
              <div className="p-4 rounded-lg bg-slate-950 border border-cyan-500/30 space-y-1.5 mt-4">
                <span className="text-[10px] text-cyan-400 uppercase tracking-wider font-semibold">
                  Preferred Configuration
                </span>
                <h4 className="text-base font-bold text-white">{topCandidate.name}</h4>
                <p className="text-[11px] text-slate-400">
                  Composite Utility: <strong className="text-cyan-300">{scoredRanks[0]?.compositeScore}</strong> · Latency: {topCandidate.metrics.inferenceLatencyMs}ms
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Section 14: Evolutionary Optimization */
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-xl bg-slate-900/60 border border-slate-800">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-sm font-semibold text-white">
                  {optimizerEngine === 'evolutionary' ? 'Genuine Genetic Evolutionary Optimization' : 'Optuna Bayesian TPE Hyperparameter Tuning'}
                </h3>
                <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  REAL ML EVALUATION
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {optimizerEngine === 'evolutionary'
                  ? 'Genuinely fits model candidate chromosomes, evaluating predictive performance, latency, and complexity across generations.'
                  : 'Executes Optuna study sampling hyperparameter configurations, measuring genuine cross-validation metrics and latency.'}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center p-0.5 bg-slate-950 border border-slate-800 rounded-lg text-xs">
                <button
                  onClick={() => setOptimizerEngine('evolutionary')}
                  className={`px-2.5 py-1 rounded transition-colors ${optimizerEngine === 'evolutionary' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'}`}
                >
                  Genetic Evolution
                </button>
                <button
                  onClick={() => setOptimizerEngine('optuna')}
                  className={`px-2.5 py-1 rounded transition-colors ${optimizerEngine === 'optuna' ? 'bg-purple-500/20 text-purple-300 font-semibold' : 'text-slate-400 hover:text-white'}`}
                >
                  Optuna TPE
                </button>
              </div>
              <button
                onClick={handleRunEvolution}
                disabled={isEvolving}
                className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 rounded-lg transition-colors shrink-0"
              >
                <Dna className={`w-4 h-4 ${isEvolving ? 'animate-spin' : ''}`} />
                <span>{isEvolving ? 'Optimizing...' : optimizerEngine === 'evolutionary' ? 'Run Evolutionary Search' : 'Run Optuna Study'}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Generations Convergence Chart */}
            <div className="lg:col-span-2 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-semibold text-white">Fitness Convergence Across Generations</h4>
                  <p className="text-xs text-slate-400">Tracking peak fitness vs population average</p>
                </div>
                {evolutionRun && (
                  <span className="text-xs text-slate-400">
                    Final Fitness: <strong className="text-purple-400 font-mono">{evolutionRun.finalFitness}</strong>
                  </span>
                )}
              </div>

              <div className="h-64 w-full">
                {evoGenerations.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={evoGenerations} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                      <XAxis dataKey="generation" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }} />
                      <Line type="monotone" dataKey="bestFitness" stroke="#a855f7" strokeWidth={2.5} name="Best Fitness" dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="avgFitness" stroke="#64748b" strokeDasharray="3 3" name="Avg Population Fitness" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500">
                    Execute evolutionary run to observe genetic chromosome convergence
                  </div>
                )}
              </div>
            </div>

            {/* Optimal Genome Hyperparameters */}
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
              <h4 className="text-sm font-semibold text-white">
                {optimizerEngine === 'evolutionary' ? 'Optimal Evolved Genome' : 'Best Optuna Hyperparameters'}
              </h4>
              <p className="text-xs text-slate-400">Derived from real validation evaluations across candidate trials</p>

              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs space-y-2">
                {evolutionRun?.optimalHyperparameters ? (
                  Object.entries(evolutionRun.optimalHyperparameters).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-slate-400">{k}:</span>
                      <span className="text-purple-300 font-semibold">{String(v)}</span>
                    </div>
                  ))
                ) : (
                  <span className="text-slate-500 italic">No evolved genome yet</span>
                )}
              </div>

              {evolutionRun && (
                <div className="pt-2 text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Initial Fitness:</span>
                    <span className="text-white font-mono">{evolutionRun.initialFitness}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Final Fitness:</span>
                    <span className="text-purple-400 font-mono font-semibold">{evolutionRun.finalFitness}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Improvement:</span>
                    <span className="text-emerald-400 font-mono">
                      +{((evolutionRun.finalFitness - evolutionRun.initialFitness) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
