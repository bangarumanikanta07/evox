import React from 'react';
import {
  Database,
  Trophy,
  Zap,
  Radio,
  GitFork,
  ArrowRight,
  TrendingUp,
  Cpu,
  Layers,
  Sparkles,
} from 'lucide-react';
import { Dataset, CandidateModel, AnomalySummary, EvolutionaryOptimizationRun } from '../../types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';

interface OverviewViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  anomalies: AnomalySummary | null;
  evolutionRun: EvolutionaryOptimizationRun | null;
  onNavigate: (view: string) => void;
  onLaunchDemo: () => void;
  isDemoLoading: boolean;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  dataset,
  models,
  anomalies,
  evolutionRun,
  onNavigate,
  onLaunchDemo,
  isDemoLoading,
}) => {
  if (!dataset) {
    return (
      <div className="p-8 max-w-4xl mx-auto text-center space-y-6 my-16">
        <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-400">
          <Database className="w-8 h-8" />
        </div>
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">No Dataset Loaded</h2>
          <p className="text-sm text-slate-400 max-w-md mx-auto">
            Upload your CSV/XLSX file in Dataset Lab or run the Customer Intelligence demo pipeline to start autonomous ML training.
          </p>
        </div>
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => onNavigate('dataset')}
            className="px-4 py-2.5 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
          >
            Go to Dataset Lab
          </button>
          <button
            onClick={onLaunchDemo}
            disabled={isDemoLoading}
            className="px-4 py-2.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
          >
            {isDemoLoading ? 'Running Demo...' : 'Launch Demo Dataset'}
          </button>
        </div>
      </div>
    );
  }

  const bestModel = [...models].sort((a, b) => {
    const scoreA = a.problemType === 'classification' ? a.metrics.accuracy || 0 : a.metrics.r2 || 0;
    const scoreB = b.problemType === 'classification' ? b.metrics.accuracy || 0 : b.metrics.r2 || 0;
    return scoreB - scoreA;
  })[0];

  const fastestModel = [...models].sort((a, b) => a.metrics.inferenceLatencyMs - b.metrics.inferenceLatencyMs)[0];
  const paretoCount = models.filter((m) => m.isParetoOptimal).length;

  const isClass = dataset.problemDetection.detectedProblem === 'classification';
  const primaryMetricLabel = isClass ? 'Accuracy' : 'R² Score';
  const primaryMetricVal = isClass
    ? bestModel?.metrics.accuracy ? `${(bestModel.metrics.accuracy * 100).toFixed(1)}%` : '—'
    : bestModel?.metrics.r2 ? bestModel.metrics.r2.toFixed(3) : '—';

  // Chart data for model comparison
  const modelChartData = models.map((m) => ({
    name: m.name.replace(' Classifier', '').replace(' Regressor', ''),
    score: isClass ? Math.round((m.metrics.accuracy || 0) * 1000) / 10 : m.metrics.r2 || 0,
    latency: m.metrics.inferenceLatencyMs,
    isPareto: m.isParetoOptimal,
  }));

  // Evolution chart data
  const evoChartData = (evolutionRun?.generations || []).map((g) => ({
    generation: `G${g.generation}`,
    bestFitness: Math.round(g.bestFitness * 1000) / 1000,
    avgFitness: Math.round(g.avgFitness * 1000) / 1000,
  }));

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Top Banner / System Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Laboratory Command Center
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            System Intelligence Overview
          </h1>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span>Active Dataset: <strong className="text-slate-200">{dataset.name}</strong></span>
          <span aria-hidden="true">·</span>
          <span>Task: <strong className="text-slate-200 uppercase">{dataset.problemDetection.detectedProblem}</strong></span>
        </div>
      </div>

      {/* KPI Metric Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Dataset Status */}
        <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Dataset Health</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white tabular-nums">
              {dataset.qualityReport.score}
            </span>
            <span className="text-xs text-slate-400">/ 100</span>
          </div>
          <p className="text-[11px] text-slate-400 truncate">
            {dataset.rowCount.toLocaleString()} rows · {dataset.columnCount} features
          </p>
        </div>

        {/* Card 2: Best Model */}
        <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Best Model</span>
            <Trophy className="w-4 h-4 text-amber-400" />
          </div>
          <div className="truncate">
            <span className="text-xl font-bold text-white truncate block">
              {bestModel ? bestModel.name : 'Training Pending'}
            </span>
          </div>
          <p className="text-[11px] text-slate-400">
            {primaryMetricLabel}: <strong className="text-cyan-400 tabular-nums">{primaryMetricVal}</strong>
          </p>
        </div>

        {/* Card 3: Inference Latency */}
        <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Inference Latency</span>
            <Zap className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white tabular-nums">
              {fastestModel ? fastestModel.metrics.inferenceLatencyMs : '0'}
            </span>
            <span className="text-xs text-slate-400">ms / sample</span>
          </div>
          <p className="text-[11px] text-slate-400 truncate">
            Fastest: {fastestModel ? fastestModel.name : 'None'}
          </p>
        </div>

        {/* Card 4: Anomalies */}
        <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Anomaly Radar</span>
            <Radio className="w-4 h-4 text-rose-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white tabular-nums">
              {anomalies ? anomalies.anomalyCount : 0}
            </span>
            <span className="text-xs text-slate-400">
              ({anomalies ? anomalies.anomalyRate : 0}%)
            </span>
          </div>
          <p className="text-[11px] text-slate-400 truncate">
            Isolation Forest threshold: {anomalies ? anomalies.threshold : '-0.4'}
          </p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Comparison Chart */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">AutoML Performance Benchmark</h3>
              <p className="text-xs text-slate-400">
                Evaluation across candidate architectures ({primaryMetricLabel})
              </p>
            </div>
            <button
              onClick={() => onNavigate('models')}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
            >
              <span>Model Arena</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64 w-full">
            {modelChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    angle={-25}
                    textAnchor="end"
                  />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }}
                    itemStyle={{ color: '#38bdf8' }}
                  />
                  <Bar dataKey="score" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No candidate models trained yet
              </div>
            )}
          </div>
        </div>

        {/* Evolutionary Optimization Progress */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Genetic Evolution Convergence</h3>
              <p className="text-xs text-slate-400">
                Hyperparameter fitness progression across generations
              </p>
            </div>
            <button
              onClick={() => onNavigate('optimization')}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
            >
              <span>Pareto Studio</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64 w-full">
            {evoChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={evoChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <XAxis dataKey="generation" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }}
                  />
                  <Line type="monotone" dataKey="bestFitness" stroke="#a855f7" strokeWidth={2.5} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="avgFitness" stroke="#64748b" strokeDasharray="3 3" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                Evolutionary optimization run ready to execute
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature Importance & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Global Feature Importance (Permutation Attribution)</h3>
              <p className="text-xs text-slate-400">Primary decision drivers for {bestModel?.name || 'Selected Model'}</p>
            </div>
            <button
              onClick={() => onNavigate('explain')}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
            >
              <span>Explain AI</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-3 pt-2">
            {(bestModel?.featureImportances?.slice(0, 5) || []).map((f) => (
              <div key={f.feature} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">{f.feature}</span>
                  <span className="text-slate-400 tabular-nums">{(f.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-indigo-500"
                    style={{ width: `${Math.min(100, f.importance * 100 * 2.5)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Next Recommended Workflow */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4 flex flex-col justify-between">
          <div className="space-y-2">
            <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
              Autonomous Recommendation
            </span>
            <h3 className="text-sm font-semibold text-white">Recommended Next Step</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {paretoCount > 0
                ? `${paretoCount} Pareto-optimal configurations have been computed. Inspect trade-offs in the Multi-Objective Studio or generate predictions.`
                : 'AutoML candidate training is available. Run multi-objective optimization to evaluate latency vs performance tradeoffs.'}
            </p>
          </div>

          <div className="space-y-2 pt-4">
            <button
              onClick={() => onNavigate(paretoCount > 0 ? 'optimization' : 'automl')}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
            >
              <span>{paretoCount > 0 ? 'Explore Pareto Trade-offs' : 'Train AutoML Pipeline'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('reports')}
              className="w-full py-2 px-4 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
            >
              View Intelligence Report &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
