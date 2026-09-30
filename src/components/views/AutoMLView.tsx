import React, { useState } from 'react';
import {
  Cpu,
  Play,
  CheckCircle2,
  Clock,
  ArrowRight,
  Database,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { Dataset, CandidateModel } from '../../types';
import { api } from '../../services/api';

interface AutoMLViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  onModelsTrained: (models: CandidateModel[]) => void;
  onNavigate: (view: string) => void;
}

export const AutoMLView: React.FC<AutoMLViewProps> = ({
  dataset,
  models,
  onModelsTrained,
  onNavigate,
}) => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentStepText, setCurrentStepText] = useState('');
  const [activeTab, setActiveTab] = useState<'pipeline' | 'models'>('models');

  if (!dataset) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <Database className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Dataset Available</h2>
        <p className="text-xs text-slate-400">
          Load or upload a dataset in the Dataset Lab before executing AutoML pipelines.
        </p>
        <button
          onClick={() => onNavigate('dataset')}
          className="px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
        >
          Open Dataset Lab
        </button>
      </div>
    );
  }

  const handleStartAutoML = async () => {
    setIsTraining(true);
    setTrainingProgress(2);
    setCurrentStepText('Connecting to Python training engine...');

    try {
      const trained = await api.trainModels(
        dataset,
        dataset.problemDetection.detectedProblem,
        dataset.targetColumn || 'target',
        (pct, step) => {
          setTrainingProgress(pct);
          setCurrentStepText(step);
        }
      );
      setTrainingProgress(100);
      setCurrentStepText('AutoML candidate evaluation complete.');
      onModelsTrained(trained);
    } catch (err: any) {
      console.error(err);
      setCurrentStepText(`Training error: ${err.message || 'Check Python logs'}`);
    } finally {
      setTimeout(() => setIsTraining(false), 600);
    }
  };

  const isClassification = dataset.problemDetection.detectedProblem === 'classification';

  const preprocessingStages = [
    { name: 'RAW DATA', desc: `${dataset.rowCount} rows · ${dataset.columnCount} features` },
    { name: 'CLEANING', desc: 'Median & mode missing imputation' },
    { name: 'ENCODING', desc: 'One-hot & ordinal categorical mapping' },
    { name: 'FEATURE ENGINEERING', desc: 'StandardScaler & outlier trimming' },
    { name: 'OPTIMIZED DATASET', desc: '80% train / 20% holdout test partition' },
  ];

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Automated Model Synthesis
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">AutoML Engine</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleStartAutoML}
            disabled={isTraining}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 rounded-lg shadow-sm transition-all"
          >
            <Play className={`w-3.5 h-3.5 fill-current ${isTraining ? 'animate-spin' : ''}`} />
            <span>{isTraining ? 'Training Pipeline...' : 'Run AutoML Pipeline'}</span>
          </button>
        </div>
      </div>

      {/* Preprocessing Visual Pipeline Flow (Section 10) */}
      <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Automated Preprocessing Flow</h3>
          <span className="text-xs text-slate-400">Deterministic pipeline execution</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 pt-2">
          {preprocessingStages.map((stg, i) => (
            <div
              key={stg.name}
              className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1 relative"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span>0{i + 1}</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <h4 className="text-xs font-semibold text-white">{stg.name}</h4>
              <p className="text-[11px] text-slate-400">{stg.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Live Training Status Card (Section 19) */}
      {isTraining && (
        <div className="p-6 rounded-xl bg-slate-900/90 border border-cyan-500/30 space-y-4 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400 animate-bounce" />
              <h3 className="text-sm font-semibold text-white">AutoML Execution in Progress</h3>
            </div>
            <span className="text-xs font-mono text-cyan-300 tabular-nums">
              {trainingProgress}%
            </span>
          </div>

          <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-cyan-400 transition-all duration-300 rounded-full"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>

          <p className="text-xs text-slate-300 font-mono">
            Status: {currentStepText}
          </p>
        </div>
      )}

      {/* Evaluated Candidates Grid / Table */}
      <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Evaluated Model Architectures ({models.length} Trained)
            </h3>
            <p className="text-xs text-slate-400">
              Validated on holdout test set with latency profiling
            </p>
          </div>
          {models.length > 0 && (
            <button
              onClick={() => onNavigate('models')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-cyan-400 hover:text-cyan-300"
            >
              <span>Inspect in Model Arena</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {models.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <Cpu className="w-10 h-10 text-slate-500 mx-auto" />
            <p className="text-xs text-slate-400">
              No models evaluated yet. Click "Run AutoML Pipeline" to train candidate algorithms.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-3 px-3">Model</th>
                  <th className="py-3 px-3">Algorithm Family</th>
                  {isClassification ? (
                    <>
                      <th className="py-3 px-3">Accuracy</th>
                      <th className="py-3 px-3">Precision</th>
                      <th className="py-3 px-3">Recall</th>
                      <th className="py-3 px-3">F1-Score</th>
                    </>
                  ) : (
                    <>
                      <th className="py-3 px-3">R² Score</th>
                      <th className="py-3 px-3">RMSE</th>
                      <th className="py-3 px-3">MAE</th>
                    </>
                  )}
                  <th className="py-3 px-3">Latency</th>
                  <th className="py-3 px-3">Train Time</th>
                  <th className="py-3 px-3">Pareto Optimal</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {models.map((m) => (
                  <tr key={m.id} className="hover:bg-slate-800/30">
                    <td className="py-3 px-3 font-semibold text-white">
                      {m.name}
                    </td>
                    <td className="py-3 px-3 text-slate-400">{m.algorithm}</td>
                    {isClassification ? (
                      <>
                        <td className="py-3 px-3 font-mono font-medium text-cyan-300 tabular-nums">
                          {(m.metrics.accuracy! * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-3 font-mono tabular-nums">
                          {(m.metrics.precision! * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-3 font-mono tabular-nums">
                          {(m.metrics.recall! * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-3 font-mono tabular-nums">
                          {(m.metrics.f1! * 100).toFixed(1)}%
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="py-3 px-3 font-mono font-medium text-cyan-300 tabular-nums">
                          {m.metrics.r2?.toFixed(3)}
                        </td>
                        <td className="py-3 px-3 font-mono tabular-nums">
                          {m.metrics.rmse}
                        </td>
                        <td className="py-3 px-3 font-mono tabular-nums">
                          {m.metrics.mae}
                        </td>
                      </>
                    )}
                    <td className="py-3 px-3 font-mono tabular-nums text-slate-300">
                      {m.metrics.inferenceLatencyMs} ms
                    </td>
                    <td className="py-3 px-3 font-mono tabular-nums text-slate-400">
                      {m.metrics.trainingTimeMs} ms
                    </td>
                    <td className="py-3 px-3">
                      {m.isParetoOptimal ? (
                        <span className="text-emerald-400 font-semibold text-[11px]">
                          Yes · Optimal
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[11px]">Dominated</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
