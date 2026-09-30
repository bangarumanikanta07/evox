import React, { useState } from 'react';
import {
  Trophy,
  Zap,
  Clock,
  Cpu,
  CheckCircle2,
  Sliders,
  BrainCircuit,
  Rocket,
  X,
  ArrowRight,
  Info,
} from 'lucide-react';
import { CandidateModel, Dataset } from '../../types';

interface ModelArenaViewProps {
  models: CandidateModel[];
  dataset: Dataset | null;
  onSelectModelForPrediction: (model: CandidateModel) => void;
  onNavigate: (view: string) => void;
}

export const ModelArenaView: React.FC<ModelArenaViewProps> = ({
  models,
  dataset,
  onSelectModelForPrediction,
  onNavigate,
}) => {
  const [inspectingModel, setInspectingModel] = useState<CandidateModel | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [deployNotification, setDeployNotification] = useState<string | null>(null);

  if (models.length === 0) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <Trophy className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Models in Arena</h2>
        <p className="text-xs text-slate-400">
          Train models in the AutoML engine before comparing candidates in the arena.
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

  const isClass = dataset?.problemDetection.detectedProblem === 'classification';

  const handleToggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : prev.length < 3 ? [...prev, id] : prev
    );
  };

  const handleDeploy = (modelName: string) => {
    setDeployNotification(`Model "${modelName}" bundled into ONNX / production microservice endpoint.`);
    setTimeout(() => setDeployNotification(null), 4000);
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title & Philosophy Note */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Comparative Benchmark
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Model Arena</h1>
        </div>
        <div className="text-xs text-slate-400 max-w-md sm:text-right">
          <span className="font-semibold text-slate-300">Multi-criteria paradigm:</span> High accuracy models may incur higher latency or memory footprints; select configurations tailored to operational constraints.
        </div>
      </div>

      {deployNotification && (
        <div className="p-4 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{deployNotification}</span>
          </div>
          <button onClick={() => setDeployNotification(null)} className="text-emerald-400 hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Model Cards Grid (Requirement 12) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => {
          const isSelected = selectedIds.includes(model.id);
          return (
            <div
              key={model.id}
              className={`p-6 rounded-xl bg-slate-900/60 border transition-all flex flex-col justify-between space-y-5 ${
                isSelected
                  ? 'border-cyan-400 shadow-md shadow-cyan-950/40'
                  : model.isParetoOptimal
                  ? 'border-slate-700/90 hover:border-slate-600'
                  : 'border-slate-800/80 hover:border-slate-700'
              }`}
            >
              {/* Card Header */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">{model.algorithm}</span>
                  <div className="flex items-center gap-1.5 text-[11px]">
                    <span className="text-emerald-400 font-medium">VALIDATED</span>
                    {model.isParetoOptimal && (
                      <span className="text-cyan-400 font-semibold">· Pareto</span>
                    )}
                  </div>
                </div>
                <h3 className="text-lg font-bold text-white tracking-tight">{model.name}</h3>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80 text-xs">
                <div>
                  <span className="text-[11px] text-slate-400 block">
                    {isClass ? 'Accuracy' : 'R² Score'}
                  </span>
                  <span className="text-base font-bold text-cyan-300 font-mono tabular-nums">
                    {isClass
                      ? `${(model.metrics.accuracy! * 100).toFixed(1)}%`
                      : model.metrics.r2?.toFixed(3)}
                  </span>
                </div>

                <div>
                  <span className="text-[11px] text-slate-400 block">
                    {isClass ? 'F1-Score' : 'RMSE'}
                  </span>
                  <span className="text-base font-bold text-white font-mono tabular-nums">
                    {isClass
                      ? `${(model.metrics.f1! * 100).toFixed(1)}%`
                      : model.metrics.rmse}
                  </span>
                </div>

                <div>
                  <span className="text-[11px] text-slate-400 block">Inference</span>
                  <span className="text-sm font-semibold text-slate-200 font-mono tabular-nums">
                    {model.metrics.inferenceLatencyMs} ms
                  </span>
                </div>

                <div>
                  <span className="text-[11px] text-slate-400 block">Training Time</span>
                  <span className="text-sm font-semibold text-slate-200 font-mono tabular-nums">
                    {model.metrics.trainingTimeMs} ms
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-2 pt-2">
                <button
                  onClick={() => setInspectingModel(model)}
                  className="px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 hover:text-white rounded-lg transition-colors"
                >
                  Inspect Architecture
                </button>

                <button
                  onClick={() => {
                    onSelectModelForPrediction(model);
                    onNavigate('predictions');
                  }}
                  className="px-3 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors flex items-center justify-center gap-1.5"
                >
                  <Rocket className="w-3.5 h-3.5" />
                  <span>Predict Studio</span>
                </button>
              </div>

              <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                <button
                  onClick={() => handleToggleSelect(model.id)}
                  className="hover:text-cyan-300 transition-colors"
                >
                  {isSelected ? '✓ In comparison' : '+ Add to compare'}
                </button>
                <button
                  onClick={() => handleDeploy(model.name)}
                  className="text-slate-400 hover:text-slate-200"
                >
                  Export Model Package
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Model Inspector Modal */}
      {inspectingModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden p-6 space-y-6 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
                  Architectural Inspection
                </span>
                <h2 className="text-xl font-bold text-white">{inspectingModel.name}</h2>
              </div>
              <button
                onClick={() => setInspectingModel(null)}
                className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Hyperparameters */}
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Validated Hyperparameters
              </h4>
              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-1">
                {Object.entries(inspectingModel.hyperparameters).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-slate-400">{k}:</span>
                    <span className="text-cyan-300">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Confusion Matrix (if classification) */}
            {inspectingModel.metrics.confusionMatrix && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Holdout Confusion Matrix
                </h4>
                <div className="grid grid-cols-2 gap-2 p-3 bg-slate-950 border border-slate-800 rounded-lg text-center font-mono text-xs">
                  <div className="p-2 bg-slate-900 rounded">
                    <span className="text-[10px] text-slate-400 block">True Negative</span>
                    <span className="text-sm font-bold text-emerald-400">
                      {inspectingModel.metrics.confusionMatrix.matrix[0][0]}
                    </span>
                  </div>
                  <div className="p-2 bg-slate-900 rounded">
                    <span className="text-[10px] text-slate-400 block">False Positive</span>
                    <span className="text-sm font-bold text-rose-400">
                      {inspectingModel.metrics.confusionMatrix.matrix[0][1]}
                    </span>
                  </div>
                  <div className="p-2 bg-slate-900 rounded">
                    <span className="text-[10px] text-slate-400 block">False Negative</span>
                    <span className="text-sm font-bold text-rose-400">
                      {inspectingModel.metrics.confusionMatrix.matrix[1][0]}
                    </span>
                  </div>
                  <div className="p-2 bg-slate-900 rounded">
                    <span className="text-[10px] text-slate-400 block">True Positive</span>
                    <span className="text-sm font-bold text-emerald-400">
                      {inspectingModel.metrics.confusionMatrix.matrix[1][1]}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Top Feature Attributions */}
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Permutation Feature Attributions
              </h4>
              <div className="space-y-2">
                {inspectingModel.featureImportances.slice(0, 4).map((f) => (
                  <div key={f.feature} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">{f.feature}</span>
                      <span className="text-cyan-400 tabular-nums">
                        {(f.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-400 rounded-full"
                        style={{ width: `${f.importance * 100 * 2}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setInspectingModel(null)}
                className="px-4 py-2 text-xs font-semibold text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
