import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  Sliders,
  AlertCircle,
  HelpCircle,
  TrendingUp,
  TrendingDown,
  Info,
  ArrowRight,
} from 'lucide-react';
import { CandidateModel, Dataset, SinglePredictionExplanation } from '../../types';
import { api } from '../../services/api';

interface ExplainViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  onNavigate: (view: string) => void;
}

export const ExplainView: React.FC<ExplainViewProps> = ({ dataset, models, onNavigate }) => {
  const [selectedModelId, setSelectedModelId] = useState<string>(models[0]?.id || '');
  const [sampleFeatures, setSampleFeatures] = useState<Record<string, number>>({
    transaction_amount: 4800,
    transaction_frequency: 8,
    credit_score: 620,
    customer_age: 38,
    annual_income: 54000,
    account_age_months: 18,
  });
  const [explanation, setExplanation] = useState<SinglePredictionExplanation | null>(null);

  const selectedModel = models.find((m) => m.id === selectedModelId) || models[0];

  useEffect(() => {
    if (!selectedModel) return;
    let isMounted = true;
    api.predict(selectedModel, sampleFeatures).then((res) => {
      if (isMounted) setExplanation(res);
    }).catch(console.error);

    return () => { isMounted = false; };
  }, [selectedModelId, sampleFeatures]);

  if (models.length === 0) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <BrainCircuit className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Models Trained</h2>
        <p className="text-xs text-slate-400">
          Train models in the AutoML engine before generating explainable AI attribution breakdowns.
        </p>
        <button
          onClick={() => onNavigate('automl')}
          className="px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
        >
          Train Models
        </button>
      </div>
    );
  }

  const handleSliderChange = (feat: string, val: number) => {
    setSampleFeatures((prev) => ({ ...prev, [feat]: val }));
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Explainable AI & Feature Attributions
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Explain AI</h1>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-400">Inspected Model:</label>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.algorithm})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Epistemological & Legal Notice (Section 15) */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 flex items-start gap-3">
        <Info className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <strong className="text-white font-semibold">Model-Based Association Notice:</strong>
          <p className="text-slate-400 leading-relaxed">
            Attributions quantify how individual feature deviations alter the trained model's decision function relative to background reference distributions. These scores represent statistical associations within the trained model and do not establish direct empirical causality.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Global Feature Importance */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-white">Global Feature Importance (Permutation / SHAP)</h3>
            <p className="text-xs text-slate-400">
              Evaluates overall loss degradation when column distributions are permuted
            </p>
          </div>

          <div className="space-y-4 pt-2">
            {(selectedModel.featureImportances || []).slice(0, 7).map((feat, idx) => (
              <div key={feat.feature} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-200 font-medium">{feat.feature}</span>
                  <span className="text-cyan-400 font-mono tabular-nums">
                    {(feat.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full"
                    style={{ width: `${Math.min(100, feat.importance * 100 * 2.2)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Local Instance Attribution & Waterfall */}
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Local Instance Waterfall Explanation</h3>
              <p className="text-xs text-slate-400">
                Single record score attribution breakdown
              </p>
            </div>
            <span className="text-xs font-bold text-cyan-300 font-mono">
              Output: {explanation ? String(explanation.prediction) : 'Evaluating...'}
            </span>
          </div>

          {/* Interactive Feature Sliders */}
          <div className="grid grid-cols-2 gap-3 p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            {Object.entries(sampleFeatures).map(([feat, val]) => (
              <div key={feat} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400 truncate">{feat}</span>
                  <span className="text-slate-200 font-mono tabular-nums">{val}</span>
                </div>
                <input
                  type="range"
                  min={feat.includes('amount') ? 100 : feat.includes('score') ? 450 : 1}
                  max={feat.includes('amount') ? 20000 : feat.includes('score') ? 850 : 100}
                  value={val}
                  onChange={(e) => handleSliderChange(feat, parseInt(e.target.value))}
                  className="w-full accent-cyan-400 bg-slate-800 h-1 rounded appearance-none cursor-pointer"
                />
              </div>
            ))}
          </div>

          {/* Contributing Factors Waterfall Breakdown */}
          <div className="space-y-2 pt-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
              Top Contributing Factors (SHAP Force Values)
            </span>
            <div className="space-y-2">
              {explanation ? (
                explanation.contributingFactors.slice(0, 5).map((f) => (
                  <div
                    key={f.feature}
                    className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      {f.impact === 'increases_score' ? (
                        <TrendingUp className="w-4 h-4 text-rose-400 shrink-0" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-emerald-400 shrink-0" />
                      )}
                      <span className="text-slate-200 font-medium">{f.feature}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-slate-400 font-mono text-[11px]">val: {f.value}</span>
                      <span
                        className={`font-mono font-semibold tabular-nums ${
                          f.impact === 'increases_score' ? 'text-rose-400' : 'text-emerald-400'
                        }`}
                      >
                        {f.impact === 'increases_score' ? '+' : '-'}
                        {(f.attribution * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-4 text-center text-xs text-slate-500">Computing SHAP attributions...</div>
              )}
            </div>
          </div>

          {/* Natural Language Synthesis */}
          <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase tracking-wider text-cyan-400 font-semibold">
              Human-Readable Synthesis
            </span>
            <p className="text-xs text-slate-300 leading-relaxed">
              {explanation?.humanExplanation || 'Generating model association analysis...'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
