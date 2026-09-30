import React, { useState } from 'react';
import { Sliders, Rocket, CheckCircle2, Clock, Sparkles, AlertCircle } from 'lucide-react';
import { CandidateModel, Dataset, SinglePredictionExplanation } from '../../types';
import { api } from '../../services/api';

interface PredictionStudioViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  selectedModel: CandidateModel | null;
  onNavigate: (view: string) => void;
}

export const PredictionStudioView: React.FC<PredictionStudioViewProps> = ({
  dataset,
  models,
  selectedModel,
  onNavigate,
}) => {
  const activeModel = selectedModel || models[0];

  const initialFormValues = dataset?.features.reduce((acc, feat) => {
    if (feat.includes('amount')) acc[feat] = 1850;
    else if (feat.includes('income')) acc[feat] = 68000;
    else if (feat.includes('score')) acc[feat] = 710;
    else if (feat.includes('age')) acc[feat] = 42;
    else if (feat.includes('frequency')) acc[feat] = 18;
    else acc[feat] = 25;
    return acc;
  }, {} as Record<string, any>) || {
    transaction_amount: 1850,
    annual_income: 68000,
    credit_score: 710,
    customer_age: 42,
    transaction_frequency: 18,
    account_age_months: 24,
  };

  const [formValues, setFormValues] = useState<Record<string, any>>(initialFormValues);
  const [predictionResult, setPredictionResult] = useState<SinglePredictionExplanation | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  if (!activeModel) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <Sliders className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Model Selected</h2>
        <p className="text-xs text-slate-400">
          Train models in AutoML and select one from the Model Arena to run real-time inference.
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

  const handleInputChange = (field: string, value: string) => {
    const num = Number(value);
    setFormValues((prev) => ({
      ...prev,
      [field]: isNaN(num) ? value : num,
    }));
  };

  const handleRunPrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    try {
      const result = await api.predict(activeModel, formValues);
      setPredictionResult(result);
    } catch (err) {
      console.error(err);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Inference & Decision Engine
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Prediction Studio</h1>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>Active Inference Model:</span>
          <span className="font-semibold text-white bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-md">
            {activeModel.name} ({activeModel.metrics.inferenceLatencyMs}ms latency)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Dynamic Input Form */}
        <div className="lg:col-span-7 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Feature Input Parameters</h3>
              <p className="text-xs text-slate-400">
                Generated dynamically from dataset feature schema
              </p>
            </div>
            <button
              type="button"
              onClick={() => setFormValues(initialFormValues)}
              className="text-xs text-slate-400 hover:text-white"
            >
              Reset to Defaults
            </button>
          </div>

          <form onSubmit={handleRunPrediction} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Object.keys(formValues).map((field) => {
                const val = formValues[field];
                return (
                  <div key={field} className="space-y-1">
                    <label className="text-xs font-medium text-slate-300 capitalize truncate block">
                      {field.replace(/_/g, ' ')}
                    </label>
                    <input
                      type="text"
                      value={val !== undefined ? val : ''}
                      onChange={(e) => handleInputChange(field, e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-400 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none transition-colors"
                      required
                    />
                  </div>
                );
              })}
            </div>

            <div className="pt-4 border-t border-slate-800">
              <button
                type="submit"
                disabled={isPredicting}
                className="w-full flex items-center justify-center gap-2 py-3 px-4 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 rounded-lg shadow-md transition-all"
              >
                <Rocket className={`w-4 h-4 ${isPredicting ? 'animate-spin' : ''}`} />
                <span>{isPredicting ? 'Executing Neural Inference...' : 'RUN PREDICTION'}</span>
              </button>
            </div>
          </form>
        </div>

        {/* Inference Results Output */}
        <div className="lg:col-span-5 p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-6 flex flex-col justify-between">
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-white">Inference Outcome</h3>

            {predictionResult ? (
              <div className="space-y-5">
                {/* Primary Prediction Output */}
                <div className="p-5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <span className="text-[11px] text-slate-400 uppercase tracking-wider font-medium">
                    Predicted Class / Value
                  </span>
                  <div className="flex items-baseline justify-between">
                    <span className="text-2xl font-bold text-white tracking-tight">
                      {String(predictionResult.prediction)}
                    </span>
                    {predictionResult.probability !== undefined && (
                      <span className="text-xs font-bold text-cyan-400 tabular-nums">
                        {(predictionResult.probability * 100).toFixed(1)}% Confidence
                      </span>
                    )}
                  </div>
                </div>

                {/* Class Probabilities Distribution */}
                {predictionResult.classProbabilities && (
                  <div className="space-y-2">
                    <span className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
                      Probability Distribution
                    </span>
                    <div className="space-y-2">
                      {predictionResult.classProbabilities.map((cp) => (
                        <div key={cp.label} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-slate-300">{cp.label}</span>
                            <span className="text-slate-400 font-mono tabular-nums">
                              {(cp.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 rounded-full bg-slate-950 overflow-hidden">
                            <div
                              className="h-full bg-cyan-400 rounded-full"
                              style={{ width: `${cp.prob * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Association Explanation Note */}
                <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                    Attribution Summary
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {predictionResult.humanExplanation}
                  </p>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center space-y-3">
                <Rocket className="w-10 h-10 text-slate-600 mx-auto" />
                <p className="text-xs text-slate-400 max-w-xs mx-auto">
                  Adjust the input parameters and click "RUN PREDICTION" to compute live inference with statistical feature attributions.
                </p>
              </div>
            )}
          </div>

          {predictionResult && (
            <div className="pt-4 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Latency: {activeModel.metrics.inferenceLatencyMs} ms</span>
              <span className="tabular-nums">
                Timestamp: {new Date(predictionResult.timestamp).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
