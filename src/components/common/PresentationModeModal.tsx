import React, { useState } from 'react';
import {
  X,
  Database,
  Search,
  Cpu,
  GitFork,
  BrainCircuit,
  Radio,
  FileCheck2,
  ChevronRight,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { Dataset, CandidateModel, AnomalySummary, IntelligenceReport } from '../../types';

interface PresentationModeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateToView: (view: string) => void;
  dataset: Dataset | null;
  models: CandidateModel[];
  anomalies: AnomalySummary | null;
  report: IntelligenceReport | null;
}

export const PresentationModeModal: React.FC<PresentationModeModalProps> = ({
  isOpen,
  onClose,
  onNavigateToView,
  dataset,
  models,
  anomalies,
  report,
}) => {
  if (!isOpen) return null;

  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      id: 'step_data',
      title: '01. Raw Data Ingestion',
      category: 'DATA',
      icon: Database,
      targetView: 'dataset',
      description:
        'Uploads raw tabular CSV or XLSX data. The system computes dataset size, column profiles, missing rates, and a 0–100 Data Quality Score with full diagnostic justifications.',
      highlight: dataset ? `${dataset.name} (${dataset.rowCount} rows · Quality: ${dataset.qualityReport.score}/100)` : 'No dataset loaded yet',
    },
    {
      id: 'step_analysis',
      title: '02. Autonomous Problem Detection',
      category: 'AI ANALYSIS',
      icon: Search,
      targetView: 'dataset',
      description:
        'Analyzes target cardinality and distribution to infer whether the task is Classification, Regression, Clustering, or Anomaly Detection, providing confidence scores and mathematical justifications.',
      highlight: dataset ? `Problem: ${dataset.problemDetection.detectedProblem.toUpperCase()} (Confidence: ${dataset.problemDetection.confidence}%)` : 'Ready to detect',
    },
    {
      id: 'step_automl',
      title: '03. AutoML Candidate Training',
      category: 'AUTOML',
      icon: Cpu,
      targetView: 'automl',
      description:
        'Applies missing value imputation, scaling, and categorical encoding. Trains multiple architectures (XGBoost, Random Forest, Extra Trees, LightGBM, Logistic/Ridge) across test splits.',
      highlight: models.length > 0 ? `${models.length} candidate models evaluated across F1, ROC-AUC, latency` : 'Awaiting training',
    },
    {
      id: 'step_optimization',
      title: '04. Multi-Objective Pareto & Evolution',
      category: 'EVOLUTIONARY OPTIMIZATION',
      icon: GitFork,
      targetView: 'optimization',
      description:
        'Computes the non-dominated Pareto frontier across Predictive Performance, Inference Latency, and Model Complexity. Runs genetic evolutionary algorithm across hyperparameter generations.',
      highlight: models.filter((m) => m.isParetoOptimal).length > 0 ? `${models.filter((m) => m.isParetoOptimal).length} Pareto-optimal configurations identified` : 'Interactive trade-off matrix',
    },
    {
      id: 'step_explain',
      title: '05. Explainable AI & Attributions',
      category: 'EXPLAINABLE AI',
      icon: BrainCircuit,
      targetView: 'explain',
      description:
        'Global feature importance via SHAP and permutation degradation. Individual prediction waterfall decompositions show exact mathematical contributions to decision scores.',
      highlight: models[0]?.featureImportances ? `Key driver: ${models[0].featureImportances[0]?.feature}` : 'Transparent decision logic',
    },
    {
      id: 'step_anomalies',
      title: '06. Isolation Forest Anomaly Radar',
      category: 'ANOMALY DETECTION',
      icon: Radio,
      targetView: 'anomalies',
      description:
        'Unsupervised Isolation Forest algorithm constructs isolation trees to project outliers onto a 2D radar manifold, pinpointing high-risk anomalous transactions with feature deviations.',
      highlight: anomalies ? `${anomalies.anomalyCount} anomalies flagged (${anomalies.anomalyRate}% rate)` : 'Outlier detection engine',
    },
    {
      id: 'step_intelligence',
      title: '07. Actionable AI Intelligence',
      category: 'INTELLIGENCE',
      icon: FileCheck2,
      targetView: 'reports',
      description:
        'Synthesizes automated executive intelligence with Pareto deployment winners, data health findings, and tailored prescriptive recommendations. Exportable as PDF or JSON.',
      highlight: report ? `Report compiled with ${report.recommendations.length} strategic recommendations` : 'Final synthesis',
    },
  ];

  const currentStepData = steps[activeStep];
  const StepIcon = currentStepData.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <h2 className="text-base font-semibold text-white">
              EVOX Hackathon Presentation Architecture
            </h2>
            <span className="text-xs text-slate-400">· 7-Stage Intelligence Pipeline</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Pipeline Stepper Navigation */}
        <div className="grid grid-cols-7 border-b border-slate-800 text-[11px] overflow-x-auto bg-slate-950/40">
          {steps.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => setActiveStep(idx)}
              className={`p-2.5 text-center transition-all border-b-2 flex flex-col items-center gap-1 ${
                activeStep === idx
                  ? 'border-cyan-400 bg-cyan-950/30 text-white font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <span className="text-[10px] text-slate-400">0{idx + 1}</span>
              <span className="truncate w-full">{s.category}</span>
            </button>
          ))}
        </div>

        {/* Step Body */}
        <div className="p-8 flex-1 overflow-y-auto space-y-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-lg bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
              <StepIcon className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <span className="text-xs uppercase tracking-wider text-cyan-400 font-medium">
                Stage {activeStep + 1} of 7 · {currentStepData.category}
              </span>
              <h3 className="text-xl font-bold text-white tracking-tight">
                {currentStepData.title}
              </h3>
            </div>
          </div>

          <p className="text-sm text-slate-300 leading-relaxed">
            {currentStepData.description}
          </p>

          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1.5">
            <span className="text-xs text-slate-400 uppercase tracking-wider font-medium">
              Live Session Signal
            </span>
            <p className="text-sm font-semibold text-cyan-300 font-mono">
              {currentStepData.highlight}
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 bg-slate-950/80 border-t border-slate-800 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))}
              disabled={activeStep === 0}
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-white disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={() => setActiveStep((prev) => Math.min(steps.length - 1, prev + 1))}
              disabled={activeStep === steps.length - 1}
              className="px-3 py-1.5 text-xs text-slate-300 hover:text-white disabled:opacity-40"
            >
              Next Step
            </button>
          </div>

          <button
            onClick={() => {
              onClose();
              onNavigateToView(currentStepData.targetView);
            }}
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors"
          >
            <span>Inspect Live {currentStepData.category}</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
