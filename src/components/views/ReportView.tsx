import React, { useState } from 'react';
import {
  FileText,
  Download,
  Printer,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Trophy,
  GitFork,
  Radio,
  BrainCircuit,
  ArrowRight,
} from 'lucide-react';
import {
  Dataset,
  CandidateModel,
  AnomalySummary,
  IntelligenceReport,
  EvolutionaryOptimizationRun,
} from '../../types';
import { api } from '../../services/api';

interface ReportViewProps {
  dataset: Dataset | null;
  models: CandidateModel[];
  anomalies: AnomalySummary | null;
  report: IntelligenceReport | null;
  onUpdateReport: (report: IntelligenceReport) => void;
  onNavigate: (view: string) => void;
}

export const ReportView: React.FC<ReportViewProps> = ({
  dataset,
  models,
  anomalies,
  report,
  onUpdateReport,
  onNavigate,
}) => {
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  if (!dataset) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center space-y-4 my-16">
        <FileText className="w-12 h-12 text-slate-500 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Experiment Data</h2>
        <p className="text-xs text-slate-400">
          Load or train models to generate the final autonomous AI intelligence report.
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

  const handleGenerateAIReport = async () => {
    setIsSynthesizing(true);
    try {
      const generated = await api.generateReport(dataset, models, anomalies || ({} as any));
      onUpdateReport(generated);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSynthesizing(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EVOX_Intelligence_Report_${report.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadCSV = () => {
    if (!models || models.length === 0) return;
    const isClass = dataset.problemDetection.detectedProblem === 'classification';
    let csv = `Model Name,Algorithm,${isClass ? 'Accuracy,F1-Score' : 'R2,RMSE'},Inference Latency (ms),Training Time (ms),Pareto Optimal\n`;
    models.forEach((m) => {
      const perf1 = isClass ? m.metrics.accuracy : m.metrics.r2;
      const perf2 = isClass ? m.metrics.f1 : m.metrics.rmse;
      csv += `"${m.name}","${m.algorithm}",${perf1},${perf2},${m.metrics.inferenceLatencyMs},${m.metrics.trainingTimeMs},${m.isParetoOptimal}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EVOX_Model_Arena_Metrics.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  const bestModel = [...models].sort((a, b) => {
    const scoreA = a.problemType === 'classification' ? a.metrics.accuracy || 0 : a.metrics.r2 || 0;
    const scoreB = b.problemType === 'classification' ? b.metrics.accuracy || 0 : b.metrics.r2 || 0;
    return scoreB - scoreA;
  })[0];

  const paretoModels = models.filter((m) => m.isParetoOptimal);
  const isClass = dataset.problemDetection.detectedProblem === 'classification';

  return (
    <div className="p-6 lg:p-8 max-w-5xl mx-auto space-y-8 print:p-0 print:max-w-none">
      {/* Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800 print:hidden">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Synthesis & Strategic Insights
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Intelligence Report</h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleGenerateAIReport}
            disabled={isSynthesizing}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 rounded-lg transition-colors"
          >
            <Sparkles className={`w-3.5 h-3.5 ${isSynthesizing ? 'animate-spin' : ''}`} />
            <span>{isSynthesizing ? 'Synthesizing AI Insights...' : 'Re-synthesize AI Report'}</span>
          </button>

          <button
            onClick={handleDownloadJSON}
            disabled={!report}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-800 hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-40"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download JSON</span>
          </button>

          <button
            onClick={handleDownloadCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-800 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>

          <button
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-900 border border-slate-800 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* Main Report Document Container */}
      <div className="p-8 lg:p-12 rounded-xl bg-slate-900/80 border border-slate-800 space-y-8 text-slate-200 print:bg-white print:text-black print:border-none print:shadow-none">
        {/* Document Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 print:border-slate-300 pb-6 gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 print:bg-black"></span>
              <span className="text-xl font-bold tracking-tight text-white print:text-black">
                EVOX — Executive Intelligence Dossier
              </span>
            </div>
            <p className="text-xs text-slate-400 print:text-slate-600">
              Autonomous machine learning synthesis & multi-objective deployment roadmap
            </p>
          </div>

          <div className="text-xs text-slate-400 print:text-slate-600 sm:text-right font-mono">
            <p>Report ID: {report?.id || 'REP_8820'}</p>
            <p>Compiled: {new Date(report?.generatedAt || Date.now()).toLocaleDateString()}</p>
          </div>
        </div>

        {/* Section 1: Executive AI Narrative */}
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-wider text-cyan-400 print:text-blue-700 font-bold">
            01. Executive AI Research Narrative
          </h3>
          <div className="p-4 rounded-lg bg-slate-950/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 text-sm leading-relaxed text-slate-200 print:text-slate-900">
            {report?.aiNarrative ||
              `The EVOX automated intelligence pipeline successfully completed end-to-end characterization across ${models.length} candidate architectures on the ${dataset.name} dataset. Peak empirical accuracy was attained by ${bestModel?.name || 'XGBoost'}, while non-dominated Pareto analysis resolved optimal trade-offs for latency-critical production environments.`}
          </div>
        </div>

        {/* Section 2: Dataset Profile & Data Quality Summary */}
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-wider text-cyan-400 print:text-blue-700 font-bold">
            02. Data Ingestion & Quality Findings
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200">
              <span className="text-slate-400 print:text-slate-500 block">Dataset Name</span>
              <span className="font-bold text-white print:text-black truncate block">{dataset.name}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200">
              <span className="text-slate-400 print:text-slate-500 block">Scale</span>
              <span className="font-bold text-white print:text-black tabular-nums">
                {dataset.rowCount.toLocaleString()} rows · {dataset.columnCount} cols
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200">
              <span className="text-slate-400 print:text-slate-500 block">Task Classification</span>
              <span className="font-bold text-cyan-300 print:text-blue-800 uppercase">
                {dataset.problemDetection.detectedProblem}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200">
              <span className="text-slate-400 print:text-slate-500 block">Quality Audit</span>
              <span className="font-bold text-emerald-400 print:text-emerald-700 tabular-nums">
                {dataset.qualityReport.score} / 100
              </span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/40 print:bg-slate-50 border border-slate-800/80 print:border-slate-200 text-xs space-y-1.5">
            <span className="text-slate-400 font-semibold block">Quality Findings:</span>
            {dataset.qualityReport.reasons.map((r, i) => (
              <div key={i} className="flex items-start gap-2 text-slate-300 print:text-slate-800">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <span>{r}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Section 3: Evaluated Candidate Benchmark */}
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-wider text-cyan-400 print:text-blue-700 font-bold">
            03. Model Arena & Multi-Objective Benchmark
          </h3>

          <div className="overflow-x-auto border border-slate-800 print:border-slate-300 rounded-lg">
            <table className="w-full text-left text-xs text-slate-300 print:text-black">
              <thead className="bg-slate-950 print:bg-slate-100 text-[11px] uppercase tracking-wider text-slate-400 print:text-slate-700 border-b border-slate-800 print:border-slate-300">
                <tr>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">{isClass ? 'Accuracy' : 'R² Score'}</th>
                  <th className="py-2.5 px-3">{isClass ? 'F1-Score' : 'RMSE'}</th>
                  <th className="py-2.5 px-3">Inference Latency</th>
                  <th className="py-2.5 px-3">Pareto Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 print:divide-slate-200 bg-slate-950/20">
                {models.map((m) => (
                  <tr key={m.id}>
                    <td className="py-2 px-3 font-semibold text-white print:text-black">
                      {m.name}
                    </td>
                    <td className="py-2 px-3 font-mono tabular-nums">
                      {isClass ? `${(m.metrics.accuracy! * 100).toFixed(1)}%` : m.metrics.r2}
                    </td>
                    <td className="py-2 px-3 font-mono tabular-nums">
                      {isClass ? `${(m.metrics.f1! * 100).toFixed(1)}%` : m.metrics.rmse}
                    </td>
                    <td className="py-2 px-3 font-mono tabular-nums">
                      {m.metrics.inferenceLatencyMs} ms
                    </td>
                    <td className="py-2 px-3">
                      {m.isParetoOptimal ? (
                        <span className="text-emerald-400 print:text-emerald-700 font-semibold">
                          Pareto Winner
                        </span>
                      ) : (
                        <span className="text-slate-500">Dominated</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 4: Anomaly Detection Summary */}
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-wider text-cyan-400 print:text-blue-700 font-bold">
            04. Anomaly Radar & Outlier Diagnostics
          </h3>
          <p className="text-xs text-slate-300 print:text-slate-800">
            Isolation Forest analyzed multidimensional path lengths, flagging{' '}
            <strong className="text-white print:text-black">{anomalies?.anomalyCount || 0} instances</strong>{' '}
            ({anomalies?.anomalyRate || 0}% contamination rate). Quarantining outlier instances is recommended before deploying real-time retraining jobs.
          </p>
        </div>

        {/* Section 5: Strategic Prescriptive Recommendations */}
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-wider text-cyan-400 print:text-blue-700 font-bold">
            05. Prescriptive Deployment Recommendations
          </h3>
          <div className="space-y-2">
            {(report?.recommendations || [
              `Deploy ${bestModel?.name || 'XGBoost'} for mission-critical batch inference where highest statistical accuracy is demanded.`,
              `Deploy Pareto-optimal low-latency candidate for edge microservices with sub-5ms SLA requirements.`,
              `Establish automated drift detection monitoring on top permutation drivers.`,
              `Perform automated human-in-the-loop audit on the flagged ${anomalies?.anomalyCount || 5} outlier transactions.`,
            ]).map((rec, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-slate-950/60 print:bg-slate-50 border border-slate-800 print:border-slate-200 text-xs flex items-start gap-2.5"
              >
                <CheckCircle2 className="w-4 h-4 text-cyan-400 print:text-blue-600 shrink-0 mt-0.5" />
                <span className="text-slate-200 print:text-slate-900">{rec}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Report Footer */}
        <div className="pt-6 border-t border-slate-800 print:border-slate-300 flex items-center justify-between text-[11px] text-slate-400 print:text-slate-500">
          <span>EVOX — Adaptive AI Intelligence Platform</span>
          <span>Validated on Production Monorepo Stack</span>
        </div>
      </div>
    </div>
  );
};
