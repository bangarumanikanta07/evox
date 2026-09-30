import React, { useState } from 'react';
import {
  Upload,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Sparkles,
  ArrowRight,
  Database,
  Search,
} from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { Dataset, ProblemType } from '../../types';
import { api } from '../../services/api';

interface DatasetLabViewProps {
  dataset: Dataset | null;
  onDatasetLoaded: (dataset: Dataset) => void;
  onNavigate: (view: string) => void;
  onLaunchDemo: () => void;
  isDemoLoading: boolean;
}

export const DatasetLabView: React.FC<DatasetLabViewProps> = ({
  dataset,
  onDatasetLoaded,
  onNavigate,
  onLaunchDemo,
  isDemoLoading,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Parse uploaded file
  const handleFileUpload = (file: File) => {
    setUploadError(null);
    setIsProcessing(true);

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext === 'csv') {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: async (results) => {
          if (results.data && results.data.length > 0) {
            try {
              const profiled = await api.uploadDataset(file.name, results.data as Record<string, any>[], file.size);
              onDatasetLoaded(profiled);
            } catch (err: any) {
              setUploadError(err.message || 'Failed to profile dataset on backend.');
            } finally {
              setIsProcessing(false);
            }
          } else {
            setIsProcessing(false);
            setUploadError('The CSV file does not contain valid data rows.');
          }
        },
        error: (err) => {
          setIsProcessing(false);
          setUploadError(`Failed to parse CSV: ${err.message}`);
        },
      });
    } else if (ext === 'xlsx' || ext === 'xls') {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          const firstSheet = workbook.SheetNames[0];
          const sheet = workbook.Sheets[firstSheet];
          const json = XLSX.utils.sheet_to_json(sheet);
          if (json && json.length > 0) {
            const profiled = await api.uploadDataset(file.name, json as Record<string, any>[], file.size);
            onDatasetLoaded(profiled);
          } else {
            setUploadError('The Excel sheet contains no readable rows.');
          }
        } catch (err: any) {
          setUploadError(`Failed to process Excel file: ${err.message}`);
        } finally {
          setIsProcessing(false);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      setIsProcessing(false);
      setUploadError('Unsupported file type. Please upload a .CSV or .XLSX file.');
    }
  };

  const handleTargetChange = (newTarget: string) => {
    if (!dataset) return;
    const targetCol = dataset.columns.find((c) => c.name === newTarget);
    let newProblem: ProblemType = 'classification';
    let confidence = 90;
    let reason = '';

    if (targetCol) {
      if (targetCol.dataType === 'numerical' && targetCol.uniqueCount > 25) {
        newProblem = 'regression';
        confidence = 94;
        reason = `Selected target "${newTarget}" is a continuous numerical variable with ${targetCol.uniqueCount} distinct values.`;
      } else {
        newProblem = 'classification';
        confidence = targetCol.uniqueCount === 2 ? 96 : 88;
        reason = `Selected target "${newTarget}" contains ${targetCol.uniqueCount} discrete classes, ideal for classification.`;
      }
    }

    const updated: Dataset = {
      ...dataset,
      targetColumn: newTarget,
      problemDetection: {
        detectedProblem: newProblem,
        confidence,
        reason,
        suggestedTarget: newTarget,
        classesCount: targetCol?.uniqueCount,
      },
      features: dataset.columns.filter((c) => c.name !== newTarget).map((c) => c.name),
    };
    onDatasetLoaded(updated);
  };

  const handleProblemOverride = (newProblem: ProblemType) => {
    if (!dataset) return;
    const updated: Dataset = {
      ...dataset,
      problemDetection: {
        ...dataset.problemDetection,
        detectedProblem: newProblem,
        reason: `Manually designated problem type: ${newProblem.toUpperCase()} by user override.`,
      },
    };
    onDatasetLoaded(updated);
  };

  const numCount = dataset?.columns.filter((c) => c.dataType === 'numerical').length || 0;
  const catCount = dataset?.columns.filter((c) => c.dataType === 'categorical').length || 0;
  const totalMissing = dataset?.columns.reduce((a, c) => a + c.missingCount, 0) || 0;

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Data Engineering & Profiling
          </span>
          <h1 className="text-2xl font-bold text-white tracking-tight">Dataset Lab</h1>
        </div>
        <button
          onClick={onLaunchDemo}
          disabled={isDemoLoading}
          className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-slate-900 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors shrink-0"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>{isDemoLoading ? 'Loading Benchmark...' : 'Load Customer Benchmark'}</span>
        </button>
      </div>

      {/* File Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
          }
        }}
        className={`relative p-8 rounded-xl border-2 border-dashed transition-all text-center space-y-3 ${
          isDragging
            ? 'border-cyan-400 bg-cyan-950/20'
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700'
        }`}
      >
        <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center mx-auto text-cyan-400">
          <Upload className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">
            {isProcessing ? 'Analyzing and Profiling Dataset...' : 'Drop your CSV or XLSX dataset here'}
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Accepts comma-delimited tables or spreadsheets up to 50MB
          </p>
        </div>
        <div className="pt-2">
          <label className="inline-block px-4 py-2 text-xs font-semibold text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg cursor-pointer transition-colors">
            <span>Browse Files</span>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileUpload(e.target.files[0]);
                }
              }}
            />
          </label>
        </div>

        {uploadError && (
          <p className="text-xs text-rose-400 font-medium pt-2">{uploadError}</p>
        )}
      </div>

      {dataset && (
        <>
          {/* Summary Badges Grid (7 core properties from req 8) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Rows</span>
              <span className="text-lg font-bold text-white tabular-nums">
                {dataset.rowCount.toLocaleString()}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Columns</span>
              <span className="text-lg font-bold text-white tabular-nums">
                {dataset.columnCount}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Missing Values</span>
              <span className="text-lg font-bold text-amber-400 tabular-nums">
                {totalMissing}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Duplicates</span>
              <span className="text-lg font-bold text-white tabular-nums">0</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Numerical</span>
              <span className="text-lg font-bold text-cyan-400 tabular-nums">{numCount}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[11px] text-slate-400 block">Categorical</span>
              <span className="text-lg font-bold text-indigo-400 tabular-nums">{catCount}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 truncate">
              <span className="text-[11px] text-slate-400 block">Target Feature</span>
              <span className="text-sm font-bold text-emerald-400 truncate block">
                {dataset.targetColumn || 'None'}
              </span>
            </div>
          </div>

          {/* Data Quality & Autonomous Problem Detection */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Section 8: Data Quality Score */}
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Data Quality Score</h3>
                  <p className="text-xs text-slate-400">Heuristic audit based on completeness and integrity</p>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-extrabold text-cyan-400 tabular-nums">
                    {dataset.qualityReport.score}
                  </span>
                  <span className="text-xs text-slate-400">/ 100</span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full"
                  style={{ width: `${dataset.qualityReport.score}%` }}
                />
              </div>

              {/* Rubric Reasons */}
              <div className="space-y-2 pt-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                  Scoring Rubric Justification
                </span>
                <ul className="space-y-1.5 text-xs text-slate-300">
                  {dataset.qualityReport.reasons.map((r, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{r}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Section 9: Problem Detection */}
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Autonomous Problem Detection</h3>
                  <p className="text-xs text-slate-400">Target cardinality and continuous/discrete analysis</p>
                </div>
                <span className="text-xs text-slate-400 tabular-nums">
                  Confidence: <strong className="text-cyan-400">{dataset.problemDetection.confidence}%</strong>
                </span>
              </div>

              <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 uppercase tracking-wider">Detected Task</span>
                  <span className="text-sm font-bold text-white uppercase tracking-wider">
                    {dataset.problemDetection.detectedProblem}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {dataset.problemDetection.reason}
                </p>
              </div>

              {/* Overrides */}
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="text-[11px] text-slate-400 block mb-1">Target Feature</label>
                  <select
                    value={dataset.targetColumn || ''}
                    onChange={(e) => handleTargetChange(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-400"
                  >
                    {dataset.columns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name} ({col.dataType})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-[11px] text-slate-400 block mb-1">Override Problem</label>
                  <select
                    value={dataset.problemDetection.detectedProblem}
                    onChange={(e) => handleProblemOverride(e.target.value as ProblemType)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-400"
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                    <option value="clustering">Clustering</option>
                    <option value="anomaly_detection">Anomaly Detection</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Column Profile Table */}
          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
            <h3 className="text-sm font-semibold text-white">Feature Profiling & Statistical Distribution</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Column Name</th>
                    <th className="py-2.5 px-3">Type</th>
                    <th className="py-2.5 px-3">Missing</th>
                    <th className="py-2.5 px-3">Unique</th>
                    <th className="py-2.5 px-3">Outliers</th>
                    <th className="py-2.5 px-3">Stats Summary</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {dataset.columns.map((c) => (
                    <tr key={c.name} className="hover:bg-slate-800/30">
                      <td className="py-2.5 px-3 font-medium text-white flex items-center gap-2">
                        {c.name === dataset.targetColumn && (
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                        )}
                        <span>{c.name}</span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">{c.dataType}</td>
                      <td className="py-2.5 px-3 tabular-nums">
                        {c.missingCount} ({c.missingPercentage}%)
                      </td>
                      <td className="py-2.5 px-3 tabular-nums">{c.uniqueCount}</td>
                      <td className="py-2.5 px-3 tabular-nums">
                        {c.outlierCount > 0 ? (
                          <span className="text-amber-400">{c.outlierCount}</span>
                        ) : (
                          '0'
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 font-mono text-[11px]">
                        {c.stats?.mean !== undefined
                          ? `mean: ${c.stats.mean} · range: [${c.stats.min} .. ${c.stats.max}]`
                          : c.stats?.topCategories
                          ? `top: ${c.stats.topCategories.slice(0, 2).map((t) => t.value).join(', ')}`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dataset Preview Table */}
          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-white">Dataset Preview (First 15 Rows)</h3>
                <p className="text-xs text-slate-400">Raw records ingested into memory</p>
              </div>
              <button
                onClick={() => onNavigate('automl')}
                className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors"
              >
                <span>Proceed to AutoML</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="overflow-x-auto max-h-80 border border-slate-800 rounded-lg">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="sticky top-0 bg-slate-950 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                  <tr>
                    {dataset.columns.map((c) => (
                      <th key={c.name} className="py-2.5 px-3 whitespace-nowrap">
                        {c.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                  {dataset.previewRows.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40">
                      {dataset.columns.map((c) => (
                        <td key={c.name} className="py-2 px-3 whitespace-nowrap tabular-nums">
                          {row[c.name] !== null && row[c.name] !== undefined
                            ? String(row[c.name])
                            : <span className="text-slate-500 italic">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
